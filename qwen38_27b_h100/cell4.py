"""Cell 4: H100 高并发 OpenAI 兼容薄代理网关 (Port 8000 -> vLLM Port 8001)。

特性:
- 彻底移除旧单线程锁 (G4FairStepScheduler)，通过 httpx.AsyncClient 连接池异步并发透传，
  完全释放 vLLM Continuous Batching + Prefix Caching 真并发。
- 完整兼容 OpenCode / Pi Agent：developer->system 角色映射、多模态 image_url 透传、
  reasoning_effort 统一映射到 chat_template_kwargs、思考模式 max_tokens 下限与 Qwen3.8 官方采样参数。
- 流式 SSE 保持 `: keep-alive` 心跳防断连，自动补齐 `reasoning_content`，并在客户端断连时立即中止上游。
"""

import os
import re
import time
import json
import uuid
import asyncio
import logging
import threading
import contextlib
import urllib.request
from contextlib import asynccontextmanager

import httpx

try:
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
except ImportError:  # 允许在未安装 fastapi/uvicorn 的纯单测环境导入纯逻辑函数
    uvicorn = None
    FastAPI = None
    Request = None
    CORSMiddleware = None
    JSONResponse = None
    PlainTextResponse = None
    StreamingResponse = None

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.getLogger("asyncio").setLevel(logging.CRITICAL)
logging.getLogger("uvicorn.error").setLevel(logging.CRITICAL)

# ==========================================================
# 0. 全局共享契约与网关配置
# ==========================================================
GATEWAY_HOST = "0.0.0.0"
GATEWAY_PORT = 8000
GATEWAY_BASE_URL = f"http://127.0.0.1:{GATEWAY_PORT}/v1"

VLLM_INTERNAL_PORT = int(globals().get("VLLM_INTERNAL_PORT", 8001))
VLLM_BASE_URL = str(globals().get("VLLM_BASE_URL", f"http://127.0.0.1:{VLLM_INTERNAL_PORT}/v1")).rstrip("/")
SERVED_MODEL_NAME = str(globals().get("SERVED_MODEL_NAME", "Qwen/Qwen3.8-27B"))
ACTIVE_MODEL_ID = str(globals().get("ACTIVE_MODEL_ID", SERVED_MODEL_NAME))
MODEL_PRECISION = str(globals().get("MODEL_PRECISION", "bf16"))

DEFAULT_REASONING_EFFORT = "medium"
FORCE_XHIGH_WHEN_THINKING = False
MIN_THINKING_MAX_TOKENS = 16384
MIN_XHIGH_MAX_TOKENS = 32768
SSE_KEEPALIVE_INTERVAL = 2.5

TAG_THINK_OPEN = "<" + "think>"
TAG_THINK_CLOSE = "</" + "think>"
REGEX_THINK_BLOCK = re.compile(re.escape(TAG_THINK_OPEN) + r"(.*?)" + re.escape(TAG_THINK_CLOSE), flags=re.DOTALL)
REGEX_MAX_CTX_LEN = re.compile(r"maximum context length is (\d+) tokens", flags=re.IGNORECASE)
REGEX_PROMPT_TOKENS_IN_ERR = re.compile(r"(\d+)\s+in the (?:messages|input)", flags=re.IGNORECASE)

if "LAST_API_REQUEST_TIME" not in globals():
    LAST_API_REQUEST_TIME = time.time()

uvicorn_server = globals().get("uvicorn_server", None)


class H100SessionTracker:
    """线程安全的高并发活跃会话与请求计数器（供网关与 Cell 5 监控共享）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active_sessions = 0
        self._total_requests = 0

    @property
    def active_sessions(self) -> int:
        with self._lock:
            return self._active_sessions

    @property
    def total_requests(self) -> int:
        with self._lock:
            return self._total_requests

    def inc(self) -> int:
        with self._lock:
            self._active_sessions += 1
            self._total_requests += 1
            return self._active_sessions

    def dec(self) -> int:
        with self._lock:
            self._active_sessions = max(0, self._active_sessions - 1)
            return self._active_sessions


h100_gateway_tracker = H100SessionTracker()


# ==========================================================
# 1. 纯逻辑标准化函数（与网络/框架解耦，便于单元测试）
# ==========================================================
def _normalize_single_message(msg: dict) -> dict:
    """标准化单条 OpenAI 消息，保留多模态 image_url 并转换 developer->system。"""
    out = dict(msg)
    role = out.get("role", "user")
    if role == "developer":
        role = "system"
    out["role"] = role

    extracted_reasoning = (
        out.get("reasoning_content")
        or out.get("reasoning")
        or out.get("reasoning_text")
    )

    content = out.get("content", "")
    if isinstance(content, list):
        new_parts = []
        reasoning_parts = []
        has_mm = False
        text_parts = []

        for part in content:
            if isinstance(part, dict):
                p_type = part.get("type", "")
                if p_type in ("text", "input_text", "output_text"):
                    txt = str(part.get("text", ""))
                    text_parts.append(txt)
                    new_parts.append({"type": "text", "text": txt})
                elif p_type in ("image_url", "input_image", "image"):
                    has_mm = True
                    if "image_url" in part:
                        new_parts.append({"type": "image_url", "image_url": part["image_url"]})
                    elif "image" in part:
                        img_val = part["image"]
                        url_obj = img_val if isinstance(img_val, dict) else {"url": str(img_val)}
                        new_parts.append({"type": "image_url", "image_url": url_obj})
                    else:
                        new_parts.append(dict(part))
                elif p_type in ("reasoning", "thinking"):
                    r_txt = part.get("text") or part.get("thinking") or part.get("reasoning") or ""
                    if r_txt:
                        reasoning_parts.append(str(r_txt))
                elif "content" in part:
                    txt = str(part.get("content", ""))
                    text_parts.append(txt)
                    new_parts.append({"type": "text", "text": txt})
                else:
                    new_parts.append(dict(part))
            elif isinstance(part, str):
                text_parts.append(part)
                new_parts.append({"type": "text", "text": part})

        out["content"] = new_parts if has_mm else "\n".join(text_parts)
        if reasoning_parts and not extracted_reasoning:
            extracted_reasoning = "\n".join(reasoning_parts)
    elif content is None:
        out["content"] = "" if not out.get("tool_calls") else None
    elif not isinstance(content, str):
        out["content"] = json.dumps(content, ensure_ascii=False)

    if role == "assistant" and isinstance(out.get("content"), str):
        c_str = out["content"]
        if TAG_THINK_OPEN in c_str and TAG_THINK_CLOSE in c_str:
            m_think = REGEX_THINK_BLOCK.search(c_str)
            if m_think:
                if not extracted_reasoning:
                    extracted_reasoning = m_think.group(1).strip()
                out["content"] = (c_str[: m_think.start()] + c_str[m_think.end() :]).strip()

    if role == "assistant" and extracted_reasoning and str(extracted_reasoning).strip():
        out["reasoning_content"] = str(extracted_reasoning).strip()

    return out


def normalize_chat_request_payload(
    body: dict,
    default_model: str = SERVED_MODEL_NAME,
    default_reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    force_xhigh_when_thinking: bool = FORCE_XHIGH_WHEN_THINKING,
    min_thinking_max_tokens: int = MIN_THINKING_MAX_TOKENS,
    min_xhigh_max_tokens: int = MIN_XHIGH_MAX_TOKENS,
) -> dict:
    """将客户端 OpenAI / OpenCode / Pi Agent 请求标准化为 H100 vLLM 最优参数格式。"""
    payload = dict(body)
    payload["model"] = default_model

    raw_messages = payload.get("messages") or []
    payload["messages"] = [_normalize_single_message(m) for m in raw_messages]

    existing_kwargs = dict(payload.get("chat_template_kwargs") or {})
    preserve_thinking = payload.pop(
        "preserve_thinking",
        existing_kwargs.get("preserve_thinking", True),
    )

    reasoning_obj = payload.pop("reasoning", None)
    thinking_level = payload.pop("thinking_level", None)
    explicit_enable_thinking = payload.pop("enable_thinking", None)

    raw_effort = None
    for candidate in (
        payload.get("reasoning_effort"),
        reasoning_obj.get("effort") if isinstance(reasoning_obj, dict) else None,
        thinking_level,
        existing_kwargs.get("reasoning_effort"),
    ):
        if candidate is not None and candidate != "":
            raw_effort = candidate
            break

    effort_map = {
        "off": None,
        "none": None,
        "false": None,
        "disabled": None,
        "minimal": "low",
        "low": "low",
        "medium": "medium",
        "high": "xhigh",
        "xhigh": "xhigh",
        "max": "xhigh",
    }

    if raw_effort is not None:
        raw_key = str(raw_effort).strip().lower()
        if raw_key in effort_map:
            mapped_effort = effort_map[raw_key]
        else:
            mapped_effort = default_reasoning_effort
        enable_thinking = mapped_effort is not None
        reasoning_effort = mapped_effort or default_reasoning_effort
    else:
        if explicit_enable_thinking is not None:
            enable_thinking = bool(explicit_enable_thinking)
        elif "enable_thinking" in existing_kwargs:
            enable_thinking = bool(existing_kwargs["enable_thinking"])
        else:
            enable_thinking = True
        reasoning_effort = default_reasoning_effort

    if enable_thinking and force_xhigh_when_thinking:
        reasoning_effort = "xhigh"

    existing_kwargs["enable_thinking"] = bool(enable_thinking)
    existing_kwargs["preserve_thinking"] = bool(preserve_thinking)
    if enable_thinking:
        existing_kwargs["reasoning_effort"] = reasoning_effort
        payload["reasoning_effort"] = reasoning_effort
    else:
        existing_kwargs.pop("reasoning_effort", None)
        payload.pop("reasoning_effort", None)

    payload["chat_template_kwargs"] = existing_kwargs

    req_max_tokens = payload.get("max_tokens") or payload.pop("max_completion_tokens", None) or 16384
    payload.pop("max_completion_tokens", None)
    if enable_thinking:
        min_tok = min_xhigh_max_tokens if reasoning_effort == "xhigh" else min_thinking_max_tokens
        payload["max_tokens"] = max(int(req_max_tokens), int(min_tok))
    else:
        payload["max_tokens"] = int(req_max_tokens)

    # 填充 Qwen3.8 官方推荐默认采样参数（绝不覆盖客户端显式传入值）
    if enable_thinking:
        if payload.get("temperature") is None:
            payload["temperature"] = 1.0
        if payload.get("top_p") is None:
            payload["top_p"] = 0.95
        if payload.get("top_k") is None:
            payload["top_k"] = 20
    else:
        if payload.get("temperature") is None:
            payload["temperature"] = 0.7
        if payload.get("top_p") is None:
            payload["top_p"] = 0.80
        if payload.get("top_k") is None:
            payload["top_k"] = 20
        if payload.get("repetition_penalty") is None:
            payload["repetition_penalty"] = 1.05

    if payload.get("stream"):
        stream_opts = dict(payload.get("stream_options") or {})
        stream_opts.setdefault("include_usage", True)
        payload["stream_options"] = stream_opts

    return payload


def normalize_response_message(msg: dict) -> dict:
    """确保非流式响应 message 在仅含 reasoning 时自动补齐 reasoning_content。"""
    if not isinstance(msg, dict):
        return msg
    out = dict(msg)
    if out.get("reasoning_content") is None and out.get("reasoning") is not None:
        out["reasoning_content"] = out["reasoning"]
    return out


def normalize_non_stream_response(resp_dict: dict) -> dict:
    """标准化非流式 ChatCompletion 响应字典。"""
    if not isinstance(resp_dict, dict):
        return resp_dict
    out = dict(resp_dict)
    choices = out.get("choices")
    if isinstance(choices, list):
        new_choices = []
        for choice in choices:
            if isinstance(choice, dict):
                c_copy = dict(choice)
                if isinstance(c_copy.get("message"), dict):
                    c_copy["message"] = normalize_response_message(c_copy["message"])
                new_choices.append(c_copy)
            else:
                new_choices.append(choice)
        out["choices"] = new_choices
    return out


def normalize_sse_chunk(chunk_dict: dict) -> dict:
    """确保流式 SSE delta 块在仅含 reasoning 时自动补齐 reasoning_content。"""
    if not isinstance(chunk_dict, dict):
        return chunk_dict
    out = dict(chunk_dict)
    choices = out.get("choices")
    if isinstance(choices, list):
        new_choices = []
        for choice in choices:
            if isinstance(choice, dict):
                c_copy = dict(choice)
                delta = c_copy.get("delta")
                if isinstance(delta, dict):
                    d_copy = dict(delta)
                    if d_copy.get("reasoning_content") is None and d_copy.get("reasoning") is not None:
                        d_copy["reasoning_content"] = d_copy["reasoning"]
                    c_copy["delta"] = d_copy
                new_choices.append(c_copy)
            else:
                new_choices.append(choice)
        out["choices"] = new_choices
    return out


def compute_clamped_max_tokens_from_400(
    err_text: str,
    current_max_tokens: int,
    safety_margin: int = 64,
) -> int | None:
    """解析 vLLM 上下文超限 400 错误并计算安全截断后的 max_tokens；无法缩减则返回 None。"""
    if not err_text:
        return None
    m_ctx = REGEX_MAX_CTX_LEN.search(err_text)
    m_prompt = REGEX_PROMPT_TOKENS_IN_ERR.search(err_text)
    if not m_ctx or not m_prompt:
        return None

    max_len = int(m_ctx.group(1))
    prompt_tok = int(m_prompt.group(1))
    available = max_len - prompt_tok - int(safety_margin)
    if available >= 128 and available < int(current_max_tokens):
        return available
    return None


# ==========================================================
# 2. FastAPI 异步薄代理网关应用
# ==========================================================
def create_gateway_app():
    if FastAPI is None:
        return None

    @asynccontextmanager
    async def lifespan(fastapi_app: FastAPI):
        limits = httpx.Limits(max_connections=64, max_keepalive_connections=32)
        timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)
        fastapi_app.state.http_client = httpx.AsyncClient(limits=limits, timeout=timeout)
        try:
            yield
        finally:
            await fastapi_app.state.http_client.aclose()

    fastapi_app = FastAPI(
        title="Qwen3.8-27B H100 vLLM High-Concurrency Gateway",
        version="1.0",
        lifespan=lifespan,
    )
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    async def get_client() -> httpx.AsyncClient:
        client = getattr(fastapi_app.state, "http_client", None)
        if client is None or client.is_closed:
            limits = httpx.Limits(max_connections=64, max_keepalive_connections=32)
            timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)
            client = httpx.AsyncClient(limits=limits, timeout=timeout)
            fastapi_app.state.http_client = client
        return client

    @fastapi_app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        globals()["LAST_API_REQUEST_TIME"] = time.time()
        body = await request.json()
        requested_model = body.get("model") or SERVED_MODEL_NAME
        normalized_payload = normalize_chat_request_payload(body, default_model=SERVED_MODEL_NAME)
        is_stream = bool(normalized_payload.get("stream", False))
        req_id = uuid.uuid4().hex[:6]

        client = await get_client()
        upstream_url = f"{VLLM_BASE_URL}/chat/completions"

        # ---------------- 分支 A：非流式并发透传 ----------------
        if not is_stream:
            active_now = h100_gateway_tracker.inc()
            post_task = None
            try:
                for attempt in range(2):
                    post_task = asyncio.create_task(client.post(upstream_url, json=normalized_payload))
                    while not post_task.done():
                        if await request.is_disconnected():
                            post_task.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await post_task
                            print(f"⏹️ [{req_id} 客户端已断开] 已立即中止非流式上游请求。")
                            return JSONResponse({"error": {"message": "Client disconnected"}}, status_code=499)
                        await asyncio.wait({post_task}, timeout=0.5)

                    upstream_resp = await post_task
                    post_task = None

                    if upstream_resp.status_code == 400 and attempt == 0:
                        clamped = compute_clamped_max_tokens_from_400(
                            upstream_resp.text,
                            normalized_payload.get("max_tokens", 16384),
                        )
                        if clamped is not None:
                            print(
                                f"✂️ [{req_id}] 上下文超限 400，自动截断 max_tokens: "
                                f"{normalized_payload['max_tokens']} -> {clamped} 并透明重试..."
                            )
                            normalized_payload["max_tokens"] = clamped
                            continue

                    if upstream_resp.status_code >= 400:
                        try:
                            err_json = upstream_resp.json()
                        except Exception:
                            err_json = {"error": {"message": upstream_resp.text}}
                        return JSONResponse(err_json, status_code=upstream_resp.status_code)

                    resp_json = normalize_non_stream_response(upstream_resp.json())
                    return JSONResponse(resp_json, status_code=upstream_resp.status_code)
            except httpx.RequestError as exc:
                return JSONResponse(
                    {"error": {"message": f"vLLM upstream unreachable ({upstream_url}): {exc}"}},
                    status_code=502,
                )
            finally:
                if post_task is not None and not post_task.done():
                    post_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await post_task
                globals()["LAST_API_REQUEST_TIME"] = time.time()
                remaining = h100_gateway_tracker.dec()
                _ = active_now, remaining

        # ---------------- 分支 B：流式 SSE 并发透传 ----------------
        async def sse_event_generator():
            active_now = h100_gateway_tracker.inc()
            stream_ctx = None
            entered_resp = None
            enter_task = None
            next_line_task = None

            try:
                err_text_400 = None
                for attempt in range(2):
                    stream_ctx = client.stream("POST", upstream_url, json=normalized_payload)
                    enter_task = asyncio.create_task(stream_ctx.__aenter__())
                    while not enter_task.done():
                        if await request.is_disconnected():
                            enter_task.cancel()
                            try:
                                entered_resp = await enter_task
                            except (asyncio.CancelledError, Exception):
                                entered_resp = None
                            enter_task = None
                            print(f"⏹️ [{req_id} 客户端建连阶段断开] 已中止上游流式请求。")
                            return
                        done, _ = await asyncio.wait({enter_task}, timeout=SSE_KEEPALIVE_INTERVAL)
                        if not done:
                            yield ": keep-alive\n\n"

                    entered_resp = await enter_task
                    enter_task = None

                    if entered_resp.status_code == 400 and attempt == 0:
                        err_bytes = await entered_resp.aread()
                        err_text_400 = err_bytes.decode("utf-8", errors="replace")
                        clamped = compute_clamped_max_tokens_from_400(
                            err_text_400,
                            normalized_payload.get("max_tokens", 16384),
                        )
                        if clamped is not None:
                            with contextlib.suppress(Exception):
                                await stream_ctx.__aexit__(None, None, None)
                            entered_resp = None
                            stream_ctx = None
                            print(
                                f"✂️ [{req_id}] 流式上下文超限 400，自动截断 max_tokens: "
                                f"{normalized_payload['max_tokens']} -> {clamped} 并透明重试..."
                            )
                            normalized_payload["max_tokens"] = clamped
                            continue
                    break

                if entered_resp.status_code >= 400:
                    if err_text_400 is None:
                        err_bytes = await entered_resp.aread()
                        err_text_400 = err_bytes.decode("utf-8", errors="replace")
                    err_payload = {"error": {"message": err_text_400, "status": entered_resp.status_code}}
                    yield f"data: {json.dumps(err_payload, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                lines_iter = entered_resp.aiter_lines()
                while True:
                    if await request.is_disconnected():
                        print(f"⏹️ [{req_id} 客户端主动断开] 已立即关闭 vLLM 上游连接中止生成 (并发={active_now})。")
                        return

                    next_line_task = asyncio.create_task(lines_iter.__anext__())
                    while not next_line_task.done():
                        if await request.is_disconnected():
                            next_line_task.cancel()
                            with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration, Exception):
                                await next_line_task
                            next_line_task = None
                            print(f"⏹️ [{req_id} 客户端主动断开] 已立即关闭 vLLM 上游连接中止生成。")
                            return
                        done, _ = await asyncio.wait({next_line_task}, timeout=SSE_KEEPALIVE_INTERVAL)
                        if not done:
                            yield ": keep-alive\n\n"

                    try:
                        line = await next_line_task
                    except StopAsyncIteration:
                        next_line_task = None
                        break
                    next_line_task = None

                    if not line:
                        continue
                    globals()["LAST_API_REQUEST_TIME"] = time.time()

                    if line.startswith(":"):
                        yield f"{line}\n\n"
                        continue

                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            yield "data: [DONE]\n\n"
                            continue
                        try:
                            chunk_obj = json.loads(data_str)
                            norm_chunk = normalize_sse_chunk(chunk_obj)
                            yield f"data: {json.dumps(norm_chunk, ensure_ascii=False)}\n\n"
                        except json.JSONDecodeError:
                            yield f"{line}\n\n"

            except (asyncio.CancelledError, BrokenPipeError, ConnectionResetError):
                print(f"⏹️ [{req_id} 流式连接中断] 已安全回收上游连接。")
            except httpx.RequestError as exc:
                err_payload = {"error": {"message": f"vLLM upstream error ({requested_model}): {exc}"}}
                yield f"data: {json.dumps(err_payload, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                for pending_task in (enter_task, next_line_task):
                    if pending_task is not None and not pending_task.done():
                        pending_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration, Exception):
                            await pending_task
                if entered_resp is not None and stream_ctx is not None:
                    with contextlib.suppress(Exception):
                        await stream_ctx.__aexit__(None, None, None)
                globals()["LAST_API_REQUEST_TIME"] = time.time()
                h100_gateway_tracker.dec()

        return StreamingResponse(
            sse_event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @fastapi_app.get("/v1/models")
    @fastapi_app.get("/models")
    async def list_models():
        fallback = {
            "object": "list",
            "data": [
                {
                    "id": SERVED_MODEL_NAME,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "qwen",
                    "aliases": [ACTIVE_MODEL_ID, f"qwen3.8-27b-{MODEL_PRECISION}"],
                }
            ],
        }
        try:
            client = await get_client()
            resp = await client.get(f"{VLLM_BASE_URL}/models", timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict) and isinstance(data.get("data"), list) and data["data"]:
                    for item in data["data"]:
                        if isinstance(item, dict) and "aliases" not in item:
                            item["aliases"] = [ACTIVE_MODEL_ID, f"qwen3.8-27b-{MODEL_PRECISION}"]
                    return data
        except Exception:
            pass
        return fallback

    @fastapi_app.get("/health")
    async def health():
        vllm_healthy = False
        try:
            client = await get_client()
            resp = await client.get(f"http://127.0.0.1:{VLLM_INTERNAL_PORT}/health", timeout=3.0)
            vllm_healthy = resp.status_code == 200
        except Exception:
            vllm_healthy = False

        status_code = 200 if vllm_healthy else 503
        return JSONResponse(
            {
                "status": "ok" if vllm_healthy else "degraded",
                "vllm_healthy": vllm_healthy,
                "active_sessions": h100_gateway_tracker.active_sessions,
                "total_requests": h100_gateway_tracker.total_requests,
                "served_model": SERVED_MODEL_NAME,
                "active_model_id": ACTIVE_MODEL_ID,
                "precision": MODEL_PRECISION,
            },
            status_code=status_code,
        )

    @fastapi_app.get("/v1/metrics")
    async def metrics():
        try:
            client = await get_client()
            resp = await client.get(f"http://127.0.0.1:{VLLM_INTERNAL_PORT}/metrics", timeout=5.0)
            return PlainTextResponse(resp.text, status_code=resp.status_code)
        except Exception as exc:
            return PlainTextResponse(f"# vLLM metrics unavailable: {exc}\n", status_code=502)

    return fastapi_app


app = create_gateway_app()


def start_gateway_server():
    """优雅重启并启动 H100 高并发 FastAPI 网关（Port 8000）。"""
    if FastAPI is None or uvicorn is None or app is None:
        raise ImportError("❌ 缺少 fastapi / uvicorn 依赖，请先运行 Cell 1 安装依赖。")

    prev_server = globals().get("uvicorn_server")
    if prev_server is not None:
        print("🔄 检测到旧网关实例，正在优雅关闭并重启...")
        prev_server.should_exit = True
        time.sleep(1.5)

    config = uvicorn.Config(
        app,
        host=GATEWAY_HOST,
        port=GATEWAY_PORT,
        log_level="critical",
        access_log=False,
        timeout_keep_alive=120,
    )
    server = uvicorn.Server(config)

    def _run_isolated_uvicorn():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve())

    threading.Thread(target=_run_isolated_uvicorn, daemon=True, name="h100_gateway_uvicorn").start()

    # 短暂等待端口监听就绪
    deadline = time.time() + 10.0
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{GATEWAY_PORT}/v1/models", timeout=1.0) as resp:
                if resp.status == 200:
                    break
        except Exception:
            time.sleep(0.2)

    return server


if __name__ == "__main__":
    globals()["LAST_API_REQUEST_TIME"] = time.time()
    uvicorn_server = start_gateway_server()
    print(
        f"✅ H100 高并发 OpenAI 兼容网关已启动 ({GATEWAY_BASE_URL} -> {VLLM_BASE_URL} | "
        f"模型={SERVED_MODEL_NAME} | 精度={MODEL_PRECISION.upper()} | 无锁 Continuous Batching)"
    )
