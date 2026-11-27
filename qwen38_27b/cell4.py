"""Cell 4 (Qwen3.8-27B G4 Blackwell 96GB 并发版 FastAPI 网关 - Colab CLI 重构版).

核心修复与升级:
- 修复 P0 信号量反转泄漏/超发 Bug：改用跨线程/跨事件循环安全的 `threading.Semaphore` 原子非阻塞获取，
  彻底消除 `asyncio.Semaphore` 取消竞态及跨 event loop 隐患，同时供 Cell 6 进程内推理共享并发配额。
- 修复 P0 `G4FairStepScheduler` `rope_deltas` 跨会话污染：无论单/多会话均按线程独立保存恢复状态。
- 修复 P0 后台推理线程异常被吞与提前释放信号量：捕获异常传回客户端，确保推理线程真正退出后再释放资源。
- 修复 `apply_chat_template` 回退遗漏 `reasoning_effort`、正文 `<think>`/`<tool_call>` 误触发与 `raw_buffer` O(N²) 拼接。
- 加强安全性：图像 URL 下载增加 20MB 上限与私网/环回 SSRF 防护；支持可选 `COLAB_API_KEY` Bearer 鉴权。
"""

import io
import os
import re
import gc
import time
import uuid
import json
import queue
import base64
import socket
import logging
import asyncio
import ipaddress
import urllib.parse
import urllib.request
import functools
import threading
import contextlib
import collections
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
import torch

try:
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse
    from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
except ImportError:
    uvicorn = None
    FastAPI = None
    Request = None
    CORSMiddleware = None
    JSONResponse = None
    StreamingResponse = None
    StoppingCriteria = object
    StoppingCriteriaList = list
    TextIteratorStreamer = object

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.getLogger("asyncio").setLevel(logging.CRITICAL)
logging.getLogger("uvicorn.error").setLevel(logging.CRITICAL)

if "LAST_API_REQUEST_TIME" not in globals():
    globals()["LAST_API_REQUEST_TIME"] = time.time()

# ================= 配置常量 =================
DEFAULT_MODEL_ID = os.environ.get("MODEL_ID") or globals().get("model_id", "Qwen/Qwen3.8-27B")
DEFAULT_OWNED_BY = os.environ.get("MODEL_OWNED_BY", "qwen")
DEFAULT_SYSTEM_FALLBACK = "You are a helpful assistant."
GATEWAY_HOST = os.environ.get("GATEWAY_HOST", "0.0.0.0")
GATEWAY_PORT = int(os.environ.get("GATEWAY_PORT", "8000"))
REQUIRED_API_KEY = os.environ.get("COLAB_API_KEY", "").strip()
MAX_IMAGE_DOWNLOAD_BYTES = int(os.environ.get("MAX_IMAGE_DOWNLOAD_BYTES", str(20 * 1024 * 1024)))
ALLOW_PRIVATE_IMAGE_URLS = os.environ.get("ALLOW_PRIVATE_IMAGE_URLS", "0").strip().lower() in ("1", "true", "yes")

TAG_THINK_OPEN = "<" + "think>"
TAG_THINK_CLOSE = "</" + "think>"
TAG_TC_OPEN = "<tool_" + "call>"
TAG_TC_CLOSE = "</tool_" + "call>"
TAG_TOOLS_OPEN = "<" + "tools>"
TAG_TOOLS_CLOSE = "</" + "tools>"
TAG_FN_PREFIX = "<function="
TAG_FN_CLOSE = "</" + "function>"
TAG_PARAM_PREFIX = "<parameter="
TAG_PARAM_CLOSE = "</" + "parameter>"

REGEX_THINK_BLOCK = re.compile(re.escape(TAG_THINK_OPEN) + r"(.*?)" + re.escape(TAG_THINK_CLOSE), flags=re.DOTALL)
REGEX_THINK_LEADING = re.compile(r"^" + re.escape(TAG_THINK_OPEN) + r"\s*")
REGEX_TC_BLOCK = re.compile(re.escape(TAG_TC_OPEN) + r"(.*?)" + re.escape(TAG_TC_CLOSE), flags=re.DOTALL)
REGEX_FN_XML = re.compile(re.escape(TAG_FN_PREFIX) + r"([^\s>]+)>(.*?)" + re.escape(TAG_FN_CLOSE), flags=re.DOTALL)
REGEX_PARAM_XML = re.compile(re.escape(TAG_PARAM_PREFIX) + r"([^\s>]+)>(.*?)" + re.escape(TAG_PARAM_CLOSE), flags=re.DOTALL)
# ============================================


def setup_g4_blackwell_optimizations() -> int:
    if not torch.cuda.is_available():
        return 2

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(True)

    props = torch.cuda.get_device_properties(0)
    total_vram_gb = props.total_memory / (1024 ** 3)
    allocated_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
    free_vram_gb = total_vram_gb - allocated_gb

    if free_vram_gb >= 55:
        recommended_concurrency = 6
    elif free_vram_gb >= 32:
        recommended_concurrency = 4
    elif free_vram_gb >= 18:
        recommended_concurrency = 3
    else:
        recommended_concurrency = 2

    print(
        f"🚀 [G4 硬件加速已就绪] GPU: {props.name} (SM {props.major}.{props.minor}) | "
        f"总显存: {total_vram_gb:.1f} GB | 模型占用: {allocated_gb:.1f} GB | "
        f"剩余可用显存: {free_vram_gb:.1f} GB -> 最大并发会话数: {recommended_concurrency}",
        flush=True,
    )
    return recommended_concurrency


DEFAULT_REASONING_EFFORT = os.environ.get("DEFAULT_REASONING_EFFORT", "medium")
FORCE_XHIGH_WHEN_THINKING = os.environ.get("FORCE_XHIGH_WHEN_THINKING", "0").strip().lower() in ("1", "true", "yes")
MIN_THINKING_MAX_TOKENS = int(os.environ.get("MIN_THINKING_MAX_TOKENS", "16384"))
MIN_XHIGH_MAX_TOKENS = int(os.environ.get("MIN_XHIGH_MAX_TOKENS", "32768"))

MAX_CONCURRENT_SESSIONS = int(os.environ.get("MAX_CONCURRENT_SESSIONS") or setup_g4_blackwell_optimizations())
VRAM_GC_WATERMARK_RATIO = 0.85

cpu_prep_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="qwen_cpu_prep")


class TokenCountingStreamer(TextIteratorStreamer):
    """在流式生成过程中以 O(1) 复杂度实时统计生成的 Token 数量。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generated_tokens = 0
        self._first_put = True

    def put(self, value):
        if self.skip_prompt and self._first_put:
            self._first_put = False
        else:
            if isinstance(value, torch.Tensor):
                self.generated_tokens += int(value.numel())
            elif isinstance(value, (list, tuple)):
                self.generated_tokens += len(value)
        super().put(value)


class G4FairStepScheduler:
    """Token 级 FIFO 公平时间片轮转调度器（始终按线程隔离 rope_deltas，杜绝跨会话污染）。"""

    def __init__(self):
        self._cond = threading.Condition(threading.Lock())
        self._busy = False
        self._waiters = collections.deque()
        self._local = threading.local()
        self.active_sessions = 0
        self._session_lock = threading.Lock()

    def inc_session(self) -> int:
        with self._session_lock:
            self.active_sessions += 1
            return self.active_sessions

    def dec_session(self) -> int:
        with self._session_lock:
            self.active_sessions = max(0, self.active_sessions - 1)
            return self.active_sessions

    def reset_thread_state(self) -> None:
        """每个新会话开始前清空本线程残留的 saved_state。"""
        if hasattr(self._local, "saved_state"):
            delattr(self._local, "saved_state")

    @contextlib.contextmanager
    def step(self, target_model):
        my_event = None
        with self._cond:
            if not self._busy and not self._waiters:
                self._busy = True
            else:
                my_event = threading.Event()
                self._waiters.append(my_event)

        if my_event is not None:
            my_event.wait()

        inner_model = getattr(target_model, "model", None)
        saved = getattr(self._local, "saved_state", None)
        if saved is not None:
            if hasattr(target_model, "rope_deltas"):
                target_model.rope_deltas = saved.get("outer_rope_deltas", None)
            if inner_model is not None and hasattr(inner_model, "rope_deltas"):
                inner_model.rope_deltas = saved.get("inner_rope_deltas", None)
        else:
            if hasattr(target_model, "rope_deltas"):
                target_model.rope_deltas = None
            if inner_model is not None and hasattr(inner_model, "rope_deltas"):
                inner_model.rope_deltas = None

        try:
            yield
        finally:
            self._local.saved_state = {
                "outer_rope_deltas": getattr(target_model, "rope_deltas", None),
                "inner_rope_deltas": getattr(inner_model, "rope_deltas", None) if inner_model is not None else None,
            }
            with self._cond:
                if self._waiters:
                    next_event = self._waiters.popleft()
                    next_event.set()
                else:
                    self._busy = False


g4_scheduler = G4FairStepScheduler()
# 跨线程 / 跨事件循环安全的并发槽位信号量（供 FastAPI 与 Cell 6 共享）
gpu_slot_semaphore = threading.Semaphore(MAX_CONCURRENT_SESSIONS)


def install_model_concurrent_hook(target_model):
    if getattr(target_model, "_g4_concurrent_hooked", False):
        return
    orig_forward = target_model.forward

    @functools.wraps(orig_forward)
    def concurrent_safe_forward(*args, **kwargs):
        with g4_scheduler.step(target_model):
            return orig_forward(*args, **kwargs)

    target_model.forward = concurrent_safe_forward
    target_model._g4_concurrent_hooked = True
    print("⚡ [G4 调度器] 已挂载自适应 Token 级并发轮转钩子。", flush=True)


if "model" in globals() and globals().get("model") is not None:
    install_model_concurrent_hook(globals()["model"])


def smart_vram_cleanup_if_needed():
    if not torch.cuda.is_available():
        return
    total_mem = torch.cuda.get_device_properties(0).total_memory
    used_mem = torch.cuda.memory_reserved(0)
    if used_mem / max(total_mem, 1) >= VRAM_GC_WATERMARK_RATIO:
        gc.collect()
        torch.cuda.empty_cache()


class StopOnCancelCriteria(StoppingCriteria):
    def __init__(self, stop_event: threading.Event):
        self.stop_event = stop_event

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return self.stop_event.is_set()


def _validate_remote_image_url(url: str) -> None:
    """基础 SSRF 防护：仅允许 http/https，默认拒绝内网/环回/链路本地 IP。"""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(f"不支持的图片 URL scheme: {parsed.scheme!r}")
    if ALLOW_PRIVATE_IMAGE_URLS:
        return
    host = parsed.hostname
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise ValueError(f"无法解析图片域名 {host}: {exc}") from exc
    for _, _, _, _, sockaddr in infos:
        ip_obj = ipaddress.ip_address(sockaddr[0])
        if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_reserved or ip_obj.is_multicast:
            raise ValueError(f"出于安全策略拒绝访问内网图片地址: {host} ({ip_obj})")


def load_pil_image_from_url(url_or_b64: str, max_bytes: int = MAX_IMAGE_DOWNLOAD_BYTES) -> Image.Image:
    """安全解析 OpenAI image_url 中的 Base64 Data URI 或 HTTP 图片链接（带大小上限与 SSRF 防护）。"""
    if url_or_b64.startswith("data:image"):
        _, b64_data = url_or_b64.split(",", 1)
        if len(b64_data) > (max_bytes * 4) // 3 + 64:
            raise ValueError(f"Base64 图片体积超出上限 ({max_bytes} bytes)")
        img_bytes = base64.b64decode(b64_data)
        if len(img_bytes) > max_bytes:
            raise ValueError(f"图片体积超出上限 ({max_bytes} bytes)")
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    elif url_or_b64.startswith(("http://", "https://")):
        _validate_remote_image_url(url_or_b64)
        req = urllib.request.Request(url_or_b64, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise ValueError(f"远程图片体积超出上限 ({max_bytes} bytes)")
            return Image.open(io.BytesIO(data)).convert("RGB")
    else:
        if len(url_or_b64) > (max_bytes * 4) // 3 + 64:
            raise ValueError(f"Base64 图片体积超出上限 ({max_bytes} bytes)")
        img_bytes = base64.b64decode(url_or_b64)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def normalize_messages(raw_messages: list) -> tuple[list, list]:
    """标准化多轮消息，同时提取 PIL 图像列表以支持多模态看图。"""
    normalized = []
    pil_images = []

    for m in raw_messages:
        role = m.get("role", "user")
        if role == "developer":
            role = "system"

        content = m.get("content", "")
        extracted_reasoning = m.get("reasoning_content") or m.get("reasoning") or m.get("reasoning_text")

        if isinstance(content, list):
            text_parts = []
            mm_content_list = []
            reasoning_parts = []
            has_image_in_msg = False

            for part in content:
                if isinstance(part, dict):
                    p_type = part.get("type", "")
                    if p_type in ("text", "input_text", "output_text"):
                        txt = str(part.get("text", ""))
                        text_parts.append(txt)
                        mm_content_list.append({"type": "text", "text": txt})
                    elif p_type in ("image_url", "input_image", "image"):
                        img_info = part.get("image_url") or part.get("image") or part
                        url_val = img_info.get("url", "") if isinstance(img_info, dict) else str(img_info)
                        if url_val:
                            try:
                                pil_img = load_pil_image_from_url(url_val)
                                pil_images.append(pil_img)
                                mm_content_list.append({"type": "image", "image": pil_img})
                                has_image_in_msg = True
                            except Exception as exc:
                                print(f"⚠️ 图像解析失败，已跳过: {exc}", flush=True)
                    elif p_type in ("reasoning", "thinking"):
                        r_txt = part.get("text") or part.get("thinking") or part.get("reasoning") or ""
                        if r_txt:
                            reasoning_parts.append(str(r_txt))
                    elif "content" in part:
                        txt = str(part.get("content", ""))
                        text_parts.append(txt)
                        mm_content_list.append({"type": "text", "text": txt})
                elif isinstance(part, str):
                    text_parts.append(part)
                    mm_content_list.append({"type": "text", "text": part})

            content = mm_content_list if has_image_in_msg else "\n".join(text_parts)
            if reasoning_parts and not extracted_reasoning:
                extracted_reasoning = "\n".join(reasoning_parts)
        elif content is None:
            content = ""
        elif not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)

        if role == "assistant" and isinstance(content, str) and TAG_THINK_OPEN in content and TAG_THINK_CLOSE in content:
            m_think = REGEX_THINK_BLOCK.search(content)
            if m_think:
                if not extracted_reasoning:
                    extracted_reasoning = m_think.group(1).strip()
                content = (content[: m_think.start()] + content[m_think.end() :]).strip()

        msg = {"role": role, "content": content}

        if role == "assistant" and extracted_reasoning and str(extracted_reasoning).strip():
            msg["reasoning_content"] = str(extracted_reasoning).strip()

        if "tool_calls" in m and m["tool_calls"]:
            parsed_tcs = []
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args_dict = json.loads(args) if args.strip() else {}
                    except Exception:
                        args_dict = {"raw": args}
                elif isinstance(args, dict):
                    args_dict = args
                else:
                    args_dict = {}
                parsed_tcs.append(
                    {
                        "id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                        "type": "function",
                        "function": {"name": fn.get("name", ""), "arguments": args_dict},
                    }
                )
            msg["tool_calls"] = parsed_tcs

        if "tool_call_id" in m and m["tool_call_id"]:
            msg["tool_call_id"] = m["tool_call_id"]
        if "name" in m and m["name"]:
            msg["name"] = m["name"]

        normalized.append(msg)
    return normalized, pil_images


def extract_single_tool_call(block_str: str, idx: int):
    block_str = block_str.strip()
    fn_xml_match = REGEX_FN_XML.search(block_str)
    if fn_xml_match:
        fn_name = fn_xml_match.group(1).strip()
        params_body = fn_xml_match.group(2)
        args_dict = {}
        for p_match in REGEX_PARAM_XML.finditer(params_body):
            p_key = p_match.group(1).strip()
            p_val = p_match.group(2).strip().strip("\n")
            try:
                args_dict[p_key] = json.loads(p_val)
            except Exception:
                args_dict[p_key] = p_val
        return {
            "index": idx,
            "id": f"call_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {"name": fn_name, "arguments": json.dumps(args_dict, ensure_ascii=False)},
        }

    start_brace = block_str.find("{")
    end_brace = block_str.rfind("}")
    if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
        json_candidate = block_str[start_brace : end_brace + 1]
        try:
            tc_obj = json.loads(json_candidate)
            fn_name = tc_obj.get("name") or tc_obj.get("function", "")
            fn_args = tc_obj.get("arguments", tc_obj.get("parameters", {}))
            if not isinstance(fn_args, str):
                fn_args = json.dumps(fn_args, ensure_ascii=False)
            return {
                "index": idx,
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {"name": fn_name, "arguments": fn_args},
            }
        except Exception as exc:
            print(f"⚠️ JSON 工具解析异常: {exc}", flush=True)
    return None


def parse_qwen_output(raw_text: str, prompt_opened_think: bool = True, parse_tools: bool = True):
    for eos_tok in ("<|im_end|>", "<|endoftext|>"):
        raw_text = raw_text.replace(eos_tok, "")

    reasoning_content = None
    body_text = raw_text

    think_match = REGEX_THINK_BLOCK.search(raw_text)
    if think_match:
        reasoning_content = think_match.group(1).strip()
        body_text = raw_text[: think_match.start()] + raw_text[think_match.end() :]
    elif TAG_THINK_CLOSE in raw_text:
        think_part, rest_part = raw_text.split(TAG_THINK_CLOSE, 1)
        think_part = REGEX_THINK_LEADING.sub("", think_part).strip()
        if think_part:
            reasoning_content = think_part
        body_text = rest_part
    elif prompt_opened_think:
        reasoning_content = REGEX_THINK_LEADING.sub("", raw_text).strip()
        body_text = ""

    tool_calls = []
    if parse_tools:
        for idx, match in enumerate(REGEX_TC_BLOCK.finditer(body_text)):
            parsed_tc = extract_single_tool_call(match.group(1), idx)
            if parsed_tc:
                tool_calls.append(parsed_tc)

        if not tool_calls and TAG_TC_OPEN in body_text:
            unclosed_part = body_text.split(TAG_TC_OPEN, 1)[1]
            parsed_tc = extract_single_tool_call(unclosed_part, 0)
            if parsed_tc:
                tool_calls.append(parsed_tc)
                body_text = body_text.split(TAG_TC_OPEN, 1)[0]

        clean_content = REGEX_TC_BLOCK.sub("", body_text).strip()
    else:
        clean_content = body_text.strip()

    return reasoning_content, clean_content, tool_calls


def build_sampling_kwargs(enable_thinking: bool, req_temp, req_top_p):
    if enable_thinking:
        temp = float(req_temp) if req_temp is not None else 1.0
        top_p = float(req_top_p) if req_top_p is not None else 0.95
        if temp <= 0:
            return dict(do_sample=False)
        return dict(do_sample=True, temperature=temp, top_p=top_p, top_k=20)
    else:
        temp = float(req_temp) if req_temp is not None else 0.7
        top_p = float(req_top_p) if req_top_p is not None else 0.80
        if temp <= 0:
            return dict(do_sample=False)
        return dict(do_sample=True, temperature=temp, top_p=top_p, top_k=20, repetition_penalty=1.05)


def prepare_prompt_and_inputs_sync(raw_messages, enable_thinking, preserve_thinking, reasoning_effort, tools):
    """在 CPU 线程池中完成消息标准化、Chat 模板渲染、多模态/文本编码与锁页内存 (pin_memory) 绑定。"""
    messages, pil_images = normalize_messages(raw_messages)
    active_proc = processor if (pil_images and "processor" in globals() and processor is not None) else tokenizer

    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
        "preserve_thinking": preserve_thinking,
    }
    if enable_thinking:
        template_kwargs["reasoning_effort"] = reasoning_effort
    if tools:
        template_kwargs["tools"] = tools

    try:
        prompt_text = active_proc.apply_chat_template(messages, **template_kwargs)
    except Exception as exc:
        print(f"⚠️ apply_chat_template 标准调用回退 ({exc})，使用兼容模式构建 Prompt...", flush=True)
        template_kwargs.pop("tools", None)
        template_kwargs.pop("preserve_thinking", None)
        template_kwargs.pop("reasoning_effort", None)
        if tools:
            tools_sys_prompt = (
                f"\n\n# Tools\nYou may call one or more functions to assist with the user query.\n"
                f"You are provided with function signatures within {TAG_TOOLS_OPEN}{TAG_TOOLS_CLOSE} XML tags:\n{TAG_TOOLS_OPEN}\n"
                + "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)
                + f"\n{TAG_TOOLS_CLOSE}\n\nFor each function call, return a json object with function name and arguments within {TAG_TC_OPEN}{TAG_TC_CLOSE} XML tags:\n"
                f"{TAG_TC_OPEN}\n{{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n{TAG_TC_CLOSE}"
            )
            if messages and messages[0]["role"] == "system" and isinstance(messages[0]["content"], str):
                messages[0]["content"] += tools_sys_prompt
            else:
                messages.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_FALLBACK + tools_sys_prompt})
        prompt_text = tokenizer.apply_chat_template(messages, **template_kwargs)

    prompt_opened_think = enable_thinking and prompt_text.rstrip().endswith(TAG_THINK_OPEN)

    if pil_images and "processor" in globals() and processor is not None:
        raw_inputs = processor(text=[prompt_text], images=pil_images, return_tensors="pt")
    else:
        raw_inputs = tokenizer([prompt_text], return_tensors="pt", add_special_tokens=False)

    pinned_inputs = {
        k: (v.pin_memory() if isinstance(v, torch.Tensor) and torch.cuda.is_available() else v)
        for k, v in raw_inputs.items()
    }
    return prompt_text, prompt_opened_think, pinned_inputs, len(pil_images)


def _create_app():
    if FastAPI is None:
        return None

    fastapi_app = FastAPI(title=f"{DEFAULT_MODEL_ID} G4-Blackwell Concurrent API", version="10.0")
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _check_api_key(request: Request):
        if not REQUIRED_API_KEY:
            return None
        if request.headers.get("authorization", "") == f"Bearer {REQUIRED_API_KEY}":
            return None
        return JSONResponse({"error": {"message": "Unauthorized: invalid API key"}}, status_code=401)

    @fastapi_app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        auth_err = _check_api_key(request)
        if auth_err is not None:
            return auth_err

        globals()["LAST_API_REQUEST_TIME"] = time.time()
        install_model_concurrent_hook(model)

        body = await request.json()
        raw_messages = body.get("messages", [])
        model_name = body.get("model", DEFAULT_MODEL_ID)
        stream = bool(body.get("stream", False))
        raw_req_max_tokens = body.get("max_tokens")
        if raw_req_max_tokens is None:
            raw_req_max_tokens = body.get("max_completion_tokens")
        req_max_tokens = int(raw_req_max_tokens) if raw_req_max_tokens is not None else 16384
        req_temp = body.get("temperature", None)
        req_top_p = body.get("top_p", None)
        tools = body.get("tools", None)
        has_tools = bool(tools)

        chat_template_kwargs = body.get("chat_template_kwargs") or {}
        preserve_thinking = body.get("preserve_thinking", chat_template_kwargs.get("preserve_thinking", True))

        raw_effort = (
            body.get("reasoning_effort")
            or (body.get("reasoning") or {}).get("effort")
            or body.get("thinking_level")
            or chat_template_kwargs.get("reasoning_effort")
        )
        pi_to_qwen_effort = {
            "off": None, "none": None, "false": None, "disabled": None,
            "minimal": "low", "low": "low",
            "medium": "medium",
            "high": "xhigh", "xhigh": "xhigh", "max": "xhigh",
        }

        if raw_effort is not None:
            mapped_effort = pi_to_qwen_effort.get(str(raw_effort).lower(), DEFAULT_REASONING_EFFORT)
            enable_thinking = mapped_effort is not None
            reasoning_effort = mapped_effort or DEFAULT_REASONING_EFFORT
        else:
            enable_thinking = bool(body.get("enable_thinking", chat_template_kwargs.get("enable_thinking", True)))
            reasoning_effort = DEFAULT_REASONING_EFFORT

        if enable_thinking and FORCE_XHIGH_WHEN_THINKING:
            reasoning_effort = "xhigh"

        if enable_thinking and (raw_req_max_tokens is None or req_max_tokens > 256):
            min_tok = MIN_XHIGH_MAX_TOKENS if reasoning_effort == "xhigh" else MIN_THINKING_MAX_TOKENS
            max_tokens = max(req_max_tokens, min_tok)
        else:
            max_tokens = req_max_tokens

        req_short_id = uuid.uuid4().hex[:6]
        tool_names = [t.get("function", {}).get("name", "unknown") for t in (tools or [])]

        loop = asyncio.get_running_loop()
        prompt_text, prompt_opened_think, cpu_inputs, img_count = await loop.run_in_executor(
            cpu_prep_executor,
            prepare_prompt_and_inputs_sync,
            raw_messages,
            enable_thinking,
            preserve_thinking,
            reasoning_effort,
            tools,
        )
        prompt_tokens = int(cpu_inputs["input_ids"].shape[1])
        sampling_kwargs = build_sampling_kwargs(enable_thinking, req_temp, req_top_p)

        print(
            f"📥 [{req_short_id} 收到请求] stream={stream} | Prompt={prompt_tokens} tok (图片:{img_count}) | "
            f"thinking={enable_thinking}({reasoning_effort if enable_thinking else 'off'}) | "
            f"并发={g4_scheduler.active_sessions}/{MAX_CONCURRENT_SESSIONS} | Tools:{len(tool_names)}",
            flush=True,
        )

        # ==================== 分支 A：非流式请求 ====================
        if not stream:
            acquired = False
            while not acquired:
                if await request.is_disconnected():
                    print(f"⏹️ [{req_short_id} 排队期间客户端断开] 已取消非流式请求。", flush=True)
                    return JSONResponse({"error": {"message": "Client disconnected"}}, status_code=499)
                acquired = gpu_slot_semaphore.acquire(blocking=False)
                if not acquired:
                    await asyncio.sleep(0.05)

            active_now = g4_scheduler.inc_session()
            stop_event = threading.Event()
            model_inputs = {
                k: (
                    v.to(model.device, dtype=model.dtype, non_blocking=True)
                    if isinstance(v, torch.Tensor) and v.is_floating_point()
                    else (v.to(model.device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                )
                for k, v in cpu_inputs.items()
            }
            del cpu_inputs
            streamer = TokenCountingStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
            generation_kwargs = dict(
                **model_inputs,
                streamer=streamer,
                max_new_tokens=max_tokens,
                use_cache=True,
                stopping_criteria=StoppingCriteriaList([StopOnCancelCriteria(stop_event)]),
                **sampling_kwargs,
            )

            t0 = time.time()
            gen_error: list[BaseException | None] = [None]

            def run_gen_sync():
                g4_scheduler.reset_thread_state()
                try:
                    with torch.inference_mode():
                        model.generate(**generation_kwargs)
                except BaseException as exc:
                    gen_error[0] = exc
                    streamer.end()

            thread = threading.Thread(target=run_gen_sync, daemon=True)
            thread.start()

            chunks = []
            try:
                while thread.is_alive() or not streamer.text_queue.empty():
                    if await request.is_disconnected():
                        stop_event.set()
                        print(f"⏹️ [{req_short_id} 客户端已断开] 已立即中止该非流式推理。", flush=True)
                        break
                    try:
                        piece = streamer.text_queue.get_nowait()
                        if piece == streamer.stop_signal:
                            break
                        if piece:
                            chunks.append(piece)
                    except queue.Empty:
                        await asyncio.sleep(0.01)
            finally:
                stop_event.set()
                await asyncio.to_thread(thread.join)
                globals()["LAST_API_REQUEST_TIME"] = time.time()
                del model_inputs, generation_kwargs
                remaining = g4_scheduler.dec_session()
                gpu_slot_semaphore.release()
                if remaining == 0:
                    smart_vram_cleanup_if_needed()

            if gen_error[0] is not None:
                return JSONResponse(
                    {"error": {"message": f"Inference failed: {type(gen_error[0]).__name__}: {gen_error[0]}"}},
                    status_code=500,
                )

            elapsed = time.time() - t0
            raw_output = "".join(chunks)
            reasoning_content, clean_content, tool_calls = parse_qwen_output(
                raw_output,
                prompt_opened_think=prompt_opened_think,
                parse_tools=has_tools,
            )
            completion_tokens = streamer.generated_tokens
            finish_reason = "tool_calls" if tool_calls else ("length" if completion_tokens >= max_tokens else "stop")
            print(
                f"📊 [{req_short_id} 非流式完成] 并发:{active_now} | Prompt:{prompt_tokens} tok | "
                f"Gen:{completion_tokens} tok | {completion_tokens / max(elapsed, 1e-5):.1f} tok/s | "
                f"工具:{[tc['function']['name'] for tc in tool_calls]}",
                flush=True,
            )

            msg_obj = {
                "role": "assistant",
                "content": clean_content if clean_content else (None if tool_calls else ""),
            }
            if reasoning_content is not None:
                msg_obj["reasoning_content"] = reasoning_content
            if tool_calls:
                msg_obj["tool_calls"] = [{k: v for k, v in tc.items() if k != "index"} for tc in tool_calls]

            return JSONResponse(
                {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{"index": 0, "message": msg_obj, "finish_reason": finish_reason}],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }
            )

        # ==================== 分支 B：流式请求 ====================
        async def event_generator():
            nonlocal cpu_inputs
            request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created_time = int(time.time())

            def make_delta_chunk(delta_dict, finish_reason=None, usage_dict=None):
                payload = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": delta_dict, "finish_reason": finish_reason}],
                }
                if usage_dict is not None:
                    payload["usage"] = usage_dict
                return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            yield make_delta_chunk({"role": "assistant", "content": ""})

            # 原子非阻塞获取并发槽位，彻底避免 asyncio.Semaphore 取消竞态导致的泄漏或超发
            acquired = False
            last_queue_hb = time.time()
            while not acquired:
                if await request.is_disconnected():
                    print(f"⏹️ [{req_short_id} 排队期间客户端断开] 已退出等待队列。", flush=True)
                    return
                acquired = gpu_slot_semaphore.acquire(blocking=False)
                if not acquired:
                    if time.time() - last_queue_hb >= 2.0:
                        yield ": keep-alive-queued\n\n"
                        last_queue_hb = time.time()
                    await asyncio.sleep(0.05)

            active_now = g4_scheduler.inc_session()
            stop_event = threading.Event()
            model_inputs = {
                k: (
                    v.to(model.device, dtype=model.dtype, non_blocking=True)
                    if isinstance(v, torch.Tensor) and v.is_floating_point()
                    else (v.to(model.device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                )
                for k, v in cpu_inputs.items()
            }
            del cpu_inputs
            streamer = TokenCountingStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)

            generation_kwargs = dict(
                **model_inputs,
                streamer=streamer,
                max_new_tokens=max_tokens,
                use_cache=True,
                stopping_criteria=StoppingCriteriaList([StopOnCancelCriteria(stop_event)]),
                **sampling_kwargs,
            )

            t0 = time.time()
            gen_error: list[BaseException | None] = [None]

            def run_gen_stream():
                g4_scheduler.reset_thread_state()
                try:
                    with torch.inference_mode():
                        model.generate(**generation_kwargs)
                except BaseException as exc:
                    gen_error[0] = exc
                    streamer.end()

            thread = threading.Thread(target=run_gen_stream, daemon=True)
            thread.start()

            try:
                raw_chunks: list[str] = []
                saw_think_close = False
                pending_text = ""
                in_thinking = bool(prompt_opened_think)
                in_tool_call = False
                stripped_leading_after_think = False
                last_heartbeat = time.time()

                while True:
                    if await request.is_disconnected():
                        stop_event.set()
                        print(f"⏹️ [{req_short_id} 客户端主动断开连接] 已立即中止该会话 GPU 推理。", flush=True)
                        return

                    try:
                        text_chunk = streamer.text_queue.get_nowait()
                        if text_chunk == streamer.stop_signal:
                            break
                    except queue.Empty:
                        if not thread.is_alive():
                            break
                        if time.time() - last_heartbeat > 2.5:
                            yield ": keep-alive\n\n"
                            last_heartbeat = time.time()
                        await asyncio.sleep(0.008)
                        continue

                    if not text_chunk:
                        continue
                    raw_chunks.append(text_chunk)
                    if TAG_THINK_CLOSE in text_chunk:
                        saw_think_close = True

                    clean_piece = text_chunk.replace("<|im_end|>", "").replace("<|endoftext|>", "")
                    if not clean_piece:
                        continue

                    if in_tool_call:
                        if time.time() - last_heartbeat > 2.5:
                            yield ": keep-alive\n\n"
                            last_heartbeat = time.time()
                        continue

                    pending_text += clean_piece

                    # ---------------- 状态 1：处于思考阶段 ----------------
                    if in_thinking:
                        if pending_text.startswith(TAG_THINK_OPEN):
                            pending_text = pending_text[len(TAG_THINK_OPEN) :]
                            if pending_text.startswith("\n"):
                                pending_text = pending_text[1:]

                        if TAG_THINK_CLOSE in pending_text:
                            think_part, after_part = pending_text.split(TAG_THINK_CLOSE, 1)
                            if think_part:
                                yield make_delta_chunk({"reasoning_content": think_part})
                                last_heartbeat = time.time()
                            in_thinking = False
                            saw_think_close = True
                            pending_text = after_part.lstrip("\n")
                            stripped_leading_after_think = bool(after_part)
                        else:
                            last_lt = pending_text.rfind("<")
                            if last_lt != -1 and (
                                TAG_THINK_CLOSE.startswith(pending_text[last_lt:])
                                or TAG_THINK_OPEN.startswith(pending_text[last_lt:])
                            ):
                                safe_think = pending_text[:last_lt]
                                pending_text = pending_text[last_lt:]
                            else:
                                safe_think = pending_text
                                pending_text = ""

                            if safe_think:
                                yield make_delta_chunk({"reasoning_content": safe_think})
                                last_heartbeat = time.time()
                            continue

                    # ---------------- 状态 2：已退出思考阶段（正文或工具调用） ----------------
                    if not in_thinking and pending_text:
                        if not stripped_leading_after_think:
                            stripped = pending_text.lstrip("\n")
                            if stripped != pending_text and not stripped:
                                pending_text = ""
                                continue
                            pending_text = stripped
                            stripped_leading_after_think = True

                        if enable_thinking and not saw_think_close and TAG_THINK_OPEN in pending_text:
                            before_t, after_t = pending_text.split(TAG_THINK_OPEN, 1)
                            if before_t:
                                yield make_delta_chunk({"content": before_t})
                            in_thinking = True
                            pending_text = after_t.lstrip("\n")
                            continue

                        if has_tools and TAG_TC_OPEN in pending_text:
                            in_tool_call = True
                            before_tc = pending_text.split(TAG_TC_OPEN, 1)[0]
                            if before_tc:
                                yield make_delta_chunk({"content": before_tc})
                                last_heartbeat = time.time()
                            pending_text = ""
                            continue

                        last_lt = pending_text.rfind("<")
                        tail = pending_text[last_lt:] if last_lt != -1 else ""
                        hold_for_tc = has_tools and last_lt != -1 and TAG_TC_OPEN.startswith(tail)
                        hold_for_think = enable_thinking and not saw_think_close and last_lt != -1 and TAG_THINK_OPEN.startswith(tail)
                        if hold_for_tc or hold_for_think:
                            safe_to_emit = pending_text[:last_lt]
                            pending_text = pending_text[last_lt:]
                        else:
                            safe_to_emit = pending_text
                            pending_text = ""

                        if safe_to_emit:
                            yield make_delta_chunk({"content": safe_to_emit})
                            last_heartbeat = time.time()

                if pending_text and not in_tool_call:
                    if in_thinking:
                        yield make_delta_chunk({"reasoning_content": pending_text})
                    else:
                        yield make_delta_chunk({"content": pending_text})

                await asyncio.to_thread(thread.join)

                if gen_error[0] is not None:
                    err_payload = {"error": {"message": f"Inference failed: {type(gen_error[0]).__name__}: {gen_error[0]}"}}
                    yield f"data: {json.dumps(err_payload, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                elapsed = time.time() - t0
                raw_buffer = "".join(raw_chunks)
                _, _, tool_calls = parse_qwen_output(
                    raw_buffer,
                    prompt_opened_think=prompt_opened_think,
                    parse_tools=has_tools,
                )
                completion_tokens = streamer.generated_tokens
                called_names = [tc["function"]["name"] for tc in tool_calls]
                print(
                    f"📊 [{req_short_id} 流式完成] 并发:{active_now} | Prompt:{prompt_tokens} tok | "
                    f"Gen:{completion_tokens} tok | {completion_tokens / max(elapsed, 1e-5):.1f} tok/s | "
                    f"触发工具:{called_names}",
                    flush=True,
                )

                if tool_calls:
                    yield make_delta_chunk({"tool_calls": tool_calls})

                finish_reason = "tool_calls" if tool_calls else ("length" if completion_tokens >= max_tokens else "stop")
                usage_obj = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
                yield make_delta_chunk({}, finish_reason=finish_reason, usage_dict=usage_obj)
                yield "data: [DONE]\n\n"

            except (asyncio.CancelledError, BrokenPipeError, ConnectionResetError):
                stop_event.set()
                print(f"⏹️ [{req_short_id} 流式传输中断] 已安全回收连接并停止该会话 GPU 生成。", flush=True)
            finally:
                stop_event.set()
                await asyncio.to_thread(thread.join)
                globals()["LAST_API_REQUEST_TIME"] = time.time()
                del model_inputs, generation_kwargs
                remaining = g4_scheduler.dec_session()
                gpu_slot_semaphore.release()
                if remaining == 0:
                    smart_vram_cleanup_if_needed()

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
        )

    @fastapi_app.get("/v1/models")
    @fastapi_app.get("/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": DEFAULT_MODEL_ID,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": DEFAULT_OWNED_BY,
                }
            ],
        }

    @fastapi_app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": DEFAULT_MODEL_ID,
            "active_sessions": g4_scheduler.active_sessions,
            "max_concurrent_sessions": MAX_CONCURRENT_SESSIONS,
        }

    return fastapi_app


app = _create_app()


def start_gateway_server():
    if FastAPI is None or uvicorn is None or app is None:
        raise ImportError("❌ 缺少 fastapi / uvicorn 依赖，请先运行 cell1.py！")

    prev_server = globals().get("uvicorn_server")
    if prev_server is not None:
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

    def start_isolated_uvicorn():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve())

    threading.Thread(target=start_isolated_uvicorn, daemon=True, name="g4_gateway_uvicorn").start()

    deadline = time.time() + 10.0
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{GATEWAY_PORT}/health", timeout=1.0) as resp:
                if resp.status == 200:
                    break
        except Exception:
            time.sleep(0.2)
    return server


if __name__ == "__main__":
    uvicorn_server = start_gateway_server()
    print(
        f"✅ {DEFAULT_MODEL_ID} (G4 Blackwell) 并发版 FastAPI 已启动 "
        f"(端口 {GATEWAY_PORT} | 最大并发会话={MAX_CONCURRENT_SESSIONS} | 支持多模态看图)",
        flush=True,
    )
