"""Cell 3: H100 vLLM 快速流式测试与首字延迟 (TTFT) / 吞吐测速工具。"""

import os
import json
import time
import urllib.request
import httpx

# 设为 False 可在点击“全部运行”时跳过自测直接启动网关；CLI 可通过 --env RUN_SANITY_TEST=1 开启
RUN_SANITY_TEST = os.environ.get("RUN_SANITY_TEST", "0").strip().lower() in ("1", "true", "yes")

DEFAULT_VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL") or globals().get("VLLM_BASE_URL", "http://127.0.0.1:8001/v1")
DEFAULT_GATEWAY_BASE_URL = os.environ.get("GATEWAY_BASE_URL") or globals().get("GATEWAY_BASE_URL", "http://127.0.0.1:8000/v1")
DEFAULT_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME") or globals().get("SERVED_MODEL_NAME", "Qwen/Qwen3.8-27B")
DEFAULT_API_KEY = os.environ.get("COLAB_API_KEY") or os.environ.get("OPENAI_API_KEY") or "sk-colab"


def resolve_chat_base_url(base_url: str | None = None) -> str:
    """优先使用显式传入 URL；若未传入，若 Cell 4 网关 (8000) 已启动则走网关，否则直连 Cell 2 vLLM (8001)。"""
    if base_url:
        return base_url.rstrip("/")

    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=0.5) as resp:
            if resp.status == 200:
                return DEFAULT_GATEWAY_BASE_URL.rstrip("/")
    except Exception:
        pass

    return DEFAULT_VLLM_BASE_URL.rstrip("/")


def build_sanity_payload(
    prompt: str,
    system_prompt: str,
    model_name: str,
    enable_thinking: bool,
    reasoning_effort: str,
    max_new_tokens: int,
) -> dict:
    """构造流式测试请求体，兼容 vLLM 原生端与 Cell 4 OpenAI 网关。"""
    if enable_thinking and reasoning_effort not in ("low", "medium", "xhigh"):
        raise ValueError(f"reasoning_effort 必须为 'low' | 'medium' | 'xhigh'，当前为: {reasoning_effort!r}")

    sampling = (
        {"temperature": 1.0, "top_p": 0.95}
        if enable_thinking
        else {"temperature": 0.7, "top_p": 0.80}
    )

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": int(max_new_tokens),
        "chat_template_kwargs": {
            "enable_thinking": bool(enable_thinking),
            "preserve_thinking": True,
        },
        **sampling,
    }
    if enable_thinking:
        payload["reasoning_effort"] = reasoning_effort
        payload["chat_template_kwargs"]["reasoning_effort"] = reasoning_effort

    return payload


def chat_qwen38(
    prompt: str,
    system_prompt: str = "你是一个专业、乐于助人的 AI 助手，请始终使用简体中文回答用户问题。",
    enable_thinking: bool = True,
    reasoning_effort: str = "low",  # 测试建议用 "low"，可选: "low" | "medium" | "xhigh"
    max_new_tokens: int = 2048,
    base_url: str | None = None,
    model_name: str | None = None,
) -> str:
    """通过本地 HTTP SSE 流式调用 H100 vLLM，分离打印思考与正文并统计 TTFT 与吞吐。"""
    endpoint_base = resolve_chat_base_url(base_url)
    target_model = model_name or globals().get("SERVED_MODEL_NAME", DEFAULT_MODEL_NAME)
    payload = build_sanity_payload(
        prompt=prompt,
        system_prompt=system_prompt,
        model_name=target_model,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
        max_new_tokens=max_new_tokens,
    )

    mode_str = f"开启 (档位: {reasoning_effort})" if enable_thinking else "关闭 (直答模式)"
    print(f"🧠 [思考模式: {mode_str} | 端点: {endpoint_base}]\n" + "=" * 64)

    t_start = time.perf_counter()
    first_token_time = None
    thinking_chunks = []
    content_chunks = []
    printed_thinking_header = False
    printed_content_header = False
    usage_completion_tokens = None
    stream_chunk_count = 0

    with httpx.Client(timeout=None) as client:
        with client.stream(
            "POST",
            f"{endpoint_base}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {DEFAULT_API_KEY}"},
        ) as resp:
            if resp.status_code >= 400:
                err_body = resp.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"❌ 请求失败 (HTTP {resp.status_code}): {err_body}")

            for raw_line in resp.iter_lines():
                if not raw_line or raw_line.startswith(":"):
                    continue
                if not raw_line.startswith("data:"):
                    continue
                data_str = raw_line[5:].strip()
                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if isinstance(chunk.get("usage"), dict):
                    comp = chunk["usage"].get("completion_tokens")
                    if comp is not None:
                        usage_completion_tokens = int(comp)

                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}

                r_piece = delta.get("reasoning_content")
                if r_piece is None:
                    r_piece = delta.get("reasoning")
                c_piece = delta.get("content")

                if r_piece:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    stream_chunk_count += 1
                    if not printed_thinking_header:
                        print("💭 [思考过程]:\n", flush=True)
                        printed_thinking_header = True
                    print(r_piece, end="", flush=True)
                    thinking_chunks.append(r_piece)

                if c_piece:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    stream_chunk_count += 1
                    if not printed_content_header:
                        prefix = "\n\n" if printed_thinking_header else ""
                        print(f"{prefix}📝 [正文输出]:\n", flush=True)
                        printed_content_header = True
                    print(c_piece, end="", flush=True)
                    content_chunks.append(c_piece)

    elapsed = time.perf_counter() - t_start
    ttft = (first_token_time - t_start) if first_token_time is not None else elapsed
    full_content = "".join(content_chunks)

    if usage_completion_tokens is not None and usage_completion_tokens > 0:
        token_count = usage_completion_tokens
        count_tag = "真实"
    else:
        token_count = max(stream_chunk_count, len(full_content) // 2)
        count_tag = "估算"

    tok_per_sec = token_count / elapsed if elapsed > 0 else 0.0
    print("\n" + "=" * 64)
    print(
        f"📊 [H100 性能统计] TTFT(首字延迟): {ttft * 1000:.0f} ms | "
        f"生成 Token({count_tag}): {token_count} | 总耗时: {elapsed:.2f}s | "
        f"吞吐速率: {tok_per_sec:.1f} tokens/s"
    )
    return full_content


if __name__ == "__main__" and RUN_SANITY_TEST:
    _ = chat_qwen38(
        prompt=os.environ.get("SANITY_PROMPT", "你好，请用一句话介绍你自己，并说明你运行在什么硬件推理栈上。"),
        enable_thinking=os.environ.get("ENABLE_THINKING", "1").strip().lower() not in ("0", "false", "no"),
        reasoning_effort=os.environ.get("REASONING_EFFORT", "low"),
        max_new_tokens=int(os.environ.get("MAX_NEW_TOKENS", "2048")),
    )
