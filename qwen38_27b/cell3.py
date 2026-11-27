"""Qwen3.8-27B Cell 3: 快速进程内流式连通性测试（已修复缺失 import torch 与模板降级）。"""

import os
import time
from threading import Thread

import torch
from transformers import TextIteratorStreamer

# 设为 False 可在全流程执行时跳过自测；CLI 可通过 --env RUN_SANITY_TEST=1 开启
RUN_SANITY_TEST = os.environ.get("RUN_SANITY_TEST", "0").strip().lower() in ("1", "true", "yes")


@torch.inference_mode()
def chat_qwen38(
    prompt: str,
    system_prompt: str = "你是一个专业、乐于助人的 AI 助手，请始终使用简体中文回答用户问题。",
    enable_thinking: bool = True,
    reasoning_effort: str = "low",  # 测试建议用 "low"，可选: "low" | "medium" | "xhigh"
    max_new_tokens: int = 2048,
):
    if "model" not in globals() or "tokenizer" not in globals():
        raise RuntimeError("❌ 未检测到已加载的 model/tokenizer，请先运行 cell2.py！")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    if enable_thinking:
        if reasoning_effort not in ("low", "medium", "xhigh"):
            raise ValueError(f"reasoning_effort 必须为 low/medium/xhigh，当前: {reasoning_effort!r}")
        template_kwargs["reasoning_effort"] = reasoning_effort

    try:
        text = tokenizer.apply_chat_template(messages, **template_kwargs)
    except Exception as exc:
        print(f"ℹ️ apply_chat_template 带 reasoning_effort 调用降级 ({exc})，以基础模板重试...", flush=True)
        template_kwargs.pop("reasoning_effort", None)
        text = tokenizer.apply_chat_template(messages, **template_kwargs)

    model_inputs = tokenizer([text], return_tensors="pt", add_special_tokens=False).to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    if enable_thinking:
        sampling_kwargs = dict(do_sample=True, temperature=1.0, top_p=0.95, top_k=20)
    else:
        sampling_kwargs = dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05)

    generation_kwargs = dict(
        **model_inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        **sampling_kwargs,
    )

    mode_str = f"开启 (档位: {reasoning_effort})" if enable_thinking else "关闭 (直答模式)"
    print(f"🧠 [思考模式: {mode_str}]\n" + "=" * 60, flush=True)

    gen_error: list[BaseException | None] = [None]

    def _run_gen() -> None:
        try:
            model.generate(**generation_kwargs)
        except BaseException as exc:
            gen_error[0] = exc
            streamer.end()

    start_time = time.time()
    thread = Thread(target=_run_gen, daemon=True)
    thread.start()

    full_output = []
    for new_text in streamer:
        print(new_text, end="", flush=True)
        full_output.append(new_text)

    thread.join()
    end_time = time.time()
    if gen_error[0] is not None:
        raise gen_error[0]

    full_response_text = "".join(full_output)
    generated_tokens = tokenizer.encode(full_response_text, add_special_tokens=False)
    token_count = len(generated_tokens)
    elapsed_time = end_time - start_time
    token_speed = token_count / elapsed_time if elapsed_time > 0 else 0.0

    print("\n" + "=" * 60)
    print(
        f"📊 [性能统计] 生成 Token 数: {token_count} | 耗时: {elapsed_time:.2f}s | 速率: {token_speed:.2f} tokens/s",
        flush=True,
    )
    return full_response_text


if __name__ == "__main__" and RUN_SANITY_TEST:
    _ = chat_qwen38(
        prompt=os.environ.get("SANITY_PROMPT", "你好，请用一句话介绍你自己。"),
        enable_thinking=os.environ.get("ENABLE_THINKING", "1").strip().lower() not in ("0", "false", "no"),
        reasoning_effort=os.environ.get("REASONING_EFFORT", "low"),
        max_new_tokens=int(os.environ.get("MAX_NEW_TOKENS", "2048")),
    )
