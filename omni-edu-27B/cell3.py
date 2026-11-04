import time
from threading import Thread
import torch
from transformers import TextIteratorStreamer

# 设为 False 可在点击“全部运行”时直接跳过测试秒启动 FastAPI；需要交互测试时改为 True
RUN_SANITY_TEST = False


@torch.inference_mode()
def chat_omni_edu(
    prompt: str,
    system_prompt: str = "You are OmniEdu, a helpful AI assistant specialized in K-12 learning, curriculum grounding, diagnostic reasoning, and pedagogical tutoring.",
    enable_thinking: bool = True,
    reasoning_effort: str = "low",  # 测试建议用 "low"，可选: "low" | "medium" | "xhigh"
    max_new_tokens: int = 2048,
):
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
        assert reasoning_effort in ("low", "medium", "xhigh")
        template_kwargs["reasoning_effort"] = reasoning_effort

    try:
        text = tokenizer.apply_chat_template(messages, **template_kwargs)
    except Exception as e:
        print(f"ℹ️ apply_chat_template 带 reasoning_effort 调用降级 ({e})，正在以基础模板重试...")
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
    print(f"🧠 [思考模式: {mode_str}]\n" + "=" * 60)

    start_time = time.time()
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    full_output = []
    for new_text in streamer:
        print(new_text, end="", flush=True)
        full_output.append(new_text)

    thread.join()
    end_time = time.time()

    full_response_text = "".join(full_output)
    generated_tokens = tokenizer.encode(full_response_text, add_special_tokens=False)
    token_count = len(generated_tokens)
    elapsed_time = end_time - start_time
    token_speed = token_count / elapsed_time if elapsed_time > 0 else 0

    print("\n" + "=" * 60)
    print(f"📊 [性能统计] 生成 Token 数: {token_count} | 耗时: {elapsed_time:.2f}s | 速率: {token_speed:.2f} tokens/s")
    return full_response_text


# 保持向后兼容别名
chat_qwen38 = chat_omni_edu

if RUN_SANITY_TEST:
    _ = chat_omni_edu(
        prompt="请用直观生动的方式向初中生解释：为什么负负得正？",
        enable_thinking=True,
        reasoning_effort="low",
    )
