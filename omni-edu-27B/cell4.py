# Cell 4 (OpenCode / Pi Agent 完美兼容版 - Omni-Edu-27B G4 Blackwell 96GB 终极优化版):
# - 修复混合架构缓存：移除硬编码 DynamicCache()，兼容 Qwen3.8 / Omni-Edu-27B Gated DeltaNet + Attention 专属混合缓存
# - 实时 Token 计数器：新增 TokenCountingStreamer，O(1) 统计生成 Token，省去流式结束后的二次分词
# - 锁页内存异步传输：CPU 输入开启 pin_memory()，使 non_blocking=True 实现真正的 PCIe 异步 DMA
# - 多模态视觉支持：自动识别 OpenAI image_url (Base64 / URL)，支持 OpenCode / Pi 发送截图看图编程与图表解析
# - 防空烧时间戳：启动时即初始化 LAST_API_REQUEST_TIME，并在生成期间持续刷新

import io
import os
import re
import gc
import time
import uuid
import json
import queue
import base64
import logging
import asyncio
import urllib.request
import functools
import threading
import contextlib
import collections
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import uvicorn
import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.getLogger("asyncio").setLevel(logging.CRITICAL)
logging.getLogger("uvicorn.error").setLevel(logging.CRITICAL)

# 初始化全局最后请求时间（修复从未发起请求时闲置计时永远为 0 的 Bug）
if "LAST_API_REQUEST_TIME" not in globals():
    globals()["LAST_API_REQUEST_TIME"] = time.time()

# ================= 标签常量定义 =================
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

# 预编译高频正则表达式（避免每次请求重复编译）
REGEX_THINK_BLOCK = re.compile(re.escape(TAG_THINK_OPEN) + r"(.*?)" + re.escape(TAG_THINK_CLOSE), flags=re.DOTALL)
REGEX_THINK_LEADING = re.compile(r"^" + re.escape(TAG_THINK_OPEN) + r"\s*")
REGEX_TC_BLOCK = re.compile(re.escape(TAG_TC_OPEN) + r"(.*?)" + re.escape(TAG_TC_CLOSE), flags=re.DOTALL)
REGEX_FN_XML = re.compile(re.escape(TAG_FN_PREFIX) + r"([^\s>]+)>(.*?)" + re.escape(TAG_FN_CLOSE), flags=re.DOTALL)
REGEX_PARAM_XML = re.compile(re.escape(TAG_PARAM_PREFIX) + r"([^\s>]+)>(.*?)" + re.escape(TAG_PARAM_CLOSE), flags=re.DOTALL)
# ================================================


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
        f"剩余可用显存: {free_vram_gb:.1f} GB -> 最大并发会话数: {recommended_concurrency}"
    )
    return recommended_concurrency


# ================= 核心推理强度与 G4 并发配置 =================
DEFAULT_REASONING_EFFORT = "medium"
FORCE_XHIGH_WHEN_THINKING = False
MIN_THINKING_MAX_TOKENS = 16384
MIN_XHIGH_MAX_TOKENS = 32768

MAX_CONCURRENT_SESSIONS = setup_g4_blackwell_optimizations()
VRAM_GC_WATERMARK_RATIO = 0.85
# ==============================================================

app = FastAPI(title="Omni-Edu-27B G4-Blackwell Concurrent API", version="9.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cpu_prep_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="omniedu_cpu_prep")


class TokenCountingStreamer(TextIteratorStreamer):
    """在流式生成过程中以 O(1) 复杂度实时统计生成的 Token 数量，消除结束后的二次分词开销"""
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
    """单会话零锁直通 + 多会话 Token 级 FIFO 公平时间片轮转调度器"""
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

        multi_session = self.active_sessions > 1
        inner_model = getattr(target_model, "model", None)

        if multi_session and hasattr(self._local, "saved_state"):
            if hasattr(target_model, "rope_deltas"):
                target_model.rope_deltas = self._local.saved_state.get("outer_rope_deltas", None)
            if inner_model is not None and hasattr(inner_model, "rope_deltas"):
                inner_model.rope_deltas = self._local.saved_state.get("inner_rope_deltas", None)

        try:
            yield
        finally:
            if multi_session:
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
gpu_session_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SESSIONS)


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
    print("⚡ [G4 调度器] 已挂载自适应 Token 级并发轮转钩子。")


if "model" in globals() and model is not None:
    install_model_concurrent_hook(model)


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


def load_pil_image_from_url(url_or_b64: str) -> Image.Image:
    """解析 OpenAI image_url 中的 Base64 Data URI 或 HTTP 图片链接"""
    if url_or_b64.startswith("data:image"):
        _, b64_data = url_or_b64.split(",", 1)
        img_bytes = base64.b64decode(b64_data)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    elif url_or_b64.startswith(("http://", "https://")):
        req = urllib.request.Request(url_or_b64, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return Image.open(io.BytesIO(resp.read())).convert("RGB")
    else:
        img_bytes = base64.b64decode(url_or_b64)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def normalize_messages(raw_messages: list) -> tuple[list, list]:
    """标准化多轮消息，同时提取 PIL 图像列表以支持多模态看图"""
    normalized = []
    pil_images = []

    for m in raw_messages:
        role = m.get("role", "user")
        if role == "developer":
            role = "system"

        content = m.get("content", "")
        extracted_reasoning = (
            m.get("reasoning_content")
            or m.get("reasoning")
            or m.get("reasoning_text")
        )

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
                            except Exception as e:
                                print(f"⚠️ 图像解析失败，已跳过: {e}")
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
                content = (content[:m_think.start()] + content[m_think.end():]).strip()

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
                parsed_tcs.append({
                    "id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {"name": fn.get("name", ""), "arguments": args_dict},
                })
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
            p_val = p_match.group(2).strip()
            if p_val.startswith("\n"):
                p_val = p_val[1:]
            if p_val.endswith("\n"):
                p_val = p_val[:-1]
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
        except Exception as e:
            print(f"⚠️ JSON 工具解析异常: {e}")
    return None


def parse_qwen_output(raw_text: str, prompt_opened_think: bool = True):
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
    """在 CPU 线程池中完成消息标准化、Chat 模板渲染、多模态/文本编码与锁页内存 (pin_memory) 绑定"""
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
    except Exception as e:
        print(f"⚠️ apply_chat_template 标准调用回退 ({e})，使用兼容模式构建 Prompt...")
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
                messages.insert(0, {
                    "role": "system",
                    "content": "You are OmniEdu, a helpful AI assistant specialized in K-12 learning, curriculum grounding, diagnostic reasoning, and pedagogical tutoring." + tools_sys_prompt,
                })
        prompt_text = tokenizer.apply_chat_template(messages, **template_kwargs)

    prompt_opened_think = enable_thinking and prompt_text.rstrip().endswith(TAG_THINK_OPEN)

    if pil_images and "processor" in globals() and processor is not None:
        raw_inputs = processor(text=[prompt_text], images=pil_images, return_tensors="pt")
    else:
        # 模板已自带特殊 Token，add_special_tokens=False 可防止重复插入 BOS
        raw_inputs = tokenizer([prompt_text], return_tensors="pt", add_special_tokens=False)

    # 将 CPU 张量固定到锁页内存 (Pinned Memory)，使后续 .to(cuda, non_blocking=True) 走真正的异步 PCIe DMA
    pinned_inputs = {
        k: (v.pin_memory() if isinstance(v, torch.Tensor) and torch.cuda.is_available() else v)
        for k, v in raw_inputs.items()
    }
    return prompt_text, prompt_opened_think, pinned_inputs, len(pil_images)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    globals()["LAST_API_REQUEST_TIME"] = time.time()
    install_model_concurrent_hook(model)

    body = await request.json()
    raw_messages = body.get("messages", [])
    model_name = body.get("model", "lhpku20010120/Omni-Edu-27B")
    stream = bool(body.get("stream", False))
    req_max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or 16384
    req_temp = body.get("temperature", None)
    req_top_p = body.get("top_p", None)
    tools = body.get("tools", None)

    chat_template_kwargs = body.get("chat_template_kwargs") or {}
    preserve_thinking = body.get(
        "preserve_thinking",
        chat_template_kwargs.get("preserve_thinking", True),
    )

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
        enable_thinking = body.get(
            "enable_thinking",
            chat_template_kwargs.get("enable_thinking", True),
        )
        reasoning_effort = DEFAULT_REASONING_EFFORT

    if enable_thinking and FORCE_XHIGH_WHEN_THINKING:
        reasoning_effort = "xhigh"

    if enable_thinking:
        min_tok = MIN_XHIGH_MAX_TOKENS if reasoning_effort == "xhigh" else MIN_THINKING_MAX_TOKENS
        max_tokens = max(int(req_max_tokens), min_tok)
    else:
        max_tokens = int(req_max_tokens)

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
        f"并发={g4_scheduler.active_sessions}/{MAX_CONCURRENT_SESSIONS} | Tools:{len(tool_names)}"
    )

    # ==================== 分支 A：非流式请求 ====================
    if not stream:
        async with gpu_session_semaphore:
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

            # 注意：不显式传入 DynamicCache()，让 model.generate 自动创建混合架构专属缓存实例
            generation_kwargs = dict(
                **model_inputs,
                streamer=streamer,
                max_new_tokens=max_tokens,
                use_cache=True,
                stopping_criteria=StoppingCriteriaList([StopOnCancelCriteria(stop_event)]),
                **sampling_kwargs,
            )

            t0 = time.time()

            def run_gen_sync():
                with torch.inference_mode():
                    model.generate(**generation_kwargs)

            thread = threading.Thread(target=run_gen_sync, daemon=True)
            thread.start()

            chunks = []
            try:
                while thread.is_alive() or not streamer.text_queue.empty():
                    if await request.is_disconnected():
                        stop_event.set()
                        print(f"⏹️ [{req_short_id} 客户端已断开] 已立即中止该非流式推理。")
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
                thread.join(timeout=2.0)
                globals()["LAST_API_REQUEST_TIME"] = time.time()
                del model_inputs, generation_kwargs
                remaining = g4_scheduler.dec_session()
                if remaining == 0:
                    smart_vram_cleanup_if_needed()

            elapsed = time.time() - t0
            raw_output = "".join(chunks)
            reasoning_content, clean_content, tool_calls = parse_qwen_output(
                raw_output, prompt_opened_think=prompt_opened_think
            )
            completion_tokens = streamer.generated_tokens
            finish_reason = (
                "tool_calls" if tool_calls else ("length" if completion_tokens >= max_tokens else "stop")
            )
            print(
                f"📊 [{req_short_id} 非流式完成] 并发:{active_now} | Prompt:{prompt_tokens} tok | "
                f"Gen:{completion_tokens} tok | {completion_tokens / max(elapsed, 1e-5):.1f} tok/s | "
                f"工具:{[tc['function']['name'] for tc in tool_calls]}"
            )

            msg_obj = {
                "role": "assistant",
                "content": clean_content if clean_content else (None if tool_calls else ""),
            }
            if reasoning_content is not None:
                msg_obj["reasoning_content"] = reasoning_content
            if tool_calls:
                msg_obj["tool_calls"] = [{k: v for k, v in tc.items() if k != "index"} for tc in tool_calls]

            return JSONResponse({
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
            })

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

        acquire_task = asyncio.create_task(gpu_session_semaphore.acquire())
        try:
            while not acquire_task.done():
                if await request.is_disconnected():
                    acquire_task.cancel()
                    try:
                        await acquire_task
                    except asyncio.CancelledError:
                        gpu_session_semaphore.release()
                    return
                done, _ = await asyncio.wait({acquire_task}, timeout=2.0)
                if not done:
                    yield ": keep-alive-queued\n\n"
        except asyncio.CancelledError:
            if acquire_task.done() and not acquire_task.cancelled():
                gpu_session_semaphore.release()
            else:
                acquire_task.cancel()
            raise

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

        def run_gen_stream():
            with torch.inference_mode():
                model.generate(**generation_kwargs)

        thread = threading.Thread(target=run_gen_stream, daemon=True)
        thread.start()

        try:
            raw_buffer = ""
            pending_text = ""
            in_thinking = bool(prompt_opened_think)
            in_tool_call = False
            stripped_leading_after_think = False
            last_heartbeat = time.time()

            while True:
                if await request.is_disconnected():
                    stop_event.set()
                    print(f"⏹️ [{req_short_id} 客户端主动断开连接] 已立即中止该会话 GPU 推理。")
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
                raw_buffer += text_chunk

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

                    if TAG_THINK_OPEN in pending_text and TAG_THINK_CLOSE not in raw_buffer:
                        before_t, after_t = pending_text.split(TAG_THINK_OPEN, 1)
                        if before_t:
                            yield make_delta_chunk({"content": before_t})
                        in_thinking = True
                        pending_text = after_t.lstrip("\n")
                        continue

                    if TAG_TC_OPEN in pending_text:
                        in_tool_call = True
                        before_tc = pending_text.split(TAG_TC_OPEN, 1)[0]
                        if before_tc:
                            yield make_delta_chunk({"content": before_tc})
                            last_heartbeat = time.time()
                        pending_text = ""
                        continue

                    last_lt = pending_text.rfind("<")
                    if last_lt != -1 and (
                        TAG_TC_OPEN.startswith(pending_text[last_lt:])
                        or TAG_THINK_OPEN.startswith(pending_text[last_lt:])
                    ):
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

            thread.join(timeout=2.0)
            elapsed = time.time() - t0
            _, _, tool_calls = parse_qwen_output(
                raw_buffer, prompt_opened_think=prompt_opened_think
            )
            completion_tokens = streamer.generated_tokens
            called_names = [tc["function"]["name"] for tc in tool_calls]
            print(
                f"📊 [{req_short_id} 流式完成] 并发:{active_now} | Prompt:{prompt_tokens} tok | "
                f"Gen:{completion_tokens} tok | {completion_tokens / max(elapsed, 1e-5):.1f} tok/s | "
                f"触发工具:{called_names}"
            )

            if tool_calls:
                yield make_delta_chunk({"tool_calls": tool_calls})

            finish_reason = (
                "tool_calls" if tool_calls else ("length" if completion_tokens >= max_tokens else "stop")
            )
            usage_obj = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
            yield make_delta_chunk({}, finish_reason=finish_reason, usage_dict=usage_obj)
            yield "data: [DONE]\n\n"

        except (asyncio.CancelledError, BrokenPipeError, ConnectionResetError):
            stop_event.set()
            print(f"⏹️ [{req_short_id} 流式传输中断] 已安全回收连接并停止该会话 GPU 生成。")
        finally:
            stop_event.set()
            globals()["LAST_API_REQUEST_TIME"] = time.time()
            del model_inputs, generation_kwargs
            remaining = g4_scheduler.dec_session()
            gpu_session_semaphore.release()
            if remaining == 0:
                smart_vram_cleanup_if_needed()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@app.get("/v1/models")
@app.get("/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "lhpku20010120/Omni-Edu-27B",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "omniedu",
            }
        ],
    }


if "uvicorn_server" in globals() and uvicorn_server is not None:
    uvicorn_server.should_exit = True
    time.sleep(1.5)

config = uvicorn.Config(
    app,
    host="0.0.0.0",
    port=8000,
    log_level="critical",
    access_log=False,
    timeout_keep_alive=120,
)
uvicorn_server = uvicorn.Server(config)


def start_isolated_uvicorn():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(uvicorn_server.serve())


threading.Thread(target=start_isolated_uvicorn, daemon=True).start()
print(
    f"✅ Omni-Edu-27B (G4 Blackwell 96GB) 终极并发版 FastAPI 已启动 (端口 8000 | 最大并发会话={MAX_CONCURRENT_SESSIONS} | 支持多模态看图)"
)
