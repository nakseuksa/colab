"""Cell 2: 在后台启动针对 H100 (80GB sm_90) 优化的 vLLM 满血推理服务 (Port 8001)。"""

import os
import json
import time
import signal
import shutil
import subprocess
import urllib.request
import urllib.error

# ==========================================================
# 0. 核心显式配置区（支持环境变量覆写，按需一键切换 BF16 / FP8 与 MTP 投机解码）
# ==========================================================
MODEL_PRECISION = os.environ.get("MODEL_PRECISION", "bf16").strip().lower()
MODEL_ID_MAP = {
    "bf16": os.environ.get("MODEL_ID_BF16", "Qwen/Qwen3.8-27B"),
    "fp8": os.environ.get("MODEL_ID_FP8", "Qwen/Qwen3.8-27B-FP8"),
}
SERVED_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "Qwen/Qwen3.8-27B")

VLLM_HOST = os.environ.get("VLLM_HOST", "127.0.0.1")
VLLM_INTERNAL_PORT = int(os.environ.get("VLLM_INTERNAL_PORT", "8001"))
VLLM_BASE_URL = f"http://{VLLM_HOST}:{VLLM_INTERNAL_PORT}/v1"

ENABLE_MTP_SPECULATIVE = os.environ.get("ENABLE_MTP_SPECULATIVE", "1").strip().lower() not in ("0", "false", "no")
MTP_NUM_SPECULATIVE_TOKENS = int(os.environ.get("MTP_NUM_SPECULATIVE_TOKENS", "3"))

KV_CACHE_DTYPE = os.environ.get("KV_CACHE_DTYPE", "fp8").strip().lower()
MAX_MM_IMAGES_PER_PROMPT = int(os.environ.get("MAX_MM_IMAGES_PER_PROMPT", "8"))

VLLM_LOG_PATH = os.environ.get("VLLM_LOG_PATH", "/tmp/vllm_server.log")
STARTUP_TIMEOUT_SECONDS = int(os.environ.get("STARTUP_TIMEOUT_SECONDS", "600"))
# ==========================================================

VLLM_PROCESS = globals().get("VLLM_PROCESS", None)
ACTIVE_MODEL_ID = MODEL_ID_MAP.get(MODEL_PRECISION, SERVED_MODEL_NAME)


def resolve_model_id(precision: str) -> str:
    """根据精度档位解析 Hugging Face 仓库 ID。"""
    key = str(precision).strip().lower()
    if key not in MODEL_ID_MAP:
        raise ValueError(f"不支持的 MODEL_PRECISION: {precision!r}，可选: {list(MODEL_ID_MAP.keys())}")
    return MODEL_ID_MAP[key]


def detect_cuda_sm_major() -> int | None:
    """探测当前 GPU 的 SM 主版本号（Hopper H100 为 9），无 CUDA 时返回 None。"""
    try:
        import torch

        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(0)
            return int(major)
    except Exception:
        pass
    return None


def resolve_kv_cache_dtype(requested_dtype: str, sm_major: int | None) -> str:
    """Hopper (sm_90) / Blackwell (sm_100+) 支持 fp8 KV Cache；旧架构自动回退 auto。"""
    req = str(requested_dtype).strip().lower()
    if req != "fp8":
        return req
    if sm_major is not None and sm_major >= 9:
        return "fp8"
    return "auto"


def get_precision_profile(precision: str, effective_kv_cache_dtype: str) -> dict:
    """返回与模型精度及 KV Cache 精度匹配的 H100 推荐参数档位。"""
    key = str(precision).strip().lower()
    if key == "bf16":
        max_len = 65536 if effective_kv_cache_dtype != "auto" else 32768
        return {
            "max_model_len": max_len,
            "gpu_memory_utilization": 0.92,
            "max_num_seqs": 8,
            "max_num_batched_tokens": 16384,
            "dtype": "bfloat16",
        }
    if key == "fp8":
        return {
            "max_model_len": 131072,
            "gpu_memory_utilization": 0.90,
            "max_num_seqs": 16,
            "max_num_batched_tokens": 32768,
            "dtype": "auto",
        }
    raise ValueError(f"不支持的 MODEL_PRECISION: {precision!r}")


def build_vllm_serve_cmd(
    model_id: str,
    served_model_name: str = SERVED_MODEL_NAME,
    host: str = VLLM_HOST,
    port: int = VLLM_INTERNAL_PORT,
    precision: str = MODEL_PRECISION,
    kv_cache_dtype: str = KV_CACHE_DTYPE,
    enable_mtp: bool = ENABLE_MTP_SPECULATIVE,
    mtp_num_speculative_tokens: int = MTP_NUM_SPECULATIVE_TOKENS,
    max_mm_images: int = MAX_MM_IMAGES_PER_PROMPT,
    sm_major: int | None = 9,
) -> list:
    """构建标准 vllm serve 启动命令参数列表（纯函数，便于单测验证）。"""
    effective_kv = resolve_kv_cache_dtype(kv_cache_dtype, sm_major=sm_major)
    profile = get_precision_profile(precision, effective_kv)

    cmd = [
        "vllm",
        "serve",
        model_id,
        "--served-model-name",
        served_model_name,
        "--host",
        host,
        "--port",
        str(port),
        "--tensor-parallel-size",
        "1",
        "--dtype",
        profile["dtype"],
        "--kv-cache-dtype",
        effective_kv,
        "--max-model-len",
        str(profile["max_model_len"]),
        "--gpu-memory-utilization",
        str(profile["gpu_memory_utilization"]),
        "--max-num-seqs",
        str(profile["max_num_seqs"]),
        "--max-num-batched-tokens",
        str(profile["max_num_batched_tokens"]),
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--reasoning-parser",
        "qwen3",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_coder",
        "--default-chat-template-kwargs",
        json.dumps({"preserve_thinking": True}),
        "--limit-mm-per-prompt",
        json.dumps({"image": int(max_mm_images), "video": 0}),
        "--trust-remote-code",
    ]

    if enable_mtp:
        spec_cfg = {
            "method": "mtp",
            "num_speculative_tokens": int(mtp_num_speculative_tokens),
        }
        cmd.extend(["--speculative-config", json.dumps(spec_cfg)])

    return cmd


def read_hf_token() -> str | None:
    """安全读取 HF_TOKEN（环境变量优先以适配 Colab CLI，其次回退 Colab Secrets），禁止硬编码。"""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token and str(token).strip():
        return str(token).strip()

    try:
        from google.colab import userdata

        token = userdata.get("HF_TOKEN")
        if token and str(token).strip():
            return str(token).strip()
    except Exception:
        pass
    return None


def _terminate_proc_or_pgroup(proc) -> None:
    """在 POSIX 下优先按进程组终止 vLLM 及其衍生的 EngineCore GPU 子进程，失败时回退单进程终止。"""
    if proc is None or proc.poll() is not None:
        return

    if os.name == "posix" and hasattr(os, "killpg") and hasattr(os, "getpgid"):
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                proc.wait(timeout=5)
            return
        except Exception:
            pass

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def cleanup_existing_vllm_process(port: int, existing_proc=None) -> None:
    """幂等清理上一次运行遗留的 VLLM_PROCESS（含子进程组）及占用指定端口的残留进程。"""
    if existing_proc is not None:
        try:
            if existing_proc.poll() is None:
                print("🧹 正在优雅停止旧的 VLLM_PROCESS 进程组 ...")
                _terminate_proc_or_pgroup(existing_proc)
        except Exception as exc:
            print(f"ℹ️ 清理旧 VLLM_PROCESS 提示: {exc}")

    # 兜底清理端口残留（兼容 Colab 重新运行 Cell 导致的孤儿进程）
    pids = set()
    if shutil.which("lsof"):
        res = subprocess.run(
            ["lsof", "-t", f"-i:{port}"],
            capture_output=True,
            text=True,
            check=False,
        )
        for line in res.stdout.splitlines():
            line = line.strip()
            if line.isdigit():
                pids.add(int(line))
    elif shutil.which("fuser"):
        subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True, check=False)

    current_pid = os.getpid()
    target_pids = [pid for pid in pids if pid != current_pid]
    for pid in target_pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass

    if target_pids:
        deadline = time.time() + 10.0
        alive = set(target_pids)
        while alive and time.time() < deadline:
            for pid in list(alive):
                try:
                    os.kill(pid, 0)
                except OSError:
                    alive.discard(pid)
            if alive:
                time.sleep(0.5)
        for pid in alive:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        if alive:
            time.sleep(1.0)


def read_log_tail(log_path: str, max_lines: int = 40) -> str:
    """读取 vLLM 服务日志尾部内容用于故障诊断。"""
    if not os.path.exists(log_path):
        return "(日志文件不存在)"
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-max_lines:]).strip()
    except Exception as exc:
        return f"(读取日志失败: {exc})"


def wait_for_vllm_ready(
    proc,
    health_url: str,
    timeout_seconds: int = STARTUP_TIMEOUT_SECONDS,
    poll_interval: float = 2.0,
    log_path: str = VLLM_LOG_PATH,
    progress_interval: float = 15.0,
) -> tuple[bool, str]:
    """轮询 vLLM /health 端点直至就绪，定期打印进度心跳防止 Colab CLI 静默超时。"""
    t0 = time.time()
    deadline = t0 + timeout_seconds
    last_report = t0
    last_err = ""
    while time.time() < deadline:
        exit_code = proc.poll()
        if exit_code is not None:
            return False, f"vLLM 进程已提前退出 (exit code={exit_code})"

        try:
            with urllib.request.urlopen(health_url, timeout=3) as resp:
                if resp.status == 200:
                    return True, "ok"
        except Exception as exc:
            last_err = str(exc)

        now = time.time()
        if now - last_report >= progress_interval:
            elapsed = now - t0
            tail = read_log_tail(log_path, max_lines=1)
            short_tail = (tail[:110] + "...") if len(tail) > 110 else tail
            print(f"⏳ 等待 vLLM 就绪 ({elapsed:.0f}s/{timeout_seconds}s) | 最新日志: {short_tail}", flush=True)
            last_report = now

        time.sleep(poll_interval)

    return False, f"等待超时 ({timeout_seconds}s)，最后一次检查异常: {last_err}"


def query_nvidia_smi_vram_gb() -> tuple[float, float] | None:
    """通过 nvidia-smi 读取子进程 vLLM 真实的 GPU 显存占用 (used_gb, total_gb)。"""
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0 or not res.stdout.strip():
            return None
        first_line = res.stdout.strip().splitlines()[0]
        used_mib, total_mib = [float(x.strip()) for x in first_line.split(",")[:2]]
        return used_mib / 1024.0, total_mib / 1024.0
    except Exception:
        return None


def send_warmup_request(base_url: str, model_name: str) -> float:
    """向已就绪的 vLLM 发送轻量级 Warmup 请求以预热 CUDA Graph 与推理核。"""
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Warmup"}],
        "max_tokens": 4,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as resp:
        _ = resp.read()
    return time.time() - t0


def start_vllm_server() -> tuple[subprocess.Popen, str, bool]:
    """启动 vLLM 后台服务，支持开启 MTP 失败时自动回退无 MTP 重试一次。"""
    model_id = resolve_model_id(MODEL_PRECISION)
    sm_major = detect_cuda_sm_major()
    effective_kv = resolve_kv_cache_dtype(KV_CACHE_DTYPE, sm_major=sm_major)
    if KV_CACHE_DTYPE == "fp8" and effective_kv != "fp8":
        print(f"⚠️ 检测到当前 GPU SM={sm_major} 不支持 FP8 KV Cache，已自动回退为 kv_cache_dtype='auto'。")

    hf_token = read_hf_token()
    child_env = os.environ.copy()
    child_env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    if hf_token:
        child_env["HF_TOKEN"] = hf_token
        child_env["HUGGING_FACE_HUB_TOKEN"] = hf_token
        print("✅ 已加载 HF_TOKEN（仅通过子进程环境变量传递）。")
    else:
        print("ℹ️ 未配置 HF_TOKEN，将以匿名高速模式拉取公开模型权重。")

    cleanup_existing_vllm_process(VLLM_INTERNAL_PORT, existing_proc=globals().get("VLLM_PROCESS"))

    health_url = f"http://{VLLM_HOST}:{VLLM_INTERNAL_PORT}/health"
    attempts: list[tuple[bool, str]] = [(ENABLE_MTP_SPECULATIVE, effective_kv)]
    if ENABLE_MTP_SPECULATIVE:
        attempts.append((False, effective_kv))
    if effective_kv == "fp8":
        attempts.append((False, "auto"))

    last_reason = ""
    for idx, (use_mtp, kv_dtype) in enumerate(attempts):
        cmd = build_vllm_serve_cmd(
            model_id=model_id,
            served_model_name=SERVED_MODEL_NAME,
            host=VLLM_HOST,
            port=VLLM_INTERNAL_PORT,
            precision=MODEL_PRECISION,
            kv_cache_dtype=kv_dtype,
            enable_mtp=use_mtp,
            mtp_num_speculative_tokens=MTP_NUM_SPECULATIVE_TOKENS,
            max_mm_images=MAX_MM_IMAGES_PER_PROMPT,
            sm_major=sm_major,
        )
        mode_tag = "MTP 投机解码" if use_mtp else "标准自回归解码 (MTP 已关闭)"
        print(
            f"🚀 [{idx + 1}/{len(attempts)}] 正在启动 vLLM "
            f"({MODEL_PRECISION.upper()} -> {model_id} | {mode_tag} | kv_cache_dtype={kv_dtype})..."
        )
        print(f"📝 服务日志输出至: {VLLM_LOG_PATH}")

        log_file = open(VLLM_LOG_PATH, "w", encoding="utf-8")
        popen_kwargs = {
            "stdout": log_file,
            "stderr": subprocess.STDOUT,
            "env": child_env,
            "text": True,
        }
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True
        proc = subprocess.Popen(cmd, **popen_kwargs)

        ok, reason = wait_for_vllm_ready(proc, health_url=health_url, timeout_seconds=STARTUP_TIMEOUT_SECONDS)
        if ok:
            log_file.close()
            return proc, model_id, use_mtp

        last_reason = reason
        log_file.close()
        print(f"❌ vLLM 启动失败 ({reason})，尾部日志如下:\n" + "-" * 60)
        print(read_log_tail(VLLM_LOG_PATH, max_lines=40))
        print("-" * 60)

        cleanup_existing_vllm_process(VLLM_INTERNAL_PORT, existing_proc=proc)
        if idx + 1 < len(attempts):
            next_mtp, next_kv = attempts[idx + 1]
            if use_mtp and not next_mtp and kv_dtype == next_kv:
                print("🔄 检测到开启 MTP 时启动失败，正在自动关闭 --speculative-config 回退重试...")
            elif next_kv != kv_dtype:
                print(
                    f"🔄 检测到 kv_cache_dtype={kv_dtype!r} 启动失败，"
                    f"正在自动回退至 kv_cache_dtype={next_kv!r} (MTP={'开启' if next_mtp else '关闭'}) 重试..."
                )

    raise RuntimeError(f"❌ vLLM 服务启动彻底失败: {last_reason}（完整日志见 {VLLM_LOG_PATH}）")


if __name__ == "__main__":
    t_start = time.time()
    VLLM_PROCESS, ACTIVE_MODEL_ID, ACTIVE_MTP_ENABLED = start_vllm_server()
    print(f"✅ vLLM 服务已就绪 (耗时 {time.time() - t_start:.1f}s | MTP={'开启' if ACTIVE_MTP_ENABLED else '关闭'})")

    try:
        warmup_s = send_warmup_request(VLLM_BASE_URL, SERVED_MODEL_NAME)
        print(f"🔥 Warmup 预热请求完成 (耗时 {warmup_s:.2f}s)")
    except Exception as exc:
        print(f"⚠️ Warmup 请求异常（不影响后续调用）: {exc}")

    vram_stats = query_nvidia_smi_vram_gb()
    if vram_stats is not None:
        used_gb, total_gb = vram_stats
        print(
            f"📊 H100 真实显存占用 (nvidia-smi): 已用 {used_gb:.2f} GB / 总计 {total_gb:.2f} GB "
            f"(剩余 {max(total_gb - used_gb, 0.0):.2f} GB)"
        )

    print(
        f"🔌 内部推理端点已暴露: VLLM_BASE_URL={VLLM_BASE_URL} | "
        f"SERVED_MODEL_NAME={SERVED_MODEL_NAME} | ACTIVE_MODEL_ID={ACTIVE_MODEL_ID}"
    )
