"""Cell 1: Google Colab H100 (80GB HBM3, Hopper sm_90) 环境初始化、云盘缓存持久化与依赖安装。"""

import os
import sys
import subprocess

CORE_DEPENDENCIES = [
    "vllm",
    "transformers",
    "accelerate",
    "hf_transfer",
    "qwen-vl-utils",
    "openai",
    "httpx",
    "fastapi",
    "uvicorn",
    "sse-starlette",
]


def mount_google_drive(mount_point: str = "/content/drive") -> bool:
    """幂等挂载 Google Drive，失败时优雅降级返回 False。"""
    if os.path.ismount(mount_point):
        print("✅ 检测到 Google Drive 已挂载，直接复用！")
        return True

    try:
        from google.colab import drive

        drive.mount(mount_point, force_remount=True)
        print("✅ Google Drive 挂载成功！")
        return True
    except Exception as exc:
        print(f"⚠️ Google Drive 挂载失败 ({exc})，已自动切换为本地临时目录缓存。")
        return False


def detect_gpu_info() -> dict:
    """检测当前 GPU 架构与显存信息（预期 H100 80GB sm_90），非 H100 时给出友好提示但不崩溃。"""
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    info = {
        "cuda_available": False,
        "gpu_name": "Unknown GPU",
        "sm_major": 0,
        "sm_minor": 0,
        "sm_arch": "cpu",
        "total_vram_gb": 0.0,
        "py_tag": py_tag,
        "is_h100": False,
    }

    try:
        import torch
    except ImportError:
        print(f"ℹ️ 当前环境尚未导入 torch (Python={py_tag})，跳过 GPU 预检，将在安装依赖后生效。")
        return info

    if not torch.cuda.is_available():
        print("⚠️ 未检测到可用 CUDA GPU！请确认 Colab 运行时已切换至 NVIDIA H100。")
        return info

    major, minor = torch.cuda.get_device_capability(0)
    sm_arch = f"{major}{minor}"
    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    is_h100 = (sm_arch == "90") or ("H100" in gpu_name.upper())

    info.update(
        {
            "cuda_available": True,
            "gpu_name": gpu_name,
            "sm_major": major,
            "sm_minor": minor,
            "sm_arch": sm_arch,
            "total_vram_gb": total_vram_gb,
            "is_h100": is_h100,
        }
    )

    print(
        f"🔍 当前运行环境: GPU={gpu_name} (sm_{sm_arch}), "
        f"显存={total_vram_gb:.1f} GB, Python={py_tag}"
    )
    if not is_h100:
        print(
            f"⚠️ 提示: 当前 GPU 为 {gpu_name} (sm_{sm_arch})，并非 H100 (sm_90)。"
            "脚本仍将继续执行，但如非 Hopper/Blackwell 架构，Cell 2 将自动回退 KV Cache 精度。"
        )
    elif total_vram_gb < 75.0:
        print(
            f"⚠️ 提示: 检测到 H100 显存为 {total_vram_gb:.1f} GB（低于 80GB），"
            "如遇到显存紧张，可在 Cell 2 将 MODEL_PRECISION 切换为 'fp8'。"
        )
    else:
        print("⚡ 已确认 NVIDIA H100 (Hopper sm_90) 大显存硬件，可满血运行 Qwen3.8-27B！")

    return info


def configure_persistent_caches(
    drive_mounted: bool,
    sm_arch: str,
    py_tag: str,
    drive_base: str = "/content/drive/MyDrive/colab_wheels",
    local_base: str = "/tmp/colab_wheels",
) -> dict:
    """持久化 Triton / vLLM / TorchInductor 编译缓存，按 sm_{arch}_{py_tag} 隔离。"""
    base_cache_dir = drive_base if drive_mounted else local_base
    env_tag = f"sm_{sm_arch}_{py_tag}"

    cache_dirs = {
        "TRITON_CACHE_DIR": os.path.join(base_cache_dir, f"triton_cache_{env_tag}"),
        "VLLM_CACHE_ROOT": os.path.join(base_cache_dir, f"vllm_cache_{env_tag}"),
        "TORCHINDUCTOR_CACHE_DIR": os.path.join(base_cache_dir, f"torchinductor_cache_{env_tag}"),
    }

    for env_key, dir_path in cache_dirs.items():
        os.makedirs(dir_path, exist_ok=True)
        os.environ[env_key] = dir_path

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    print(f"⚡ 编译缓存已绑定 ({env_tag}):")
    for env_key, dir_path in cache_dirs.items():
        print(f"   - {env_key} = {dir_path}")
    print("   - HF_HUB_ENABLE_HF_TRANSFER = 1")

    return {**cache_dirs, "BASE_CACHE_DIR": base_cache_dir}


def install_dependencies(packages: list | None = None) -> subprocess.CompletedProcess:
    """使用 subprocess.run 安装 vLLM 与网关核心依赖。"""
    pkg_list = list(packages) if packages is not None else list(CORE_DEPENDENCIES)
    cmd = [sys.executable, "-m", "pip", "install", "-q", "-U", *pkg_list]
    print(f"📦 正在安装 H100 vLLM 推理及网关依赖 ({len(pkg_list)} 个包): {' '.join(pkg_list)}")

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"❌ 依赖安装失败 (exit code={result.returncode})。"
            "请检查 Colab 网络连接，或尝试手动运行: "
            f"{' '.join(cmd)}"
        )

    print("✅ 核心依赖安装完成！")
    return result


def print_environment_summary() -> dict:
    """打印安装后的 vllm / torch / transformers 版本及 Hopper 就绪状态。"""
    versions = {}
    for mod_name in ("vllm", "torch", "transformers"):
        try:
            mod = __import__(mod_name)
            versions[mod_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[mod_name] = "not installed"

    cuda_info = "N/A"
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            cuda_info = f"{torch.cuda.get_device_name(0)} (sm_{major}{minor}, CUDA {torch.version.cuda})"
    except Exception:
        pass

    print("\n" + "=" * 68)
    print("🚀 H100 vLLM 满血推理环境初始化完成！")
    print(f"   - vLLM        : {versions['vllm']}")
    print(f"   - PyTorch     : {versions['torch']}")
    print(f"   - Transformers: {versions['transformers']}")
    print(f"   - GPU / CUDA  : {cuda_info}")
    print("=" * 68)
    return versions


if __name__ == "__main__":
    DRIVE_MOUNTED = mount_google_drive()
    GPU_INFO = detect_gpu_info()
    CACHE_PATHS = configure_persistent_caches(
        drive_mounted=DRIVE_MOUNTED,
        sm_arch=GPU_INFO["sm_arch"],
        py_tag=GPU_INFO["py_tag"],
    )
    install_dependencies()
    print_environment_summary()
