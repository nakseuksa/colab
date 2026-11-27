"""Qwen3.8-27B (G4 Blackwell / Transformers + FLA) Cell 1: 环境初始化、云盘缓存与 causal-conv1d 预编译。

Colab CLI 用法:
    colab new -s qwen38 --gpu G4
    colab drivemount -s qwen38
    colab exec -s qwen38 --timeout 0 -f qwen38_27b/cell1.py
"""

import os

# 必须在 import torch / huggingface_hub 之前设置底层分配器与高速下载环境变量
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import glob
import re
import shutil
import subprocess
import sys

import torch

ALLOW_INTERACTIVE_MOUNT = os.environ.get("COLAB_AUTO_MOUNT_DRIVE", "0").strip().lower() in ("1", "true", "yes")
SKIP_PIP_INSTALL = os.environ.get("SKIP_PIP_INSTALL", "0").strip().lower() in ("1", "true", "yes")

CORE_PACKAGES = [
    "ninja",
    "accelerate",
    "transformers",
    "hf_transfer",
    "flash-linear-attention>=0.5.2",
    "fla-core>=0.5.2",
    "fastapi",
    "uvicorn",
    "sse-starlette",
]


def mount_google_drive(mount_point: str = "/content/drive") -> bool:
    """幂等检测/挂载 Google Drive（CLI 模式下默认跳过阻塞式交互挂载）。"""
    if os.path.ismount(mount_point):
        print("✅ 检测到 Google Drive 已挂载，直接复用！", flush=True)
        return True

    if not ALLOW_INTERACTIVE_MOUNT:
        print(
            f"ℹ️ 未检测到已挂载的 Google Drive ({mount_point})，已切换为本地临时目录缓存。\n"
            "   （如需持久化 wheel 与 Triton 缓存，请先运行: colab drivemount -s <session>）",
            flush=True,
        )
        return False

    try:
        from google.colab import drive

        drive.mount(mount_point, force_remount=True)
        print("✅ Google Drive 挂载成功！", flush=True)
        return True
    except Exception as exc:
        print(f"⚠️ Google Drive 挂载失败 ({exc})，已自动切换为本地临时目录缓存。", flush=True)
        return False


def _sanitize_tag(val: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", val or "unknown")


def run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print(f"🔧 执行: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def main() -> None:
    drive_mounted = mount_google_drive()

    if not torch.cuda.is_available():
        raise RuntimeError("❌ 未检测到可用 CUDA GPU！请使用 `colab new --gpu G4`（或 A100/H100）申请 GPU 实例。")

    major, minor = torch.cuda.get_device_capability(0)
    sm_arch = f"{major}{minor}"
    cuda_arch_str = f"{major}.{minor}"
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    torch_tag = _sanitize_tag(torch.__version__.split("+")[0])
    cuda_tag = _sanitize_tag(torch.version.cuda or "nocuda")
    abi_tag = f"sm_{sm_arch}_{py_tag}_pt{torch_tag}_cu{cuda_tag}"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"🔍 当前运行环境: GPU={gpu_name} (sm_{sm_arch}), Python={py_tag}, PyTorch={torch.__version__}", flush=True)

    # 持久化 Triton 算子编译缓存
    base_cache_dir = "/content/drive/MyDrive/colab_wheels" if drive_mounted else "/tmp/colab_wheels"
    triton_cache_dir = f"{base_cache_dir}/triton_cache_{abi_tag}"
    os.makedirs(triton_cache_dir, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = triton_cache_dir
    print(f"⚡ Triton 硬件算子缓存目录已绑定: {triton_cache_dir}", flush=True)

    # 安装核心依赖（流式输出，避免 CLI 静默超时）
    if not SKIP_PIP_INSTALL:
        run_cmd([sys.executable, "-m", "pip", "install", "-U", "--progress-bar", "off", *CORE_PACKAGES])
    else:
        print("⏭️ SKIP_PIP_INSTALL=1，跳过基础包安装。", flush=True)

    # 按【架构 + Python + PyTorch + CUDA ABI】隔离 causal-conv1d 预编译包，防止 Colab 升级 PyTorch 后 ABI 崩溃
    wheel_dir = f"{base_cache_dir}/{abi_tag}"
    existing_wheels = sorted(glob.glob(f"{wheel_dir}/causal_conv1d-*.whl"))

    if existing_wheels:
        chosen_wheel = existing_wheels[-1]
        print(f"✅ 检测到匹配当前 ABI ({abi_tag}) 的预编译包: {chosen_wheel}，直接秒装！", flush=True)
        run_cmd([sys.executable, "-m", "pip", "install", "--progress-bar", "off", chosen_wheel])
    else:
        cpu_count = os.cpu_count() or 8
        print(f"⚙️ 未检测到 {abi_tag} 预编译包，开始使用 {cpu_count} 核并行编译 causal-conv1d...", flush=True)
        os.makedirs(wheel_dir, exist_ok=True)

        repo_tmp = "/tmp/causal-conv1d"
        if os.path.exists(repo_tmp):
            shutil.rmtree(repo_tmp)
        run_cmd(["git", "clone", "--depth", "1", "https://github.com/Dao-AILab/causal-conv1d.git", repo_tmp])

        setup_path = os.path.join(repo_tmp, "setup.py")
        with open(setup_path, "r", encoding="utf-8") as f:
            setup_code = f.read()
        target_gencode = f'+ ["-gencode", "arch=compute_{sm_arch},code=sm_{sm_arch}"]'
        if "+ cc_flag" in setup_code:
            setup_code = setup_code.replace("+ cc_flag", target_gencode)
            with open(setup_path, "w", encoding="utf-8") as f:
                f.write(setup_code)
            print(f"🩹 已为 setup.py 注入目标架构 gencode=sm_{sm_arch}", flush=True)
        else:
            print(
                f"⚠️ 上游 causal-conv1d/setup.py 未匹配到 '+ cc_flag'，将依赖 TORCH_CUDA_ARCH_LIST={cuda_arch_str} 编译。",
                flush=True,
            )

        build_env = os.environ.copy()
        build_env["TORCH_CUDA_ARCH_LIST"] = cuda_arch_str
        build_env["CAUSAL_CONV1D_FORCE_BUILD"] = "TRUE"
        build_env["MAX_JOBS"] = str(cpu_count)

        run_cmd(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                repo_tmp,
                "--no-build-isolation",
                "--no-deps",
                "-w",
                wheel_dir,
            ],
            env=build_env,
        )
        built_wheels = sorted(glob.glob(f"{wheel_dir}/causal_conv1d-*.whl"))
        if not built_wheels:
            raise RuntimeError(f"❌ 编译完成但未在 {wheel_dir} 找到生成的 causal_conv1d wheel！")
        run_cmd([sys.executable, "-m", "pip", "install", "--progress-bar", "off", built_wheels[-1]])
        print(f"🎉 针对 {abi_tag} 编译并安装完成！下次启动将直接秒装。", flush=True)


if __name__ == "__main__":
    main()
