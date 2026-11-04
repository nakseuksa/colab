import os
import sys
import glob
import torch
from google.colab import drive

# 1. 挂载 Google Drive（带已挂载检测与容错降级）
drive_mounted = os.path.ismount("/content/drive")
if not drive_mounted:
    try:
        drive.mount("/content/drive", force_remount=True)
        drive_mounted = True
        print("✅ Google Drive 挂载成功！")
    except Exception as e:
        print(f"⚠️ Google Drive 挂载失败 ({e})，已自动切换为本地临时目录缓存。")
else:
    print("✅ 检测到 Google Drive 已挂载，直接复用！")

# 2. 动态识别当前 GPU 架构（G4 RTX PRO 6000 Blackwell 为 sm_120）与 Python 版本
major, minor = torch.cuda.get_device_capability()
sm_arch = f"{major}{minor}"
cuda_arch_str = f"{major}.{minor}"
py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
gpu_name = torch.cuda.get_device_name(0)
print(f"🔍 当前运行环境: GPU={gpu_name} (sm_{sm_arch}), Python={py_tag}")

# 3. 持久化 Triton 算子编译缓存到云盘（避免每次重启 Colab 重新编译 flash-linear-attention 耗时 30 秒）
base_cache_dir = "/content/drive/MyDrive/colab_wheels" if drive_mounted else "/tmp/colab_wheels"
triton_cache_dir = f"{base_cache_dir}/triton_cache_sm_{sm_arch}_{py_tag}"
os.makedirs(triton_cache_dir, exist_ok=True)
os.environ["TRITON_CACHE_DIR"] = triton_cache_dir
print(f"⚡ Triton 硬件算子缓存目录已绑定: {triton_cache_dir}")

# 4. 安装核心依赖（复用 Colab 自带 pillow，避免触发 gradio/google-adk 的无关版本警告）
!pip install -q -U ninja accelerate transformers hf_transfer "flash-linear-attention>=0.5.2" "fla-core>=0.5.2"
!pip install -q fastapi uvicorn sse-starlette


# 5. 按【架构 + Python版本】隔离 causal-conv1d 预编译包
wheel_dir = f"{base_cache_dir}/sm_{sm_arch}_{py_tag}"
existing_wheels = glob.glob(f"{wheel_dir}/causal_conv1d-*.whl")

if existing_wheels:
    print(f"✅ 检测到匹配当前架构 (sm_{sm_arch}, {py_tag}) 的预编译包: {existing_wheels[0]}，直接秒装！")
    !pip install -q "{existing_wheels[0]}"
else:
    cpu_count = os.cpu_count() or 8
    print(f"⚙️ 未检测到 sm_{sm_arch} ({py_tag}) 预编译包，开始使用 {cpu_count} 核并行编译...")
    os.makedirs(wheel_dir, exist_ok=True)
    !rm -rf /tmp/causal-conv1d
    !git clone --depth 1 https://github.com/Dao-AILab/causal-conv1d.git /tmp/causal-conv1d

    setup_path = "/tmp/causal-conv1d/setup.py"
    with open(setup_path, "r", encoding="utf-8") as f:
        setup_code = f.read()
    target_gencode = f'+ ["-gencode", "arch=compute_{sm_arch},code=sm_{sm_arch}"]'
    setup_code = setup_code.replace("+ cc_flag", target_gencode)
    with open(setup_path, "w", encoding="utf-8") as f:
        f.write(setup_code)

    !TORCH_CUDA_ARCH_LIST="{cuda_arch_str}" CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS={cpu_count} pip wheel /tmp/causal-conv1d --no-build-isolation --no-deps -w "{wheel_dir}" -q
    !pip install -q "{wheel_dir}"/causal_conv1d-*.whl
    print(f"🎉 针对 sm_{sm_arch} 编译并安装完成！下次启动将直接秒装。")
