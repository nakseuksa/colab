import os
import shutil
import site
import subprocess
from pathlib import Path

# ==========================================================
# 1. 将模型从 Google Drive 拷贝到 Colab 本地高速磁盘 (/content)
# ==========================================================
MINERU_HOME = "/content/drive/MyDrive/mineru_home"
DRIVE_MODELS_DIR = f"{MINERU_HOME}/models"
LOCAL_MODELS_DIR = "/content/mineru_models"  # Colab 本地高速 SSD 路径

os.environ["MINERU_HOME"] = MINERU_HOME
os.environ["MINERU_MODEL_SOURCE"] = "huggingface"

if not os.path.exists(LOCAL_MODELS_DIR):
    print(f"⏳ 正在将模型从 Google Drive 拷贝到 Colab 本地高速磁盘: {LOCAL_MODELS_DIR} ...")
    shutil.copytree(DRIVE_MODELS_DIR, LOCAL_MODELS_DIR)
    print("✅ 模型拷贝完成！")
else:
    print(f"✅ 本地高速磁盘已存在模型目录: {LOCAL_MODELS_DIR}，跳过拷贝。")

# 将 config.yaml 中的 base_dir 指向 Colab 本地高速磁盘
config_path = Path(MINERU_HOME) / "config.yaml"
config_path.parent.mkdir(parents=True, exist_ok=True)
config_path.write_text(f"""model:
  base_dir: {LOCAL_MODELS_DIR}
  source: auto
  small_backend: auto
  vlm:
    engine: auto
""", encoding="utf-8")
print(f"✅ MinerU 配置已指向本地高速磁盘: {LOCAL_MODELS_DIR}")

# ==========================================================
# 2. 修复 Colab 的 cuDNN 子库版本不匹配问题 (锁死 cudnn.enabled=False)
# ==========================================================
dist_packages = site.getsitepackages()[0]
hook_py = Path(dist_packages) / "_disable_cudnn_hook.py"
pth_file = Path(dist_packages) / "zz_disable_cudnn.pth"

hook_code = """
import torch
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

try:
    _cudnn_cls = torch.backends.cudnn.__class__
    _cudnn_cls.enabled = property(lambda self: False, lambda self, val: None)
    _cudnn_cls.benchmark = property(lambda self: False, lambda self, val: None)
except Exception:
    pass
"""

hook_py.write_text(hook_code.strip() + "\n", encoding="utf-8")
pth_file.write_text("import _disable_cudnn_hook\n", encoding="utf-8")

sys_sc = Path("/usr/lib/python3.13/sitecustomize.py")
if sys_sc.exists() and "_disable_cudnn_hook" not in sys_sc.read_text():
    with open(sys_sc, "a", encoding="utf-8") as f:
        f.write("\nimport _disable_cudnn_hook\n")

check = subprocess.run(
    ["python3", "-c", "import torch; torch.backends.cudnn.enabled = True; print('子进程 cudnn.enabled =', torch.backends.cudnn.enabled)"],
    capture_output=True, text=True
)
print("✅ cuDNN 修复验证:", check.stdout.strip())
