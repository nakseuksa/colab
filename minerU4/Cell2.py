"""MinerU 4.0 Step 2: 原子拷贝模型至本地高速盘，生成隔离配置与仅作用于子进程的 cuDNN Hook。

改进说明:
- 原子拷贝 + 周期进度打印：避免拷贝中断后留下残缺目录被误判已完成，同时防止 colab exec 30s 静默超时。
- 不修改 Google Drive 上的 config.yaml：在 /content/mineru_local_home 生成本地配置，避免污染多机共享云盘。
- 隔离式 cuDNN Hook：不再污染全局 site-packages/.pth 或硬编码 python3.13，仅通过 PYTHONPATH 注入 mineru-kit 子进程。
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

DRIVE_MINERU_HOME = Path(os.environ.get("DRIVE_MINERU_HOME", "/content/drive/MyDrive/mineru_home"))
DRIVE_MODELS_DIR = Path(os.environ.get("DRIVE_MODELS_DIR", str(DRIVE_MINERU_HOME / "models")))
LOCAL_MINERU_HOME = Path(os.environ.get("LOCAL_MINERU_HOME", "/content/mineru_local_home"))
LOCAL_MODELS_DIR = Path(os.environ.get("LOCAL_MODELS_DIR", "/content/mineru_models"))
HOOK_DIR = Path(os.environ.get("MINERU_HOOK_DIR", "/content/_mineru_site_hook"))
ENV_STATE_FILE = Path(os.environ.get("MINERU_ENV_STATE_FILE", "/content/mineru_env.json"))
DISABLE_CUDNN = os.environ.get("MINERU_DISABLE_CUDNN", "1").strip().lower() not in ("0", "false", "no")


def copy_models_atomically(src_dir: Path, dst_dir: Path) -> None:
    """将模型目录从 Google Drive 原子拷贝至本地高速盘，并定期输出进度防止 CLI 静默超时。"""
    marker = dst_dir / ".copy_complete"
    if dst_dir.exists() and marker.exists():
        print(f"✅ 本地高速盘已存在完整模型目录: {dst_dir}，跳过拷贝。", flush=True)
        return

    if not src_dir.exists():
        raise FileNotFoundError(
            f"❌ 找不到 Google Drive 模型目录: {src_dir}\n"
            "请确认已执行 `colab drivemount` 且 DRIVE_MODELS_DIR 路径正确。"
        )

    tmp_dir = dst_dir.parent / f".{dst_dir.name}.copying_tmp"
    if tmp_dir.exists():
        print(f"🧹 清理上次未完成的临时拷贝目录: {tmp_dir}", flush=True)
        shutil.rmtree(tmp_dir)
    if dst_dir.exists() and not marker.exists():
        print(f"⚠️ 检测到不完整的历史模型目录 {dst_dir}，重新拷贝...", flush=True)
        shutil.rmtree(dst_dir)

    print(f"⏳ 正在扫描模型文件: {src_dir} ...", flush=True)
    files_to_copy: list[tuple[Path, Path, int]] = []
    for root, _, files in os.walk(src_dir):
        root_p = Path(root)
        rel_dir = root_p.relative_to(src_dir)
        (tmp_dir / rel_dir).mkdir(parents=True, exist_ok=True)
        for fname in files:
            s_file = root_p / fname
            files_to_copy.append((s_file, tmp_dir / rel_dir / fname, s_file.stat().st_size))

    total_bytes = sum(item[2] for item in files_to_copy)
    print(
        f"⏳ 开始拷贝 {len(files_to_copy)} 个模型文件 ({total_bytes / (1024 ** 3):.2f} GB) -> {dst_dir} ...",
        flush=True,
    )

    copied_bytes = 0
    last_report = time.time()
    for idx, (s_file, d_file, size) in enumerate(files_to_copy, start=1):
        shutil.copy2(s_file, d_file)
        copied_bytes += size
        now = time.time()
        if now - last_report >= 10.0 or idx == len(files_to_copy):
            pct = (copied_bytes / max(total_bytes, 1)) * 100.0
            print(
                f"   📦 拷贝进度: {idx}/{len(files_to_copy)} 文件 "
                f"({copied_bytes / (1024 ** 3):.2f}/{total_bytes / (1024 ** 3):.2f} GB, {pct:.1f}%)",
                flush=True,
            )
            last_report = now

    (tmp_dir / ".copy_complete").write_text(f"copied_at={time.time():.0f}\n", encoding="utf-8")
    tmp_dir.rename(dst_dir)
    print(f"✅ 模型原子拷贝完成: {dst_dir}", flush=True)


def write_local_config(local_home: Path, local_models_dir: Path) -> Path:
    """在本地生成 MinerU config.yaml，避免覆写 Google Drive 共享配置。"""
    local_home.mkdir(parents=True, exist_ok=True)
    config_path = local_home / "config.yaml"
    config_path.write_text(
        f"""model:
  base_dir: {local_models_dir}
  source: auto
  small_backend: auto
  vlm:
    engine: auto
""",
        encoding="utf-8",
    )
    print(f"✅ 已生成本地隔离 MinerU 配置: {config_path} -> {local_models_dir}", flush=True)
    return config_path


def setup_isolated_cudnn_hook(hook_dir: Path) -> str:
    """生成仅通过 PYTHONPATH 生效的 cuDNN 禁用 Hook（不污染全局 Python 进程，兼容所有 Python 版本）。"""
    hook_dir.mkdir(parents=True, exist_ok=True)
    sitecustomize_py = hook_dir / "sitecustomize.py"
    hook_code = '''"""Isolated sitecustomize hook injected only into mineru-kit subprocesses."""
import sys

def _patch_cudnn():
    try:
        import torch
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        _cudnn_cls = torch.backends.cudnn.__class__
        _cudnn_cls.enabled = property(lambda self: False, lambda self, val: None)
        _cudnn_cls.benchmark = property(lambda self: False, lambda self, val: None)
    except Exception:
        pass

_patch_cudnn()
'''
    sitecustomize_py.write_text(hook_code, encoding="utf-8")

    verify_env = os.environ.copy()
    existing_pp = verify_env.get("PYTHONPATH", "")
    verify_env["PYTHONPATH"] = f"{hook_dir}:{existing_pp}" if existing_pp else str(hook_dir)

    check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; torch.backends.cudnn.enabled = True; "
            "assert torch.backends.cudnn.enabled is False, 'cuDNN hook failed'; "
            "print('子进程 cudnn.enabled =', torch.backends.cudnn.enabled)",
        ],
        env=verify_env,
        capture_output=True,
        text=True,
        check=False,
    )
    if check.returncode != 0:
        raise RuntimeError(
            f"❌ 隔离 cuDNN Hook 验证失败 (exit={check.returncode}):\n"
            f"stdout: {check.stdout.strip()}\nstderr: {check.stderr.strip()}"
        )
    print(f"✅ 隔离 cuDNN 修复验证通过: {check.stdout.strip()}", flush=True)
    return str(hook_dir)


def main() -> None:
    copy_models_atomically(DRIVE_MODELS_DIR, LOCAL_MODELS_DIR)
    write_local_config(LOCAL_MINERU_HOME, LOCAL_MODELS_DIR)

    os.environ["MINERU_HOME"] = str(LOCAL_MINERU_HOME)
    os.environ["MINERU_MODEL_SOURCE"] = "huggingface"

    hook_pythonpath = ""
    if DISABLE_CUDNN:
        hook_pythonpath = setup_isolated_cudnn_hook(HOOK_DIR)
        os.environ["MINERU_HOOK_PYTHONPATH"] = hook_pythonpath
    else:
        print("ℹ️ MINERU_DISABLE_CUDNN=0，跳过 cuDNN 禁用 Hook。", flush=True)

    state = {
        "MINERU_HOME": str(LOCAL_MINERU_HOME),
        "MINERU_MODEL_SOURCE": "huggingface",
        "MINERU_HOOK_PYTHONPATH": hook_pythonpath,
        "LOCAL_MODELS_DIR": str(LOCAL_MODELS_DIR),
    }
    ENV_STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"💾 已持久化本地运行环境契约: {ENV_STATE_FILE}", flush=True)


if __name__ == "__main__":
    main()
