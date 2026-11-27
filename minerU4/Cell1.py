"""MinerU 4.0 Step 1: 校验 Google Drive 挂载并安装 uv 与 MinerU 4.0 [full]。

Colab CLI 用法:
    colab new -s mineru --gpu L4
    colab drivemount -s mineru
    colab exec -s mineru --timeout 0 -f minerU4/Cell1.py
"""

import os
import sys
import subprocess

DRIVE_MOUNT_POINT = os.environ.get("DRIVE_MOUNT_POINT", "/content/drive")
MINERU_SPEC = os.environ.get("MINERU_SPEC", "mineru[full]>=4.0,<5")
ALLOW_INTERACTIVE_MOUNT = os.environ.get("COLAB_AUTO_MOUNT_DRIVE", "0").strip().lower() in ("1", "true", "yes")


def ensure_drive_mounted(mount_point: str = DRIVE_MOUNT_POINT) -> bool:
    """检查 Google Drive 挂载状态。CLI 模式下避免触发阻塞式 OAuth 侧信道。"""
    if os.path.ismount(mount_point):
        print(f"✅ 检测到 Google Drive 已挂载: {mount_point}")
        return True

    if ALLOW_INTERACTIVE_MOUNT:
        try:
            from google.colab import drive

            drive.mount(mount_point)
            print(f"✅ Google Drive 交互式挂载成功: {mount_point}")
            return True
        except Exception as exc:
            raise RuntimeError(f"❌ Google Drive 挂载失败: {exc}") from exc

    raise RuntimeError(
        f"❌ 未检测到已挂载的 Google Drive ({mount_point})。\n"
        "👉 在 Colab CLI 中请先执行: colab drivemount -s <session_name>\n"
        "   （如在网页端交互运行，可设置环境变量 COLAB_AUTO_MOUNT_DRIVE=1 后重试）"
    )


def run_cmd(cmd: list[str], desc: str) -> None:
    """以纯 Python subprocess 执行命令并通过 sys.stdout 实时流式输出（确保 Jupyter/Colab CLI 可见），非零退出码立即中止。"""
    print(f"📦 {desc}: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def main() -> None:
    ensure_drive_mounted()
    run_cmd([sys.executable, "-m", "pip", "install", "-U", "uv"], "安装/升级 uv")
    run_cmd(["uv", "pip", "install", "--system", "-U", MINERU_SPEC], f"安装 {MINERU_SPEC}")
    print("✅ MinerU 4.0 环境依赖安装完成！", flush=True)


if __name__ == "__main__":
    main()
