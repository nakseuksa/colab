"""Qwen3.8-27B Cell 5: Tailscale 内网固定域名转发 + 后台保活监控守护（Colab CLI 非阻塞版）。"""

import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime

import torch

# ================= 配置区（支持环境变量覆写）=================
HOSTNAME = os.environ.get("TS_HOSTNAME", "colab-llm").strip() or "colab-llm"
LOCAL_PORT = int(os.environ.get("GATEWAY_PORT") or globals().get("GATEWAY_PORT", 8000))
MAX_IDLE_MINUTES = float(os.environ.get("MAX_IDLE_MINUTES", "120"))
KEEPALIVE_FOREGROUND = os.environ.get("KEEPALIVE_FOREGROUND", "0").strip().lower() in ("1", "true", "yes")
KEEPALIVE_STATUS_FILE = os.environ.get("KEEPALIVE_STATUS_FILE", "/tmp/colab_keepalive_status.json")
TAILSCALE_SOCKET = "/var/run/tailscale/tailscaled.sock"
TAILSCALE_STATE_FILE = "/var/lib/tailscale/tailscaled.state"
DEFAULT_MODEL_NAME = os.environ.get("MODEL_ID") or globals().get("DEFAULT_MODEL_ID", "Qwen/Qwen3.8-27B")
# ===============================================================


def read_ts_authkey(required: bool = True) -> str | None:
    """安全读取 TS_AUTHKEY（环境变量优先适配 Colab CLI，其次回退 Colab Secrets）。"""
    key = os.environ.get("TS_AUTHKEY")
    if key and key.strip():
        return key.strip()
    try:
        from google.colab import userdata

        key = userdata.get("TS_AUTHKEY")
        if key and str(key).strip():
            return str(key).strip()
    except Exception:
        pass
    if required:
        raise RuntimeError(
            "❌ 未配置 TS_AUTHKEY！\n"
            "👉 在 Colab CLI 中请通过 `--env TS_AUTHKEY=\"$TS_AUTHKEY\"` 传入，"
            "或在网页端左侧 Secrets 添加 TS_AUTHKEY。"
        )
    return None


def _run_tailscale_up_with_key(socket_path: str, hostname: str, authkey: str) -> None:
    fd, tmp_key_path = tempfile.mkstemp(prefix="ts_key_", dir="/tmp")
    try:
        os.chmod(tmp_key_path, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(authkey)
        subprocess.run(
            [
                "tailscale",
                f"--socket={socket_path}",
                "up",
                f"--authkey=file:{tmp_key_path}",
                f"--hostname={hostname}",
                "--accept-dns=false",
                "--reset",
            ],
            check=True,
        )
    finally:
        try:
            os.unlink(tmp_key_path)
        except OSError:
            pass


def start_tailscaled(state_file: str = TAILSCALE_STATE_FILE, socket_path: str = TAILSCALE_SOCKET):
    ts_log = open("/tmp/tailscaled.log", "w+", encoding="utf-8")
    proc = subprocess.Popen(
        [
            "/usr/sbin/tailscaled",
            "--tun=userspace-networking",
            f"--state={state_file}",
            f"--socket={socket_path}",
        ],
        stdout=ts_log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=(os.name == "posix"),
    )
    time.sleep(3)
    if proc.poll() is not None:
        ts_log.seek(0)
        err = ts_log.read()
        ts_log.close()
        raise RuntimeError(f"❌ tailscaled 启动失败:\n{err}")
    return proc, ts_log


def _try_unassign_runtime() -> None:
    try:
        from google.colab import runtime

        runtime.unassign()
    except Exception as exc:
        print(f"⚠️ runtime.unassign() 调用异常（CLI 模式下请在本地执行 `colab stop` 释放实例）: {exc}", flush=True)


def _write_status_snapshot(payload: dict) -> None:
    try:
        tmp = f"{KEEPALIVE_STATUS_FILE}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp, KEEPALIVE_STATUS_FILE)
    except OSError:
        pass


def stop_keepalive_watchdog() -> None:
    ev = globals().get("_KEEPALIVE_STOP_EVENT")
    if isinstance(ev, threading.Event):
        ev.set()
    th = globals().get("_KEEPALIVE_THREAD")
    if isinstance(th, threading.Thread) and th.is_alive():
        th.join(timeout=3.0)


def main() -> dict:
    stop_keepalive_watchdog()

    # 0. 检查本地大模型服务
    print(f"⏳ [0/3] 正在检查 Colab 本地 {LOCAL_PORT} 端口大模型服务...", flush=True)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{LOCAL_PORT}/v1/models", timeout=5) as r:
            print(f"✅ 本地大模型服务运行正常 (HTTP {r.status})！", flush=True)
    except urllib.error.HTTPError as e:
        print(f"✅ 本地大模型服务已监听 (HTTP {e.code})！", flush=True)
    except Exception as exc:
        raise RuntimeError(f"❌ 127.0.0.1:{LOCAL_PORT} 尚未启动，请先运行 cell4.py！({exc})")

    # 1. 云盘持久化节点身份
    os.makedirs("/var/lib/tailscale", exist_ok=True)
    os.makedirs("/var/run/tailscale", exist_ok=True)

    drive_state_file = None
    if os.path.ismount("/content/drive"):
        drive_dir = f"/content/drive/MyDrive/colab_wheels/tailscale_node_{HOSTNAME}"
        os.makedirs(drive_dir, exist_ok=True)
        drive_state_file = f"{drive_dir}/tailscaled.state"
        if os.path.exists(drive_state_file) and not os.path.exists(TAILSCALE_STATE_FILE):
            shutil.copy2(drive_state_file, TAILSCALE_STATE_FILE)
            os.chmod(TAILSCALE_STATE_FILE, 0o600)
            print("✅ 已从云盘恢复固定节点身份（免重新注册，不会产生 -1 后缀）！", flush=True)

    print("⏳ [1/3] 正在检查/安装 Tailscale...", flush=True)
    if not os.path.exists("/usr/sbin/tailscaled"):
        install_res = subprocess.run(
            ["sh", "-c", "curl -fsSL https://tailscale.com/install.sh | sh"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if install_res.returncode != 0:
            raise RuntimeError(f"❌ Tailscale 安装失败: {install_res.stderr}")

    subprocess.run(["pkill", "-x", "tailscaled"], capture_output=True, check=False)
    time.sleep(1)

    print("⏳ [2/3] 正在启动 tailscaled 守护进程...", flush=True)
    daemon_proc, ts_log = start_tailscaled()

    print("⏳ [3/3] 正在连接 Tailscale 并开启内网 HTTP 80 端口转发...", flush=True)
    status_check = subprocess.run(
        ["tailscale", f"--socket={TAILSCALE_SOCKET}", "status", "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    backend_state = ""
    if status_check.returncode == 0 and status_check.stdout.strip():
        try:
            backend_state = json.loads(status_check.stdout).get("BackendState", "")
        except Exception:
            pass

    if backend_state == "Running":
        subprocess.run(
            ["tailscale", f"--socket={TAILSCALE_SOCKET}", "up", f"--hostname={HOSTNAME}", "--accept-dns=false"],
            check=True,
        )
    else:
        ts_authkey = read_ts_authkey(required=True)
        assert ts_authkey is not None
        _run_tailscale_up_with_key(TAILSCALE_SOCKET, HOSTNAME, ts_authkey)

    if drive_state_file and os.path.exists(TAILSCALE_STATE_FILE):
        shutil.copy2(TAILSCALE_STATE_FILE, drive_state_file)

    subprocess.run(
        ["tailscale", f"--socket={TAILSCALE_SOCKET}", "serve", "reset"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    subprocess.run(
        [
            "tailscale",
            f"--socket={TAILSCALE_SOCKET}",
            "serve",
            "--bg",
            "--http=80",
            f"http://127.0.0.1:{LOCAL_PORT}",
        ],
        check=True,
    )

    status_raw = subprocess.check_output(
        ["tailscale", f"--socket={TAILSCALE_SOCKET}", "status", "--json"],
        text=True,
    )
    status_json = json.loads(status_raw)
    dns_name = status_json.get("Self", {}).get("DNSName", "").rstrip(".")
    ts_ip = (status_json.get("Self", {}).get("TailscaleIPs") or [""])[0]
    http_url_domain = f"http://{dns_name}" if dns_name else f"http://{ts_ip}"
    api_key_hint = os.environ.get("COLAB_API_KEY") or "sk-colab"

    print("\n" + "=" * 72, flush=True)
    print("✅ Tailscale 内网免证书通道已就绪！（永久复用同一节点，免手动删机器）", flush=True)
    print(f"✨ 固定域名 Base URL: {http_url_domain}/v1", flush=True)
    print(f"✨ 当前节点内网 IP:   {ts_ip}", flush=True)
    print(f"🔍 浏览器自检地址:     {http_url_domain}/v1/models", flush=True)
    print(f"🔑 API Key: {api_key_hint} | 🤖 Model: {DEFAULT_MODEL_NAME}", flush=True)
    print("=" * 72, flush=True)

    if "LAST_API_REQUEST_TIME" not in globals():
        globals()["LAST_API_REQUEST_TIME"] = time.time()

    stop_event = threading.Event()
    globals()["_KEEPALIVE_STOP_EVENT"] = stop_event
    globals()["TAILSCALE_DAEMON_PROC"] = daemon_proc

    info = {
        "base_url": f"{http_url_domain}/v1",
        "ip": ts_ip,
        "hostname": HOSTNAME,
        "model": DEFAULT_MODEL_NAME,
        "status_file": KEEPALIVE_STATUS_FILE,
    }

    def _watchdog_loop() -> None:
        nonlocal daemon_proc, ts_log
        try:
            while not stop_event.wait(timeout=60.0):
                now = time.time()
                sched = globals().get("g4_scheduler")
                active_cnt = sched.active_sessions if sched is not None else 0
                if active_cnt > 0:
                    globals()["LAST_API_REQUEST_TIME"] = now

                last_req = globals()["LAST_API_REQUEST_TIME"]
                idle_mins = (now - last_req) / 60.0

                if daemon_proc.poll() is not None:
                    print("\n⚠️ [Watchdog] 检测到 tailscaled 退出，正在自动重新拉起...", flush=True)
                    ts_log.close()
                    try:
                        daemon_proc, ts_log = start_tailscaled()
                        globals()["TAILSCALE_DAEMON_PROC"] = daemon_proc
                        subprocess.run(
                            ["tailscale", f"--socket={TAILSCALE_SOCKET}", "up", f"--hostname={HOSTNAME}", "--accept-dns=false"],
                            check=False,
                        )
                        subprocess.run(
                            [
                                "tailscale",
                                f"--socket={TAILSCALE_SOCKET}",
                                "serve",
                                "--bg",
                                "--http=80",
                                f"http://127.0.0.1:{LOCAL_PORT}",
                            ],
                            check=False,
                        )
                    except Exception as exc:
                        print(f"\n❌ [Watchdog] tailscaled 重拉失败: {exc}", flush=True)

                vram_gb = torch.cuda.memory_allocated(0) / (1024 ** 3) if torch.cuda.is_available() else 0.0
                _write_status_snapshot(
                    {
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                        "base_url": info["base_url"],
                        "ip": ts_ip,
                        "model": DEFAULT_MODEL_NAME,
                        "active_sessions": active_cnt,
                        "vram_allocated_gb": round(vram_gb, 2),
                        "idle_minutes": round(idle_mins, 1),
                        "max_idle_minutes": MAX_IDLE_MINUTES,
                    }
                )

                if KEEPALIVE_FOREGROUND:
                    print(
                        f"💓 [{datetime.now().strftime('%H:%M:%S')}] G4 运行中 | 活跃并发: {active_cnt} | "
                        f"显存: {vram_gb:.1f}GB | 闲置计时: {idle_mins:.0f}/{MAX_IDLE_MINUTES:.0f} 分钟",
                        flush=True,
                    )

                if idle_mins >= MAX_IDLE_MINUTES and active_cnt == 0:
                    print(f"\n🛑 已连续 {MAX_IDLE_MINUTES:.0f} 分钟无活跃请求，正在自动释放 G4 GPU...", flush=True)
                    srv = globals().get("uvicorn_server")
                    if srv is not None:
                        srv.should_exit = True
                    if not drive_state_file:
                        subprocess.run(["tailscale", f"--socket={TAILSCALE_SOCKET}", "logout"], check=False)
                    _try_unassign_runtime()
                    break
        finally:
            try:
                ts_log.close()
            except Exception:
                pass

    _write_status_snapshot(
        {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "base_url": info["base_url"],
            "ip": ts_ip,
            "model": DEFAULT_MODEL_NAME,
            "active_sessions": 0,
            "idle_minutes": 0.0,
            "max_idle_minutes": MAX_IDLE_MINUTES,
        }
    )

    if KEEPALIVE_FOREGROUND:
        print("\n🛡️ 前台保活守护已启动（按 Ctrl+C 停止）...", flush=True)
        try:
            _watchdog_loop()
        except KeyboardInterrupt:
            stop_event.set()
            print("\n⏹️ 已手动停止保活循环。", flush=True)
            if not drive_state_file:
                subprocess.run(["tailscale", f"--socket={TAILSCALE_SOCKET}", "logout"], check=False)
    else:
        bg_thread = threading.Thread(target=_watchdog_loop, daemon=True, name="g4_keepalive_watchdog")
        bg_thread.start()
        globals()["_KEEPALIVE_THREAD"] = bg_thread
        print(
            f"\n🛡️ 后台保活守护已启动（非阻塞模式，kernel 保持空闲可随时执行诊断命令）！\n"
            f"📊 实时状态文件: {KEEPALIVE_STATUS_FILE} | 闲置自动回收阈值: {MAX_IDLE_MINUTES:.0f} 分钟",
            flush=True,
        )

    return info


if __name__ == "__main__":
    main()
