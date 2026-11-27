"""Cell 5: Tailscale 内网固定域名转发 + H100/vLLM 真实显存与健康监控后台守护（Colab CLI 非阻塞版）。

CLI 改造亮点:
- 默认非阻塞模式 (KEEPALIVE_FOREGROUND=0)：拉起 Tailscale 与后台监控守护线程后立即退出，
  不阻塞 Jupyter kernel，可随时继续执行 Cell 6 / Cell 7 或其他 `colab exec` 命令。
- 环境变量优先读取 TS_AUTHKEY；若云盘恢复的 tailscaled.state 已处于 Running 状态则无需提供 AuthKey。
- AuthKey 通过 0600 临时文件 `--authkey=file:...` 传递并立即销毁，避免在进程列表中明文暴露。
- 新增 vLLM 存活检测：若 vLLM 进程崩溃或连续不可达，不再空烧 120 分钟 H100 算力。
- 实时写入状态至 `/tmp/colab_keepalive_status.json`，随时可查看运行状态。
"""

import os
import re
import json
import time
import signal
import shutil
import tempfile
import threading
import subprocess
import urllib.request
import urllib.error
from datetime import datetime

# ================= 配置区（均支持环境变量覆写）=================
HOSTNAME = os.environ.get("TS_HOSTNAME", "colab-llm").strip() or "colab-llm"
LOCAL_PORT = int(os.environ.get("GATEWAY_PORT") or globals().get("GATEWAY_PORT", 8000))
VLLM_INTERNAL_PORT = int(os.environ.get("VLLM_INTERNAL_PORT") or globals().get("VLLM_INTERNAL_PORT", 8001))
MAX_IDLE_MINUTES = float(os.environ.get("MAX_IDLE_MINUTES", "120"))
MAX_VLLM_DEAD_MINUTES = float(os.environ.get("MAX_VLLM_DEAD_MINUTES", "5"))
KEEPALIVE_FOREGROUND = os.environ.get("KEEPALIVE_FOREGROUND", "0").strip().lower() in ("1", "true", "yes")
KEEPALIVE_STATUS_FILE = os.environ.get("KEEPALIVE_STATUS_FILE", "/tmp/colab_keepalive_status.json")
TAILSCALE_SOCKET = "/var/run/tailscale/tailscaled.sock"
TAILSCALE_STATE_FILE = "/var/lib/tailscale/tailscaled.state"
# ===============================================================

_METRIC_LINE_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|NaN|Inf|-Inf)$"
)


def read_ts_authkey(required: bool = True) -> str | None:
    """安全读取 TS_AUTHKEY（环境变量优先以适配 Colab CLI，其次回退 Colab Secrets），禁止硬编码。"""
    key = os.environ.get("TS_AUTHKEY")
    if key and str(key).strip():
        return str(key).strip()

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


def query_nvidia_smi_vram_gb() -> tuple[float, float]:
    """通过 nvidia-smi 读取子进程 vLLM 真实的 H100 已用/总显存 (GB)。"""
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
            return 0.0, 0.0
        first_line = res.stdout.strip().splitlines()[0]
        used_mib, total_mib = [float(x.strip()) for x in first_line.split(",")[:2]]
        return used_mib / 1024.0, total_mib / 1024.0
    except Exception:
        return 0.0, 0.0


def parse_vllm_prometheus_metrics(metrics_text: str) -> dict[str, float]:
    """解析 vLLM Prometheus 文本指标，提取 running / waiting / kv_cache_usage_perc。"""
    running = 0.0
    waiting = 0.0
    kv_usage = None
    gpu_usage = None

    for raw_line in (metrics_text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _METRIC_LINE_RE.match(line)
        if not match:
            continue
        metric_name, val_str = match.group(1), match.group(2)
        try:
            val = float(val_str)
        except ValueError:
            continue
        if val != val:  # NaN check
            continue

        if metric_name == "vllm:num_requests_running":
            running += val
        elif metric_name == "vllm:num_requests_waiting":
            waiting += val
        elif metric_name == "vllm:kv_cache_usage_perc":
            kv_usage = val if kv_usage is None else max(kv_usage, val)
        elif metric_name == "vllm:gpu_cache_usage_perc":
            gpu_usage = val if gpu_usage is None else max(gpu_usage, val)

    effective_kv = kv_usage if kv_usage is not None else (gpu_usage if gpu_usage is not None else 0.0)
    return {
        "num_requests_running": running,
        "num_requests_waiting": waiting,
        "kv_cache_usage_perc": effective_kv,
    }


def compute_effective_active_requests(gateway_active: int, vllm_metrics: dict[str, float]) -> int:
    """综合网关活跃会话与 vLLM 引擎队列 (running + waiting) 计算真实活跃并发数。"""
    vllm_active = int(
        round(
            float(vllm_metrics.get("num_requests_running", 0.0))
            + float(vllm_metrics.get("num_requests_waiting", 0.0))
        )
    )
    return max(int(gateway_active), vllm_active)


def format_kv_cache_pct(kv_usage_val: float) -> float:
    """将 vLLM 0~1 或 0~100 的 KV Cache 指标统一格式化为百分比数值 (0~100)。"""
    val = float(kv_usage_val)
    if val <= 1.0:
        return val * 100.0
    return val


def is_vllm_healthy(vllm_port: int = VLLM_INTERNAL_PORT, vllm_proc=None) -> bool:
    """检查 vLLM 子进程存活状态及 /health 接口。"""
    if vllm_proc is not None and vllm_proc.poll() is not None:
        return False
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{vllm_port}/health", timeout=2.5) as resp:
            return resp.status == 200
    except Exception:
        return False


def fetch_vllm_metrics(gateway_port: int = LOCAL_PORT, vllm_port: int = VLLM_INTERNAL_PORT) -> dict[str, float]:
    """拉取 vLLM Prometheus 指标（优先经 Cell 4 /v1/metrics，其次直连 Cell 2 /metrics）。"""
    urls = [
        f"http://127.0.0.1:{gateway_port}/v1/metrics",
        f"http://127.0.0.1:{vllm_port}/metrics",
    ]
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=2.5) as resp:
                if resp.status == 200:
                    text = resp.read().decode("utf-8", errors="replace")
                    return parse_vllm_prometheus_metrics(text)
        except Exception:
            continue
    return {"num_requests_running": 0.0, "num_requests_waiting": 0.0, "kv_cache_usage_perc": 0.0}


def check_local_services(gateway_port: int = LOCAL_PORT, vllm_port: int = VLLM_INTERNAL_PORT) -> None:
    """启动 Tailscale 前校验 Cell 4 网关与 Cell 2 vLLM 健康状态。"""
    print(f"⏳ [0/3] 正在检查本地 Cell 4 网关 ({gateway_port}) 与 Cell 2 vLLM ({vllm_port}) 服务...", flush=True)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{gateway_port}/v1/models", timeout=5) as r:
            print(f"✅ Cell 4 网关运行正常 (HTTP {r.status})！", flush=True)
    except urllib.error.HTTPError as e:
        print(f"✅ Cell 4 网关已监听 (HTTP {e.code})！", flush=True)
    except Exception as exc:
        raise RuntimeError(f"❌ 127.0.0.1:{gateway_port} 网关尚未启动，请先运行 Cell 4！({exc})")

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{vllm_port}/health", timeout=5) as r:
            print(f"✅ Cell 2 vLLM 后端健康 (HTTP {r.status})！", flush=True)
    except Exception as exc:
        raise RuntimeError(f"❌ 127.0.0.1:{vllm_port}/health 尚未就绪，请先运行 Cell 2！({exc})")


def graceful_shutdown_services(vllm_proc=None, uvicorn_srv=None) -> None:
    """闲置超时释放 Colab GPU 前，先优雅终止 uvicorn_server 与 VLLM_PROCESS。"""
    if uvicorn_srv is not None:
        try:
            uvicorn_srv.should_exit = True
        except Exception as exc:
            print(f"ℹ️ 停止 uvicorn_server 提示: {exc}", flush=True)

    if vllm_proc is not None:
        try:
            if vllm_proc.poll() is None:
                print("🧹 正在优雅终止 VLLM_PROCESS 进程组 ...", flush=True)
                killed_pg = False
                if os.name == "posix" and hasattr(os, "killpg") and hasattr(os, "getpgid"):
                    try:
                        pgid = os.getpgid(vllm_proc.pid)
                        os.killpg(pgid, signal.SIGTERM)
                        try:
                            vllm_proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(pgid, signal.SIGKILL)
                            vllm_proc.wait(timeout=5)
                        killed_pg = True
                    except Exception:
                        killed_pg = False
                if not killed_pg:
                    vllm_proc.terminate()
                    try:
                        vllm_proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        vllm_proc.kill()
                        vllm_proc.wait(timeout=5)
        except Exception as exc:
            print(f"ℹ️ 停止 VLLM_PROCESS 提示: {exc}", flush=True)


def _try_unassign_runtime() -> None:
    """尝试释放 Colab 运行时（CLI 模式下兼容异常并提示手动 `colab stop`）。"""
    try:
        from google.colab import runtime

        runtime.unassign()
    except Exception as exc:
        print(f"⚠️ runtime.unassign() 调用异常（CLI 模式下请在本地运行 `colab stop` 释放实例）: {exc}", flush=True)


def _run_tailscale_up_with_key(socket_path: str, hostname: str, authkey: str) -> None:
    """通过 0600 临时文件传递 Tailscale AuthKey，避免在 ps 命令行参数中泄露密钥。"""
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
    """启动 tailscaled 用户态网络守护进程。"""
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
        err_out = ts_log.read()
        ts_log.close()
        raise RuntimeError(f"❌ tailscaled 启动失败:\n{err_out}")
    return proc, ts_log


def _write_status_snapshot(payload: dict) -> None:
    try:
        tmp_path = f"{KEEPALIVE_STATUS_FILE}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, KEEPALIVE_STATUS_FILE)
    except OSError:
        pass


def stop_keepalive_watchdog() -> None:
    """优雅停止上一次运行启动的后台守护线程（幂等）。"""
    ev = globals().get("_KEEPALIVE_STOP_EVENT")
    if isinstance(ev, threading.Event):
        ev.set()
    th = globals().get("_KEEPALIVE_THREAD")
    if isinstance(th, threading.Thread) and th.is_alive():
        th.join(timeout=3.0)


def run_tailscale_and_keepalive() -> dict:
    """配置 Tailscale 持久化节点与内网转发，并启动 H100/vLLM 监控保活守护。"""
    stop_keepalive_watchdog()
    check_local_services(LOCAL_PORT, VLLM_INTERNAL_PORT)

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

    served_model = os.environ.get("SERVED_MODEL_NAME") or globals().get("SERVED_MODEL_NAME", "Qwen/Qwen3.8-27B")
    api_key_hint = os.environ.get("COLAB_API_KEY") or "sk-colab"

    print("\n" + "=" * 72, flush=True)
    print("✅ Tailscale 内网免证书通道已就绪！（永久复用同一节点，免手动删机器）", flush=True)
    print(f"✨ 固定域名 Base URL: {http_url_domain}/v1", flush=True)
    print(f"✨ 当前节点内网 IP:   {ts_ip}", flush=True)
    print(f"🔍 浏览器自检地址:     {http_url_domain}/v1/models", flush=True)
    print(f"🔑 API Key: {api_key_hint} | 🤖 Model: {served_model}", flush=True)
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
        "model": served_model,
        "status_file": KEEPALIVE_STATUS_FILE,
    }

    def _watchdog_loop() -> None:
        nonlocal daemon_proc, ts_log
        vllm_dead_since = None
        try:
            while not stop_event.wait(timeout=60.0):
                now = time.time()
                tracker = globals().get("h100_gateway_tracker")
                gateway_active = tracker.active_sessions if tracker is not None else 0
                vllm_proc = globals().get("VLLM_PROCESS")
                vllm_ok = is_vllm_healthy(VLLM_INTERNAL_PORT, vllm_proc=vllm_proc)
                vllm_metrics = fetch_vllm_metrics(LOCAL_PORT, VLLM_INTERNAL_PORT) if vllm_ok else {
                    "num_requests_running": 0.0,
                    "num_requests_waiting": 0.0,
                    "kv_cache_usage_perc": 0.0,
                }
                active_cnt = compute_effective_active_requests(gateway_active, vllm_metrics)

                if active_cnt > 0:
                    globals()["LAST_API_REQUEST_TIME"] = now

                last_req = globals()["LAST_API_REQUEST_TIME"]
                idle_mins = (now - last_req) / 60.0

                if not vllm_ok:
                    if vllm_dead_since is None:
                        vllm_dead_since = now
                        print("\n⚠️ [Watchdog] 检测到 vLLM 后端不健康或已退出！", flush=True)
                else:
                    vllm_dead_since = None

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

                used_gb, total_gb = query_nvidia_smi_vram_gb()
                kv_pct = format_kv_cache_pct(vllm_metrics.get("kv_cache_usage_perc", 0.0))
                running_cnt = int(round(vllm_metrics.get("num_requests_running", 0.0)))
                waiting_cnt = int(round(vllm_metrics.get("num_requests_waiting", 0.0)))

                _write_status_snapshot(
                    {
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                        "base_url": info["base_url"],
                        "ip": ts_ip,
                        "vllm_healthy": vllm_ok,
                        "active_sessions": active_cnt,
                        "vllm_running": running_cnt,
                        "vllm_waiting": waiting_cnt,
                        "kv_cache_pct": round(kv_pct, 1),
                        "vram_used_gb": round(used_gb, 2),
                        "vram_total_gb": round(total_gb, 2),
                        "idle_minutes": round(idle_mins, 1),
                        "max_idle_minutes": MAX_IDLE_MINUTES,
                    }
                )

                if KEEPALIVE_FOREGROUND:
                    print(
                        f"💓 [{datetime.now().strftime('%H:%M:%S')}] H100 vLLM={'OK' if vllm_ok else 'DOWN'} | "
                        f"活跃并发: {active_cnt} (Run:{running_cnt}/Wait:{waiting_cnt}) | "
                        f"KV-Cache: {kv_pct:.1f}% | 显存: {used_gb:.1f}/{total_gb:.1f}GB | "
                        f"闲置: {idle_mins:.0f}/{MAX_IDLE_MINUTES:.0f} 分钟",
                        flush=True,
                    )

                dead_too_long = (
                    vllm_dead_since is not None
                    and (now - vllm_dead_since) / 60.0 >= MAX_VLLM_DEAD_MINUTES
                )
                idle_too_long = idle_mins >= MAX_IDLE_MINUTES and active_cnt == 0

                if dead_too_long or idle_too_long:
                    reason = (
                        f"vLLM 连续异常超过 {MAX_VLLM_DEAD_MINUTES:.0f} 分钟"
                        if dead_too_long
                        else f"已连续 {MAX_IDLE_MINUTES:.0f} 分钟无活跃请求"
                    )
                    print(f"\n🛑 {reason}，正在优雅停止服务并释放 H100 GPU...", flush=True)
                    graceful_shutdown_services(
                        vllm_proc=globals().get("VLLM_PROCESS"),
                        uvicorn_srv=globals().get("uvicorn_server"),
                    )
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
            "vllm_healthy": True,
            "active_sessions": 0,
            "idle_minutes": 0.0,
            "max_idle_minutes": MAX_IDLE_MINUTES,
        }
    )

    if KEEPALIVE_FOREGROUND:
        print("\n🛡️ H100 vLLM 前台保活守护已启动（按 Ctrl+C 停止）...", flush=True)
        try:
            _watchdog_loop()
        except KeyboardInterrupt:
            stop_event.set()
            print("\n⏹️ 已手动停止保活循环。", flush=True)
            if not drive_state_file:
                print("🧹 未挂载云盘，正在自动执行 logout 释放主机名...", flush=True)
                subprocess.run(["tailscale", f"--socket={TAILSCALE_SOCKET}", "logout"], check=False)
    else:
        bg_thread = threading.Thread(target=_watchdog_loop, daemon=True, name="h100_keepalive_watchdog")
        bg_thread.start()
        globals()["_KEEPALIVE_THREAD"] = bg_thread
        print(
            f"\n🛡️ H100 vLLM 后台保活与健康监控守护已启动（非阻塞模式，kernel 保持空闲可随时执行诊断命令）！\n"
            f"📊 实时状态文件: {KEEPALIVE_STATUS_FILE} | 闲置自动回收阈值: {MAX_IDLE_MINUTES:.0f} 分钟",
            flush=True,
        )

    return info


if __name__ == "__main__":
    run_tailscale_and_keepalive()
