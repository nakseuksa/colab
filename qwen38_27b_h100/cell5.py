"""Cell 5: Tailscale 内网固定域名转发 + H100/vLLM 真实显存与 KV-Cache 监控保活守护。"""

import os
import re
import json
import time
import signal
import shutil
import subprocess
import urllib.request
import urllib.error
from datetime import datetime

# ================= 配置区 =================
HOSTNAME = "colab-llm"
LOCAL_PORT = int(globals().get("GATEWAY_PORT", 8000))
VLLM_INTERNAL_PORT = int(globals().get("VLLM_INTERNAL_PORT", 8001))
MAX_IDLE_MINUTES = 120
TAILSCALE_SOCKET = "/var/run/tailscale/tailscaled.sock"
TAILSCALE_STATE_FILE = "/var/lib/tailscale/tailscaled.state"
# ==========================================

_METRIC_LINE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|NaN|Inf|-Inf)$")


def read_ts_authkey() -> str:
    """安全读取 TS_AUTHKEY（Colab Secrets 优先，其次环境变量），禁止硬编码。"""
    key = None
    try:
        from google.colab import userdata

        key = userdata.get("TS_AUTHKEY")
    except Exception:
        pass

    if not key:
        key = os.environ.get("TS_AUTHKEY")

    if not key or not str(key).strip():
        raise RuntimeError("❌ 未配置 TS_AUTHKEY！请在 Colab 左侧 Secrets 添加 TS_AUTHKEY 或设置环境变量。")
    return str(key).strip()


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
    print(f"⏳ [0/3] 正在检查本地 Cell 4 网关 ({gateway_port}) 与 Cell 2 vLLM ({vllm_port}) 服务...")
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{gateway_port}/v1/models", timeout=5) as r:
            print(f"✅ Cell 4 网关运行正常 (HTTP {r.status})！")
    except urllib.error.HTTPError as e:
        print(f"✅ Cell 4 网关已监听 (HTTP {e.code})！")
    except Exception as exc:
        raise RuntimeError(f"❌ 127.0.0.1:{gateway_port} 网关尚未启动，请先运行 Cell 4！({exc})")

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{vllm_port}/health", timeout=5) as r:
            print(f"✅ Cell 2 vLLM 后端健康 (HTTP {r.status})！")
    except Exception as exc:
        raise RuntimeError(f"❌ 127.0.0.1:{vllm_port}/health 尚未就绪，请先运行 Cell 2！({exc})")


def graceful_shutdown_services(vllm_proc=None, uvicorn_srv=None) -> None:
    """闲置超时释放 Colab GPU 前，先优雅终止 uvicorn_server 与 VLLM_PROCESS。"""
    if uvicorn_srv is not None:
        try:
            uvicorn_srv.should_exit = True
        except Exception as exc:
            print(f"ℹ️ 停止 uvicorn_server 提示: {exc}")

    if vllm_proc is not None:
        try:
            if vllm_proc.poll() is None:
                print("🧹 正在优雅终止 VLLM_PROCESS 进程组 ...")
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
            print(f"ℹ️ 停止 VLLM_PROCESS 提示: {exc}")


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
    )
    time.sleep(3)
    if proc.poll() is not None:
        ts_log.seek(0)
        err_out = ts_log.read()
        ts_log.close()
        raise RuntimeError(f"❌ tailscaled 启动失败:\n{err_out}")
    return proc, ts_log


def run_tailscale_and_keepalive() -> None:
    """配置 Tailscale 持久化节点与内网转发，并启动 H100/vLLM 监控保活守护。"""
    from google.colab import runtime

    ts_authkey = read_ts_authkey()
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
            print("✅ 已从云盘恢复固定节点身份（免重新注册，不会产生 -1 后缀）！")

    print("⏳ [1/3] 正在检查/安装 Tailscale...")
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

    subprocess.run(["pkill", "-f", "tailscaled"], capture_output=True, check=False)
    time.sleep(1)

    print("⏳ [2/3] 正在启动 tailscaled 守护进程...")
    daemon_proc, ts_log = start_tailscaled()

    print("⏳ [3/3] 正在连接 Tailscale 并开启内网 HTTP 80 端口转发...")
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
        subprocess.run(
            [
                "tailscale",
                f"--socket={TAILSCALE_SOCKET}",
                "up",
                f"--authkey={ts_authkey}",
                f"--hostname={HOSTNAME}",
                "--accept-dns=false",
                "--reset",
            ],
            check=True,
        )

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
    http_url_domain = f"http://{dns_name}"

    served_model = globals().get("SERVED_MODEL_NAME", "Qwen/Qwen3.8-27B")
    print("\n" + "=" * 72)
    print("✅ Tailscale 内网免证书通道已就绪！（永久复用同一节点，免手动删机器）")
    print(f"✨ 固定域名 Base URL: {http_url_domain}/v1")
    print(f"✨ 当前节点内网 IP:   {ts_ip}")
    print(f"🔍 浏览器自检地址:     {http_url_domain}/v1/models")
    print(f"🔑 API Key: sk-colab | 🤖 Model: {served_model}")
    print("=" * 72)

    if "LAST_API_REQUEST_TIME" not in globals():
        globals()["LAST_API_REQUEST_TIME"] = time.time()

    print("\n🛡️ H100 vLLM 保活守护已启动...")
    try:
        while True:
            time.sleep(60)
            now = time.time()

            tracker = globals().get("h100_gateway_tracker")
            gateway_active = tracker.active_sessions if tracker is not None else 0
            vllm_metrics = fetch_vllm_metrics(LOCAL_PORT, VLLM_INTERNAL_PORT)
            active_cnt = compute_effective_active_requests(gateway_active, vllm_metrics)

            if active_cnt > 0:
                globals()["LAST_API_REQUEST_TIME"] = now

            last_req = globals()["LAST_API_REQUEST_TIME"]
            idle_mins = (now - last_req) / 60.0

            if daemon_proc.poll() is not None:
                print("\n⚠️ 检测到 tailscaled 退出，正在自动重新拉起...")
                ts_log.close()
                daemon_proc, ts_log = start_tailscaled()
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

            used_gb, total_gb = query_nvidia_smi_vram_gb()
            kv_pct = format_kv_cache_pct(vllm_metrics.get("kv_cache_usage_perc", 0.0))
            running_cnt = int(round(vllm_metrics.get("num_requests_running", 0.0)))
            waiting_cnt = int(round(vllm_metrics.get("num_requests_waiting", 0.0)))

            print(
                f"\r💓 [{datetime.now().strftime('%H:%M:%S')}] H100 vLLM 运行中 | "
                f"活跃并发: {active_cnt} (Run:{running_cnt}/Wait:{waiting_cnt}) | "
                f"KV-Cache: {kv_pct:.1f}% | 显存: {used_gb:.1f}/{total_gb:.1f}GB | "
                f"闲置: {idle_mins:.0f}/{MAX_IDLE_MINUTES} 分钟",
                end="",
                flush=True,
            )

            if idle_mins >= MAX_IDLE_MINUTES and active_cnt == 0:
                print(f"\n🛑 已连续 {MAX_IDLE_MINUTES} 分钟无活跃请求，正在优雅停止服务并释放 H100 GPU...")
                graceful_shutdown_services(
                    vllm_proc=globals().get("VLLM_PROCESS"),
                    uvicorn_srv=globals().get("uvicorn_server"),
                )
                if not drive_state_file:
                    subprocess.run(["tailscale", f"--socket={TAILSCALE_SOCKET}", "logout"], check=False)
                runtime.unassign()
                break
    except KeyboardInterrupt:
        print("\n⏹️ 已手动停止保活循环。")
        if not drive_state_file:
            print("🧹 未挂载云盘，正在自动执行 logout 释放主机名...")
            subprocess.run(["tailscale", f"--socket={TAILSCALE_SOCKET}", "logout"], check=False)
    finally:
        ts_log.close()


if __name__ == "__main__":
    run_tailscale_and_keepalive()
