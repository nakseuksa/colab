# Cell 5 (Tailscale 内网 HTTP 永久免维护版): 固定域名 + 固定 IP + 自动复用节点身份(免删后台机器)
import os
import json
import time
import shutil
import subprocess
import urllib.request
import torch
from datetime import datetime
from google.colab import runtime, userdata

# ================= 配置区 =================
TS_AUTHKEY = userdata.get("TS_AUTHKEY").strip()
HOSTNAME = "colab-llm"
LOCAL_PORT = 8000
# ==========================================

# 0. 先检查本地 8000 端口大模型服务是否已就绪
print(f"⏳ [0/3] 正在检查 Colab 本地 {LOCAL_PORT} 端口大模型服务...")
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{LOCAL_PORT}/v1/models", timeout=5) as r:
        print(f"✅ 本地大模型服务运行正常 (HTTP {r.status})！")
except urllib.error.HTTPError as e:
    print(f"✅ 本地大模型服务已监听 (HTTP {e.code})！")
except Exception as e:
    raise RuntimeError(f"❌ 127.0.0.1:{LOCAL_PORT} 尚未启动，请先运行前面的模型加载单元格！({e})")

# 1. 云盘持久化节点身份（让每次重启都复用同一台机器，永远不用去后台删节点！）
local_state_file = "/var/lib/tailscale/tailscaled.state"
os.makedirs("/var/lib/tailscale", exist_ok=True)
os.makedirs("/var/run/tailscale", exist_ok=True)

drive_state_file = None
if os.path.ismount("/content/drive"):
    drive_dir = f"/content/drive/MyDrive/colab_wheels/tailscale_node_{HOSTNAME}"
    os.makedirs(drive_dir, exist_ok=True)
    drive_state_file = f"{drive_dir}/tailscaled.state"
    if os.path.exists(drive_state_file) and not os.path.exists(local_state_file):
        shutil.copy2(drive_state_file, local_state_file)
        os.chmod(local_state_file, 0o600)
        print("✅ 已从云盘恢复固定节点身份（免重新注册，不会产生 -1 后缀）！")

print("⏳ [1/3] 正在检查/安装 Tailscale...")
if not os.path.exists("/usr/sbin/tailscaled"):
    !curl -fsSL https://tailscale.com/install.sh | sh > /dev/null 2>&1
!pkill -f tailscaled || true
time.sleep(1)

print("⏳ [2/3] 正在启动 tailscaled 守护进程...")
ts_log = open("/tmp/tailscaled.log", "w+")

def start_tailscaled():
    proc = subprocess.Popen(
        [
            "/usr/sbin/tailscaled",
            "--tun=userspace-networking",
            f"--state={local_state_file}",
            "--socket=/var/run/tailscale/tailscaled.sock",
        ],
        stdout=ts_log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    time.sleep(3)
    if proc.poll() is not None:
        ts_log.seek(0)
        raise RuntimeError(f"❌ tailscaled 启动失败:\n{ts_log.read()}")
    return proc

daemon_proc = start_tailscaled()

print("⏳ [3/3] 正在连接 Tailscale 并开启内网 HTTP 80 端口转发...")
# 判断是否已恢复有效身份，若已恢复则直接上线，不重复消耗 AuthKey
status_check = subprocess.run(
    ["tailscale", "--socket=/var/run/tailscale/tailscaled.sock", "status", "--json"],
    capture_output=True,
    text=True,
)
backend_state = ""
if status_check.returncode == 0 and status_check.stdout.strip():
    try:
        backend_state = json.loads(status_check.stdout).get("BackendState", "")
    except Exception:
        pass

if backend_state == "Running":
    !tailscale --socket=/var/run/tailscale/tailscaled.sock up --hostname={HOSTNAME} --accept-dns=false
else:
    !tailscale --socket=/var/run/tailscale/tailscaled.sock up --authkey={TS_AUTHKEY} --hostname={HOSTNAME} --accept-dns=false --reset

# 备份节点身份到云盘
if drive_state_file and os.path.exists(local_state_file):
    shutil.copy2(local_state_file, drive_state_file)

# 开启内网 HTTP 80 -> 127.0.0.1:8000 转发
!tailscale --socket=/var/run/tailscale/tailscaled.sock serve reset > /dev/null 2>&1 || true
!tailscale --socket=/var/run/tailscale/tailscaled.sock serve --bg --http=80 http://127.0.0.1:{LOCAL_PORT}

status_raw = subprocess.check_output(
    ["tailscale", "--socket=/var/run/tailscale/tailscaled.sock", "status", "--json"],
    text=True,
)
status_json = json.loads(status_raw)
dns_name = status_json.get("Self", {}).get("DNSName", "").rstrip(".")
ts_ip = status_json.get("Self", {}).get("TailscaleIPs", [""])[0]

http_url_domain = f"http://{dns_name}"

print("\n" + "=" * 72)
print("✅ Tailscale 内网免证书通道已就绪！（永久复用同一节点，免手动删机器）")
print(f"✨ 固定域名 Base URL: {http_url_domain}/v1")
print(f"✨ 当前节点内网 IP:   {ts_ip}")
print(f"🔍 浏览器自检地址:     {http_url_domain}/v1/models")
print(f"🔑 API Key: sk-colab | 🤖 Model: Qwen/Qwen3.8-27B")
print("=" * 72)

if "LAST_API_REQUEST_TIME" not in globals():
    globals()["LAST_API_REQUEST_TIME"] = time.time()

MAX_IDLE_MINUTES = 120

print("\n🛡️ 保活守护已启动...")
try:
    while True:
        time.sleep(60)
        now = time.time()
        active_cnt = g4_scheduler.active_sessions if "g4_scheduler" in globals() else 0
        if active_cnt > 0:
            globals()["LAST_API_REQUEST_TIME"] = now

        last_req = globals()["LAST_API_REQUEST_TIME"]
        idle_mins = (now - last_req) / 60.0

        if daemon_proc.poll() is not None:
            print("\n⚠️ 检测到 tailscaled 退出，正在自动重新拉起...")
            daemon_proc = start_tailscaled()
            subprocess.run(
                ["tailscale", "--socket=/var/run/tailscale/tailscaled.sock", "up", f"--hostname={HOSTNAME}", "--accept-dns=false"],
                check=False,
            )
            subprocess.run(
                ["tailscale", "--socket=/var/run/tailscale/tailscaled.sock", "serve", "--bg", "--http=80", f"http://127.0.0.1:{LOCAL_PORT}"],
                check=False,
            )

        vram_gb = torch.cuda.memory_allocated(0) / (1024**3) if torch.cuda.is_available() else 0.0
        print(
            f"\r💓 [{datetime.now().strftime('%H:%M:%S')}] G4 运行中 | 活跃并发: {active_cnt} | "
            f"显存: {vram_gb:.1f}GB | 闲置计时: {idle_mins:.0f}/{MAX_IDLE_MINUTES} 分钟",
            end="",
            flush=True,
        )

        if idle_mins >= MAX_IDLE_MINUTES and active_cnt == 0:
            print(f"\n🛑 已连续 {MAX_IDLE_MINUTES} 分钟无活跃请求，正在自动释放 G4 GPU...")
            # 如果没有挂载云盘才执行 logout；若挂载了云盘则保留节点身份供下次直接复用
            if not drive_state_file:
                subprocess.run(["tailscale", "--socket=/var/run/tailscale/tailscaled.sock", "logout"], check=False)
            runtime.unassign()
            break
except KeyboardInterrupt:
    print("\n⏹️ 已手动停止保活循环。")
    if not drive_state_file:
        print("🧹 未挂载云盘，正在自动执行 logout 释放主机名...")
        subprocess.run(["tailscale", "--socket=/var/run/tailscale/tailscaled.sock", "logout"], check=False)
