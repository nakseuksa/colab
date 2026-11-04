import os
import time
import torch

# ==========================================================
# 0. 显存防碎片与 Blackwell (sm_120) 底层加速配置
# ==========================================================
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
try:
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
except Exception as e:
    print(f"ℹ️ 显存分配器动态设置提示: {e}")

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_cudnn_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

# ==========================================================
# 1. 安全读取 Hugging Face Token（切勿在代码中明文硬编码 Token！）
# ==========================================================
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import login
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = AutoModelForCausalLM

HF_TOKEN = None
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get("HF_TOKEN")
except Exception:
    pass

if not HF_TOKEN:
    MANUAL_TOKEN = ""  # ⚠️ 请优先使用 Colab 左侧 Secrets 配置 HF_TOKEN，切勿泄露明文 Token
    if MANUAL_TOKEN.startswith("hf_"):
        HF_TOKEN = MANUAL_TOKEN.strip()

if HF_TOKEN:
    login(token=HF_TOKEN, add_to_git_credential=True)
    print("✅ Hugging Face Token 认证成功！")
else:
    print("ℹ️ 未配置 HF_TOKEN，将以匿名方式高速下载公开仓库权重。")

# ==========================================================
# 2. 加载分词器、多模态处理器与 Omni-Edu-27B 模型
# ==========================================================
model_id = "lhpku20010120/Omni-Edu-27B"
cache_dir = None

print(f"🚀 开始加载 {model_id} (原生 BF16 + Gated DeltaNet FLA + SDPA)...")
t_load_start = time.time()

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    token=HF_TOKEN,
    cache_dir=cache_dir,
    trust_remote_code=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

try:
    processor = AutoProcessor.from_pretrained(
        model_id,
        token=HF_TOKEN,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
except Exception:
    processor = tokenizer

load_kwargs = dict(
    token=HF_TOKEN,
    cache_dir=cache_dir,
    device_map="cuda:0",
    attn_implementation="sdpa",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

try:
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, **load_kwargs
    ).eval()
except TypeError:
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, **load_kwargs
    ).eval()
except Exception:
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, **load_kwargs
    ).eval()

print(f"⏱️ 权重加载耗时: {time.time() - t_load_start:.1f}s")

# ==========================================================
# 3. 关键优化：执行“长序列 Prefill (Chunk 算子) + 多步 Decode (Recurrent 算子)”双阶段预热
#    同时覆盖 FLA 线性注意力的分块预填充核与单步递归核！
# ==========================================================
print("🔥 正在预热 Blackwell (sm_120) Triton Chunk-Prefill 与 Recurrent-Decode 双阶段算子...")
t_warmup = time.time()
with torch.inference_mode():
    warmup_text = "Warmup sequence for Blackwell Triton chunked linear attention kernels. " * 12
    warmup_inputs = tokenizer([warmup_text], return_tensors="pt").to(model.device)
    _ = model.generate(**warmup_inputs, max_new_tokens=3, use_cache=True, do_sample=False)
    del warmup_inputs, _
torch.cuda.synchronize()
torch.cuda.empty_cache()
print(f"✅ 双阶段算子预热完成 (耗时 {time.time() - t_warmup:.1f}s)！如已挂载云盘，编译结果已持久化。")

# 4. 验证显存占用
used_vram = torch.cuda.memory_allocated(0) / (1024**3)
reserved_vram = torch.cuda.memory_reserved(0) / (1024**3)
total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print(
    f"📊 当前显存占用: 已分配 {used_vram:.2f} GB (已预留 {reserved_vram:.2f} GB) / 总计 {total_vram:.2f} GB "
    f"(剩余 {total_vram - used_vram:.2f} GB 可用于并发会话缓存)"
)
