import os
import subprocess
from pathlib import Path

# ==========================================================
# 参数设置（每次换新 PDF 只需修改这里）
# ==========================================================
PDF_PATH = "/content/drive/MyDrive/预代数_5_8.pdf"              # 待解析的 PDF 路径
OUTPUT_DIR = Path("/content/drive/MyDrive/Pre-Algebra_zips") # ZIP 结果保存目录
TOTAL_PAGES = 1280                                              # PDF 总页数
BATCH_SIZE = 192                                               # 每批处理页数
TIER = "advanced"                                              # 复杂公式推荐 standard（极端难度可用 advanced）
OCR_MODE = "ocr"                                               # 强制开启 OCR 模式

assert os.path.exists(PDF_PATH), f"❌ 找不到 PDF 文件: {PDF_PATH}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# 分批循环提取
# ==========================================================
for start_page in range(1, TOTAL_PAGES + 1, BATCH_SIZE):
    end_page = min(start_page + BATCH_SIZE - 1, TOTAL_PAGES)
    page_range = f"{start_page}-{end_page}"

    zip_out_path = OUTPUT_DIR / f"pages_{start_page:03d}_{end_page:03d}.zip"

    # 断点续传：已存在且非空则跳过
    if zip_out_path.exists() and zip_out_path.stat().st_size > 0:
        print(f"✅ 第 {page_range} 页已存在，跳过: {zip_out_path.name}")
        continue

    print(f"\n🚀 正在解析第 {page_range} 页 (tier={TIER}, ocr-mode={OCR_MODE}, format=zip)...")

    cmd = [
        "mineru-kit", "parse", PDF_PATH,
        "-o", str(zip_out_path),
        "--tier", TIER,
        "--ocr-mode", OCR_MODE,
        "--format", "zip",
        "--pages", page_range
    ]

    # 实时将子进程日志打印到 Colab 页面
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=os.environ.copy()
    )
    for line in process.stdout:
        print(line, end="")
    process.wait()

    if process.returncode != 0:
        print(f"\n❌ 第 {page_range} 页解析失败（返回码 {process.returncode}）")
        break
    else:
        print(f"\n🎉 第 {page_range} 页导出完成 -> {zip_out_path}")
