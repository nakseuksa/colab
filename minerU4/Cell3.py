"""MinerU 4.0 Step 3: 分批解析 PDF 为 ZIP（CLI 参数化、自动探测页数、动态补零与原子断点续传）。

环境变量配置:
    PDF_PATH     : 待解析 PDF 路径
    OUTPUT_DIR   : ZIP 输出目录
    TOTAL_PAGES  : PDF 总页数（留空或 0 则自动检测）
    BATCH_SIZE   : 每批页数（默认 192）
    TIER         : standard | advanced（默认 advanced）
    OCR_MODE     : ocr | auto | txt（默认 ocr）
"""

import json
import os
import shutil
import subprocess
import threading
import time
import zipfile
from pathlib import Path

PDF_PATH = os.environ.get("PDF_PATH", "/content/drive/MyDrive/预代数_5_8.pdf")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/content/drive/MyDrive/Pre-Algebra_zips"))
TOTAL_PAGES_RAW = os.environ.get("TOTAL_PAGES", "0").strip()
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "192"))
TIER = os.environ.get("TIER", "advanced")
OCR_MODE = os.environ.get("OCR_MODE", "ocr")
ENV_STATE_FILE = Path(os.environ.get("MINERU_ENV_STATE_FILE", "/content/mineru_env.json"))
HEARTBEAT_SECONDS = float(os.environ.get("MINERU_HEARTBEAT_SECONDS", "20"))


def detect_pdf_total_pages(pdf_path: str) -> int:
    """自动读取 PDF 总页数，依次尝试 pypdfium2 / pypdf / fitz。"""
    try:
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(pdf_path)
        count = len(doc)
        doc.close()
        return count
    except Exception:
        pass

    try:
        from pypdf import PdfReader

        return len(PdfReader(pdf_path).pages)
    except Exception:
        pass

    try:
        import fitz

        with fitz.open(pdf_path) as doc:
            return len(doc)
    except Exception:
        pass

    raise RuntimeError(
        "❌ 无法自动识别 PDF 总页数（未安装 pypdfium2 / pypdf / PyMuPDF）。\n"
        "👉 请通过环境变量显式指定: --env TOTAL_PAGES=<页数>"
    )


def load_mineru_subprocess_env() -> dict[str, str]:
    """加载 Step 2 持久化的 MINERU_HOME 与隔离 cuDNN PYTHONPATH Hook。"""
    env = os.environ.copy()
    if ENV_STATE_FILE.exists():
        try:
            state = json.loads(ENV_STATE_FILE.read_text(encoding="utf-8"))
            if state.get("MINERU_HOME"):
                env.setdefault("MINERU_HOME", state["MINERU_HOME"])
            if state.get("MINERU_MODEL_SOURCE"):
                env.setdefault("MINERU_MODEL_SOURCE", state["MINERU_MODEL_SOURCE"])
            hook_pp = state.get("MINERU_HOOK_PYTHONPATH", "")
            if hook_pp:
                existing_pp = env.get("PYTHONPATH", "")
                if hook_pp not in existing_pp.split(":"):
                    env["PYTHONPATH"] = f"{hook_pp}:{existing_pp}" if existing_pp else hook_pp
        except Exception as exc:
            print(f"⚠️ 读取 {ENV_STATE_FILE} 异常（将继续使用当前环境变量）: {exc}", flush=True)
    elif "MINERU_HOME" not in env:
        print(
            "⚠️ 未检测到 Step 2 环境状态文件或 MINERU_HOME，mineru-kit 可能使用默认缓存路径。",
            flush=True,
        )
    return env


def is_valid_zip(path: Path) -> bool:
    """校验 ZIP 存在、非空且结构完整（防止上次中途中断残留损坏包被错误跳过）。"""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with zipfile.ZipFile(path, "r") as zf:
            return len(zf.namelist()) > 0
    except zipfile.BadZipFile:
        return False


def format_batch_filename(start_page: int, end_page: int, total_pages: int) -> str:
    """按总页数动态补零（至少 4 位），避免 >999 页时字典序错乱。"""
    width = max(4, len(str(total_pages)))
    return f"pages_{start_page:0{width}d}_{end_page:0{width}d}.zip"


def find_existing_valid_batch(output_dir: Path, start_page: int, end_page: int, total_pages: int) -> Path | None:
    """兼容旧版 3 位补零文件与新版动态补零文件的断点续传检测。"""
    candidates = [
        output_dir / format_batch_filename(start_page, end_page, total_pages),
        output_dir / f"pages_{start_page:03d}_{end_page:03d}.zip",
    ]
    for cand in candidates:
        if is_valid_zip(cand):
            return cand
    return None


def run_batch(
    pdf_path: str,
    zip_out_path: Path,
    page_range: str,
    tier: str,
    ocr_mode: str,
    sub_env: dict[str, str],
) -> None:
    """先写入本地临时 ZIP，校验完整性后再原子拷贝到目标目录，并带静默心跳防 CLI 超时。"""
    tmp_local_zip = Path("/tmp") / f"{zip_out_path.stem}.{os.getpid()}.tmp.zip"
    tmp_local_zip.unlink(missing_ok=True)

    cmd = [
        "mineru-kit",
        "parse",
        pdf_path,
        "-o",
        str(tmp_local_zip),
        "--tier",
        tier,
        "--ocr-mode",
        ocr_mode,
        "--format",
        "zip",
        "--pages",
        page_range,
    ]

    last_output_time = [time.time()]
    stop_heartbeat = threading.Event()

    def _heartbeat():
        while not stop_heartbeat.wait(timeout=max(5.0, HEARTBEAT_SECONDS / 2)):
            idle = time.time() - last_output_time[0]
            if idle >= HEARTBEAT_SECONDS:
                print(f"💓 [心跳] 正在解析第 {page_range} 页（已运行且等待输出 {idle:.0f}s）...", flush=True)
                last_output_time[0] = time.time()

    hb_thread = threading.Thread(target=_heartbeat, daemon=True)
    hb_thread.start()

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=sub_env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            last_output_time[0] = time.time()
            print(line, end="", flush=True)
        process.wait()
    finally:
        stop_heartbeat.set()
        hb_thread.join(timeout=2.0)

    if process.returncode != 0:
        tmp_local_zip.unlink(missing_ok=True)
        raise RuntimeError(f"❌ 第 {page_range} 页解析失败（返回码 {process.returncode}）")

    if not is_valid_zip(tmp_local_zip):
        tmp_local_zip.unlink(missing_ok=True)
        raise RuntimeError(f"❌ 第 {page_range} 页生成的 ZIP 缺失或损坏: {tmp_local_zip}")

    drive_tmp = zip_out_path.with_suffix(".zip.uploading")
    shutil.copy2(tmp_local_zip, drive_tmp)
    drive_tmp.replace(zip_out_path)
    tmp_local_zip.unlink(missing_ok=True)


def main() -> None:
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"❌ 找不到 PDF 文件: {PDF_PATH}")
    if BATCH_SIZE <= 0:
        raise ValueError(f"❌ BATCH_SIZE 必须为正整数，当前: {BATCH_SIZE}")

    total_pages = int(TOTAL_PAGES_RAW) if TOTAL_PAGES_RAW.isdigit() and int(TOTAL_PAGES_RAW) > 0 else detect_pdf_total_pages(PDF_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sub_env = load_mineru_subprocess_env()

    print(
        f"📄 开始分批解析: {PDF_PATH} | 总页数={total_pages} | 每批={BATCH_SIZE} | "
        f"tier={TIER} | ocr={OCR_MODE} -> {OUTPUT_DIR}",
        flush=True,
    )

    for start_page in range(1, total_pages + 1, BATCH_SIZE):
        end_page = min(start_page + BATCH_SIZE - 1, total_pages)
        page_range = f"{start_page}-{end_page}"

        existing = find_existing_valid_batch(OUTPUT_DIR, start_page, end_page, total_pages)
        if existing is not None:
            print(f"✅ 第 {page_range} 页已存在且校验完整，跳过: {existing.name}", flush=True)
            continue

        zip_out_path = OUTPUT_DIR / format_batch_filename(start_page, end_page, total_pages)
        print(f"\n🚀 正在解析第 {page_range} 页 -> {zip_out_path.name} ...", flush=True)
        run_batch(PDF_PATH, zip_out_path, page_range, TIER, OCR_MODE, sub_env)
        print(f"\n🎉 第 {page_range} 页导出完成 -> {zip_out_path}", flush=True)

    print("\n✅ 所有批次解析完成！", flush=True)


if __name__ == "__main__":
    main()
