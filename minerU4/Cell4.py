"""MinerU 4.0 Step 4: 按页码数值顺序安全合并所有批次 ZIP 为完整 Markdown 与图片目录。

修复说明:
- 使用正则解析 pages_<start>_<end>.zip 中的整数起始页码排序，彻底修复 >999 页时字典序导致全书乱序的 P0 Bug。
- 增加 Zip Slip 路径穿越防护与页码断档/重叠告警。
- 支持通过环境变量 ZIP_DIR / MERGED_DIR 参数化配置。
"""

import os
import re
import zipfile
from pathlib import Path

ZIP_DIR = Path(os.environ.get("ZIP_DIR", "/content/drive/MyDrive/Pre-Algebra_zips"))
MERGED_DIR = Path(os.environ.get("MERGED_DIR", "/content/drive/MyDrive/mineru_merged_book"))

PAGE_RANGE_RE = re.compile(r"^pages_(\d+)_(\d+)\.zip$")


def parse_zip_page_range(path: Path) -> tuple[int, int]:
    """从 pages_<start>_<end>.zip 提取 (start_page, end_page) 整数元组。"""
    match = PAGE_RANGE_RE.match(path.name)
    if not match:
        raise ValueError(f"无法解析批次文件名页码范围: {path.name}")
    return int(match.group(1)), int(match.group(2))


def sort_batch_zips(zip_dir: Path) -> list[tuple[Path, int, int]]:
    """按起始页码数值升序排列所有批次 ZIP 文件，并检查页码连续性。"""
    matched: list[tuple[Path, int, int]] = []
    for p in zip_dir.glob("pages_*.zip"):
        if PAGE_RANGE_RE.match(p.name):
            start_p, end_p = parse_zip_page_range(p)
            matched.append((p, start_p, end_p))
    matched.sort(key=lambda item: (item[1], item[2], item[0].name))
    return matched


def _is_within_dir(target: Path, base: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def safe_extract_image(zf: zipfile.ZipFile, member: str, merged_dir: Path) -> Path | None:
    """防 Zip Slip 路径穿越，并检测潜在图片同名覆盖。"""
    if not member.startswith("images/") or member.endswith("/"):
        return None
    dest = (merged_dir / member).resolve()
    if not _is_within_dir(dest, merged_dir):
        raise ValueError(f"❌ 检测到非法 ZIP 成员路径 (Zip Slip): {member}")

    data = zf.read(member)
    if dest.exists() and dest.read_bytes() != data:
        print(f"⚠️ 图片同名但内容不同，将被覆盖: {member}", flush=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return dest


def merge_zips(zip_dir: Path = ZIP_DIR, merged_dir: Path = MERGED_DIR) -> Path:
    if not zip_dir.exists():
        raise FileNotFoundError(f"❌ 找不到 ZIP 目录: {zip_dir}")

    merged_images_dir = merged_dir / "images"
    merged_images_dir.mkdir(parents=True, exist_ok=True)

    sorted_batches = sort_batch_zips(zip_dir)
    if not sorted_batches:
        raise RuntimeError(f"❌ 在 {zip_dir} 中未找到任何 pages_<start>_<end>.zip 文件")

    all_markdown_parts: list[str] = []
    prev_end: int | None = None
    extracted_images = 0

    for zf_path, start_p, end_p in sorted_batches:
        if prev_end is not None and start_p != prev_end + 1:
            print(
                f"⚠️ 页码范围不连续: 上一批结束于第 {prev_end} 页，当前批次为 {start_p}-{end_p} ({zf_path.name})",
                flush=True,
            )
        prev_end = end_p

        print(f"📦 正在提取并合并 (第 {start_p}-{end_p} 页): {zf_path.name}", flush=True)
        with zipfile.ZipFile(zf_path, "r") as zf:
            names = zf.namelist()
            if "markdown.md" in names:
                md_text = zf.read("markdown.md").decode("utf-8").strip()
                if md_text:
                    all_markdown_parts.append(md_text)
            else:
                print(f"⚠️ {zf_path.name} 中缺少 markdown.md，已跳过文本合并。", flush=True)

            for member in names:
                if safe_extract_image(zf, member, merged_dir) is not None:
                    extracted_images += 1

    final_md_path = merged_dir / "full_book.md"
    final_md_path.write_text("\n\n".join(all_markdown_parts) + "\n", encoding="utf-8")

    print("\n✅ 全书合并完成！", flush=True)
    print(f"📄 完整 Markdown 路径: {final_md_path} ({len(sorted_batches)} 批次)", flush=True)
    print(f"🖼️ 完整图片目录路径: {merged_images_dir} (共 {extracted_images} 张图片)", flush=True)
    return final_md_path


if __name__ == "__main__":
    merge_zips()
