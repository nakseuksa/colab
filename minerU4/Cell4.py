import zipfile
from pathlib import Path

ZIP_DIR = Path("/content/drive/MyDrive/Pre-Algebra_zips")
MERGED_DIR = Path("/content/drive/MyDrive/mineru_merged_book")
MERGED_IMAGES_DIR = MERGED_DIR / "images"
MERGED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

zip_files = sorted(ZIP_DIR.glob("pages_*.zip"))
all_markdown_parts = []

for zf_path in zip_files:
    print(f"📦 正在提取并合并: {zf_path.name}")
    with zipfile.ZipFile(zf_path, "r") as zf:
        # 1. 读取该批次的 markdown.md
        if "markdown.md" in zf.namelist():
            md_text = zf.read("markdown.md").decode("utf-8").strip()
            all_markdown_parts.append(md_text)

        # 2. 解压所有图片到汇总目录（MinerU 图片带原始页码索引，不会重名冲突）
        for member in zf.namelist():
            if member.startswith("images/") and not member.endswith("/"):
                zf.extract(member, path=MERGED_DIR)

# 保存合并后的完整 Markdown
final_md_path = MERGED_DIR / "full_book.md"
final_md_path.write_text("\n\n".join(all_markdown_parts) + "\n", encoding="utf-8")

print(f"\n✅ 全书合并完成！")
print(f"📄 完整 Markdown 路径: {final_md_path}")
print(f"🖼️ 完整图片目录路径: {MERGED_IMAGES_DIR}")
