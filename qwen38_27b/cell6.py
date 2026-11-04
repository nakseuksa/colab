# Cell 6 (Colab 本地高速代码库深度理解工作台 - Qwen3.8-27B 进程内直连版):
# 1. 高速导入：将指定 Google Drive 代码库文件夹智能过滤（剔除 .git / node_modules 等）并多线程同步至 Colab 本地 NVMe 高速盘 (/content)
# 2. 双模式编程运行 Qwen3.8-27B（零网络开销，进程内直接调用 GPU 模型，实时刷新保活时间戳）：
#    - 模式 A [analyze_code_pack]: 自动生成目录树 + 按模块/Glob 通配符打包源码投喂
#    - 模式 B [explore_codebase_agent]: 内置本地只读工具循环 (list_dir / find_files / grep_search / read_file)，让模型在高速盘上自主多轮检索阅读代码
# 3. 纯净 MD 实时同步：自动剥离 <think>...</think> 思考过程，仅将最终正文保存为独立 .md 文档，并实时同步回 Google Drive（支持断点续传）

import os
import re
import gc
import time
import uuid
import json
import fnmatch
import shutil
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from transformers import TextIteratorStreamer

# ==============================================================================
# 0. 核心路径与全局默认配置（请根据实际 Google Drive 目录修改此处）
# ==============================================================================
# Google Drive 上的源代码库文件夹路径（支持普通代码目录，也兼容直接填入 .zip / .tar.gz 压缩包路径）
DRIVE_REPO_PATH = "/content/drive/MyDrive/my_project"

# 导入到 Colab 本地 NVMe 高速磁盘的目标路径
LOCAL_WORKSPACE_ROOT = "/content/local_codebase"

# 生成的 Markdown 报告在 Google Drive 上的持久化同步目录
DRIVE_MD_OUTPUT_DIR = "/content/drive/MyDrive/qwen_code_reports"

# 生成的 Markdown 报告在 Colab 本地高速盘的缓存目录
LOCAL_MD_OUTPUT_DIR = "/content/local_outputs"

# 默认推理参数
DEFAULT_CODE_REASONING_EFFORT = "medium"  # 可选: "low" | "medium" | "xhigh"
DEFAULT_MAX_NEW_TOKENS = 16384
MAX_SINGLE_FILE_BYTES = 1 * 1024 * 1024   # 同步与读取时默认跳过超过 1MB 的超大单文件（防止混入数据/二进制文件）

# 同步与检索代码库时默认忽略的目录名与文件后缀
DEFAULT_IGNORE_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", "out", "target", ".idea", ".vscode", ".next", ".nuxt",
    "coverage", "vendor", ".gradle", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "eggs", ".eggs", "bower_components", "site-packages",
}

DEFAULT_IGNORE_EXTS = {
    ".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib", ".exe", ".bin", ".o", ".a", ".lib",
    ".pkl", ".pt", ".pth", ".onnx", ".safetensors", ".gguf", ".h5", ".ckpt",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar", ".jar", ".war", ".class",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".svg", ".mp4", ".mp3", ".wav",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".db", ".sqlite", ".sqlite3",
    ".lock", ".min.js", ".min.css", ".map",
}


# ==============================================================================
# 1. 环境自检与底层推理/解析兼容层（无论是否运行过 Cell 4 均可独立工作）
# ==============================================================================
if "model" not in globals() or model is None or "tokenizer" not in globals() or tokenizer is None:
    raise RuntimeError("❌ 未检测到已加载的 Qwen 模型或分词器，请先运行 Cell 1 和 Cell 2！")

if "LAST_API_REQUEST_TIME" not in globals():
    globals()["LAST_API_REQUEST_TIME"] = time.time()


def _touch_keepalive():
    """刷新全局活跃时间戳，防止 Cell 5 的保活守护误判闲置"""
    globals()["LAST_API_REQUEST_TIME"] = time.time()


if "TokenCountingStreamer" not in globals():
    class TokenCountingStreamer(TextIteratorStreamer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.generated_tokens = 0
            self._first_put = True

        def put(self, value):
            if self.skip_prompt and self._first_put:
                self._first_put = False
            else:
                if isinstance(value, torch.Tensor):
                    self.generated_tokens += int(value.numel())
                elif isinstance(value, (list, tuple)):
                    self.generated_tokens += len(value)
            super().put(value)

_TAG_THINK_OPEN = "<" + "think>"
_TAG_THINK_CLOSE = "</" + "think>"
_TAG_TC_OPEN = "<tool_" + "call>"
_TAG_TC_CLOSE = "</tool_" + "call>"
_TAG_TOOLS_OPEN = "<" + "tools>"
_TAG_TOOLS_CLOSE = "</" + "tools>"
_TAG_FN_PREFIX = "<function="
_TAG_FN_CLOSE = "</" + "function>"
_TAG_PARAM_PREFIX = "<parameter="
_TAG_PARAM_CLOSE = "</" + "parameter>"

_REGEX_THINK_BLOCK = re.compile(re.escape(_TAG_THINK_OPEN) + r"(.*?)" + re.escape(_TAG_THINK_CLOSE), flags=re.DOTALL)
_REGEX_THINK_LEADING = re.compile(r"^" + re.escape(_TAG_THINK_OPEN) + r"\s*")
_REGEX_TC_BLOCK = re.compile(re.escape(_TAG_TC_OPEN) + r"(.*?)" + re.escape(_TAG_TC_CLOSE), flags=re.DOTALL)
_REGEX_FN_XML = re.compile(re.escape(_TAG_FN_PREFIX) + r"([^\s>]+)>(.*?)" + re.escape(_TAG_FN_CLOSE), flags=re.DOTALL)
_REGEX_PARAM_XML = re.compile(re.escape(_TAG_PARAM_PREFIX) + r"([^\s>]+)>(.*?)" + re.escape(_TAG_PARAM_CLOSE), flags=re.DOTALL)


def _extract_tool_call_fallback(block_str: str, idx: int):
    block_str = block_str.strip()
    fn_xml_match = _REGEX_FN_XML.search(block_str)
    if fn_xml_match:
        fn_name = fn_xml_match.group(1).strip()
        params_body = fn_xml_match.group(2)
        args_dict = {}
        for p_match in _REGEX_PARAM_XML.finditer(params_body):
            p_key = p_match.group(1).strip()
            p_val = p_match.group(2).strip().strip("\n")
            try:
                args_dict[p_key] = json.loads(p_val)
            except Exception:
                args_dict[p_key] = p_val
        return {
            "index": idx,
            "id": f"call_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {"name": fn_name, "arguments": json.dumps(args_dict, ensure_ascii=False)},
        }

    start_brace = block_str.find("{")
    end_brace = block_str.rfind("}")
    if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
        try:
            tc_obj = json.loads(block_str[start_brace : end_brace + 1])
            fn_name = tc_obj.get("name") or tc_obj.get("function", "")
            fn_args = tc_obj.get("arguments", tc_obj.get("parameters", {}))
            if not isinstance(fn_args, str):
                fn_args = json.dumps(fn_args, ensure_ascii=False)
            return {
                "index": idx,
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {"name": fn_name, "arguments": fn_args},
            }
        except Exception:
            pass
    return None


def parse_qwen_clean_output(raw_text: str, prompt_opened_think: bool = True):
    """解析模型输出，剥离 <think> 思考过程，返回 (reasoning_content, clean_markdown, tool_calls)"""
    if "parse_qwen_output" in globals() and callable(globals()["parse_qwen_output"]):
        return globals()["parse_qwen_output"](raw_text, prompt_opened_think=prompt_opened_think)

    for eos_tok in ("<|im_end|>", "<|endoftext|>"):
        raw_text = raw_text.replace(eos_tok, "")

    reasoning_content = None
    body_text = raw_text

    think_match = _REGEX_THINK_BLOCK.search(raw_text)
    if think_match:
        reasoning_content = think_match.group(1).strip()
        body_text = raw_text[: think_match.start()] + raw_text[think_match.end() :]
    elif _TAG_THINK_CLOSE in raw_text:
        think_part, rest_part = raw_text.split(_TAG_THINK_CLOSE, 1)
        think_part = _REGEX_THINK_LEADING.sub("", think_part).strip()
        if think_part:
            reasoning_content = think_part
        body_text = rest_part
    elif prompt_opened_think:
        reasoning_content = _REGEX_THINK_LEADING.sub("", raw_text).strip()
        body_text = ""

    tool_calls = []
    for idx, match in enumerate(_REGEX_TC_BLOCK.finditer(body_text)):
        parsed_tc = _extract_tool_call_fallback(match.group(1), idx)
        if parsed_tc:
            tool_calls.append(parsed_tc)

    if not tool_calls and _TAG_TC_OPEN in body_text:
        unclosed_part = body_text.split(_TAG_TC_OPEN, 1)[1]
        parsed_tc = _extract_tool_call_fallback(unclosed_part, 0)
        if parsed_tc:
            tool_calls.append(parsed_tc)
            body_text = body_text.split(_TAG_TC_OPEN, 1)[0]

    clean_content = _REGEX_TC_BLOCK.sub("", body_text).strip()
    return reasoning_content, clean_content, tool_calls


# ==============================================================================
# 2. 模块一：Google Drive 代码库高速过滤导入 Colab 本地磁盘
# ==============================================================================
def _should_ignore_dir(dir_name: str, extra_ignore_dirs=None) -> bool:
    ignore_set = DEFAULT_IGNORE_DIRS | set(extra_ignore_dirs or [])
    return dir_name in ignore_set or (dir_name.startswith(".") and dir_name not in {".github", ".kilo"})


def _should_ignore_file(file_path: Path, max_file_bytes: int = MAX_SINGLE_FILE_BYTES) -> bool:
    name_lower = file_path.name.lower()
    for ext in DEFAULT_IGNORE_EXTS:
        if name_lower.endswith(ext):
            return True
    try:
        st = file_path.stat()
        if st.st_size > max_file_bytes:
            return True
    except OSError:
        return True
    return False


def sync_drive_repo_to_local(
    drive_source_path: str = DRIVE_REPO_PATH,
    local_workspace_root: str = LOCAL_WORKSPACE_ROOT,
    force_clean: bool = False,
    max_file_bytes: int = MAX_SINGLE_FILE_BYTES,
    workers: int = 8,
) -> Path:
    """
    将 Google Drive 中的指定代码文件夹（或压缩包）高速导入到 Colab 本地 NVMe 磁盘。
    - 自动跳过 .git / node_modules / __pycache__ / build 等无关目录与二进制大文件
    - 支持增量比对（已存在且大小/修改时间一致的文件直接跳过）
    - 采用多线程并发克服 Google Drive FUSE 小文件读取延迟
    """
    src_path = Path(drive_source_path)
    if not src_path.exists():
        raise FileNotFoundError(
            f"❌ 找不到指定的 Google Drive 路径: {src_path}\n"
            f"请确认 Google Drive 已挂载 (/content/drive/MyDrive/...) 且路径填写正确！"
        )

    repo_name = src_path.stem if src_path.is_file() else src_path.name
    dst_root = Path(local_workspace_root) / repo_name

    if force_clean and dst_root.exists():
        print(f"🧹 正在清理旧的本地高速盘缓存: {dst_root}")
        shutil.rmtree(dst_root)

    dst_root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 兼容直接指向 .zip / .tar.gz 压缩包的情况
    if src_path.is_file() and src_path.suffix.lower() in {".zip", ".gz", ".tar", ".tgz"}:
        local_archive = Path(local_workspace_root) / src_path.name
        print(f"📦 检测到压缩包源 [{src_path.name}]，正在拷入高速盘并解压...")
        shutil.copy2(src_path, local_archive)
        shutil.unpack_archive(str(local_archive), str(dst_root))
        local_archive.unlink(missing_ok=True)
        print(f"✅ 压缩包导入解压完成！本地高速路径: {dst_root} (耗时 {time.time() - t0:.2f}s)")
        return dst_root

    print(f"🚀 开始扫描 Google Drive 代码库: {src_path} ...")
    copy_tasks = []
    skipped_unchanged = 0
    ignored_files = 0

    for root, dirs, files in os.walk(src_path):
        # 原地修改 dirs 以剪枝无需遍历的重型目录（如 .git, node_modules）
        dirs[:] = [d for d in sorted(dirs) if not _should_ignore_dir(d)]
        root_path = Path(root)
        rel_dir = root_path.relative_to(src_path)
        target_dir = dst_root / rel_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        for fname in files:
            s_file = root_path / fname
            if _should_ignore_file(s_file, max_file_bytes=max_file_bytes):
                ignored_files += 1
                continue

            d_file = target_dir / fname
            try:
                s_stat = s_file.stat()
                if d_file.exists():
                    d_stat = d_file.stat()
                    if d_stat.st_size == s_stat.st_size and abs(d_stat.st_mtime - s_stat.st_mtime) < 1.0:
                        skipped_unchanged += 1
                        continue
                copy_tasks.append((s_file, d_file, s_stat.st_size))
            except OSError:
                continue

    total_bytes = sum(item[2] for item in copy_tasks)
    print(
        f"📋 扫描完毕: 需拷贝更新 {len(copy_tasks)} 个源码文件 ({total_bytes / (1024 * 1024):.2f} MB) | "
        f"增量跳过 {skipped_unchanged} 个未变更文件 | 过滤无关/大文件 {ignored_files} 个"
    )

    if copy_tasks:
        copied_count = 0

        def _copy_one(task):
            s_f, d_f, _ = task
            shutil.copy2(s_f, d_f)
            return 1

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_copy_one, t) for t in copy_tasks]
            for fut in as_completed(futures):
                copied_count += fut.result()
                if copied_count % 50 == 0 or copied_count == len(copy_tasks):
                    print(f"\r⚡ 正在高速导入至本地 NVMe: {copied_count}/{len(copy_tasks)} 文件...", end="", flush=True)
        print()

    elapsed = time.time() - t0
    print(f"✅ 代码库已就绪！Colab 本地高速盘路径: {dst_root} (总耗时 {elapsed:.2f}s)")
    return dst_root


# ==============================================================================
# 3. 模块二：本地高速盘代码检索、目录树生成与只读工具集
# ==============================================================================
def build_repo_tree(
    repo_dir: Path,
    rel_subpath: str = ".",
    max_depth: int = 4,
    max_entries: int = 300,
) -> str:
    """生成代码库的清晰 ASCII 目录树（供 Prompt 上下文或 list_dir 工具使用）"""
    base = (Path(repo_dir) / rel_subpath).resolve()
    repo_root = Path(repo_dir).resolve()
    if not str(base).startswith(str(repo_root)) or not base.exists():
        return f"❌ 路径不存在或越界: {rel_subpath}"

    lines = [f"📁 {base.relative_to(repo_root) if base != repo_root else repo_root.name}/"]
    entry_count = 0

    def _walk(curr_path: Path, prefix: str, depth: int):
        nonlocal entry_count
        if depth > max_depth or entry_count >= max_entries:
            return
        try:
            items = sorted(curr_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            return

        filtered = []
        for it in items:
            if it.is_dir():
                if not _should_ignore_dir(it.name):
                    filtered.append(it)
            else:
                if not _should_ignore_file(it):
                    filtered.append(it)

        for idx, item in enumerate(filtered):
            if entry_count >= max_entries:
                lines.append(f"{prefix}└── ... (已达到显示上限 {max_entries} 项)")
                break
            is_last = idx == len(filtered) - 1
            connector = "└── " if is_last else "├── "
            if item.is_dir():
                lines.append(f"{prefix}{connector}{item.name}/")
                entry_count += 1
                next_prefix = prefix + ("    " if is_last else "│   ")
                _walk(item, next_prefix, depth + 1)
            else:
                size_kb = item.stat().st_size / 1024.0
                lines.append(f"{prefix}{connector}{item.name} ({size_kb:.1f} KB)")
                entry_count += 1

    _walk(base, "", 1)
    return "\n".join(lines)


def tool_find_files(repo_dir: Path, pattern: str = "*", max_results: int = 100) -> str:
    """按文件名或相对路径通配符（如 '*.py' 或 'src/**/*.ts'）在本地高速盘查找文件"""
    repo_root = Path(repo_dir).resolve()
    matches = []
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in sorted(dirs) if not _should_ignore_dir(d)]
        for f in sorted(files):
            fpath = Path(root) / f
            if _should_ignore_file(fpath):
                continue
            rel_str = fpath.relative_to(repo_root).as_posix()
            if fnmatch.fnmatch(f, pattern) or fnmatch.fnmatch(rel_str, pattern):
                matches.append(rel_str)
                if len(matches) >= max_results:
                    break
        if len(matches) >= max_results:
            break

    if not matches:
        return f"未找到匹配 '{pattern}' 的文件。"
    header = f"共找到 {len(matches)} 个匹配文件" + (f"（已截断前 {max_results} 个）：\n" if len(matches) == max_results else "：\n")
    return header + "\n".join(f"- {m}" for m in matches)


def tool_grep_search(
    repo_dir: Path,
    pattern: str,
    subpath: str = ".",
    file_pattern: str = "*",
    max_results: int = 60,
) -> str:
    """在本地高速盘代码库中使用正则表达式或关键字搜索代码行"""
    repo_root = Path(repo_dir).resolve()
    search_base = (repo_root / subpath).resolve()
    if not str(search_base).startswith(str(repo_root)) or not search_base.exists():
        return f"❌ 搜索路径不存在或越界: {subpath}"

    try:
        regex = re.compile(pattern)
    except re.error:
        regex = re.compile(re.escape(pattern), re.IGNORECASE)

    results = []
    files_to_check = [search_base] if search_base.is_file() else []
    if search_base.is_dir():
        for root, dirs, files in os.walk(search_base):
            dirs[:] = [d for d in sorted(dirs) if not _should_ignore_dir(d)]
            for f in sorted(files):
                fpath = Path(root) / f
                if _should_ignore_file(fpath):
                    continue
                rel_str = fpath.relative_to(repo_root).as_posix()
                if fnmatch.fnmatch(f, file_pattern) or fnmatch.fnmatch(rel_str, file_pattern):
                    files_to_check.append(fpath)

    for fpath in files_to_check:
        rel_str = fpath.relative_to(repo_root).as_posix()
        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                snippet = line.strip()
                if len(snippet) > 180:
                    snippet = snippet[:180] + "..."
                results.append(f"{rel_str}:{lineno}: {snippet}")
                if len(results) >= max_results:
                    return f"🔍 搜索 '{pattern}' 命中结果（前 {max_results} 条）：\n" + "\n".join(results)

    if not results:
        return f"🔍 未找到匹配 '{pattern}' 的代码行。"
    return f"🔍 搜索 '{pattern}' 共命中 {len(results)} 条结果：\n" + "\n".join(results)


def tool_read_file(
    repo_dir: Path,
    file_path: str,
    start_line: int = 1,
    end_line: int = 250,
) -> str:
    """读取本地高速盘上的指定源码文件（带行号，支持指定起止行防止撑爆上下文）"""
    repo_root = Path(repo_dir).resolve()
    target = (repo_root / file_path.lstrip("/\\")).resolve()
    if not str(target).startswith(str(repo_root)) or not target.exists() or not target.is_file():
        return f"❌ 文件不存在或路径非法: {file_path}"

    try:
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as e:
        return f"❌ 读取文件失败 ({file_path}): {e}"

    total_lines = len(lines)
    s_idx = max(1, int(start_line))
    e_idx = min(total_lines, max(s_idx, int(end_line)))
    selected = [f"{i:4d} | {lines[i - 1]}" for i in range(s_idx, e_idx + 1)]
    rel_str = target.relative_to(repo_root).as_posix()
    header = f"📄 文件: {rel_str} (显示第 {s_idx}~{e_idx} 行 / 共 {total_lines} 行)\n"
    return header + "```" + target.suffix.lstrip(".") + "\n" + "\n".join(selected) + "\n```"


# 提供给 Qwen3.8 Agent 模式的 OpenAI 标准 Tools 定义
CODEBASE_AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "查看代码库指定子目录的树状结构与文件大小。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对代码库根目录的子路径，默认为 '.'"},
                    "max_depth": {"type": "integer", "description": "最大展开层级，默认为 3"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_files",
            "description": "按文件名或相对路径通配符（如 '*.py' 或 'src/**/*.ts'）查找代码库中的文件列表。",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob 通配符模式，如 '*.py' 或 '*router*'"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_search",
            "description": "在代码库中使用正则表达式或关键字搜索类、函数定义或调用位置，返回文件路径、行号和代码行。",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "正则表达式或搜索关键字，例如 'class .*Service' 或 'def main'"},
                    "subpath": {"type": "string", "description": "限制搜索的相对子目录或文件路径，默认为 '.'"},
                    "file_pattern": {"type": "string", "description": "限制文件后缀通配符，例如 '*.py'，默认为 '*'"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "按行号范围读取指定源码文件的内容（带行号）。建议每次读取 150~300 行以保持上下文高效。",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "相对代码库根目录的文件路径"},
                    "start_line": {"type": "integer", "description": "起始行号（从 1 开始），默认为 1"},
                    "end_line": {"type": "integer", "description": "结束行号，默认为 250"},
                },
                "required": ["file_path"],
            },
        },
    },
]


def _execute_local_code_tool(repo_dir: Path, fn_name: str, args_raw) -> str:
    """在 Colab 本地高速盘执行只读代码工具"""
    if isinstance(args_raw, str):
        try:
            args = json.loads(args_raw) if args_raw.strip() else {}
        except Exception:
            args = {}
    elif isinstance(args_raw, dict):
        args = args_raw
    else:
        args = {}

    if fn_name == "list_dir":
        return build_repo_tree(
            repo_dir,
            rel_subpath=str(args.get("path", ".")),
            max_depth=int(args.get("max_depth", 3)),
        )
    elif fn_name == "find_files":
        return tool_find_files(repo_dir, pattern=str(args.get("pattern", "*")))
    elif fn_name == "grep_search":
        return tool_grep_search(
            repo_dir,
            pattern=str(args.get("pattern", "")),
            subpath=str(args.get("subpath", ".")),
            file_pattern=str(args.get("file_pattern", "*")),
        )
    elif fn_name == "read_file":
        return tool_read_file(
            repo_dir,
            file_path=str(args.get("file_path", "")),
            start_line=int(args.get("start_line", 1)),
            end_line=int(args.get("end_line", 250)),
        )
    else:
        return f"❌ 未知工具调用: {fn_name}"


# ==============================================================================
# 4. 模块三：Markdown 智能命名、纯净正文落盘与 Google Drive 实时同步
# ==============================================================================
if "CHAT_HISTORY" not in globals():
    globals()["CHAT_HISTORY"] = []

if "MD_AUTO_SEQ" not in globals():
    globals()["MD_AUTO_SEQ"] = 1


def reset_chat_history():
    """手动清空多轮对话历史记忆"""
    globals()["CHAT_HISTORY"] = []
    print("🧹 已清空多轮对话历史 (CHAT_HISTORY)。")


def _build_auto_md_name(prompt: str, repo_name: str = None, drive_out_dir: str = DRIVE_MD_OUTPUT_DIR) -> str:
    """当 OUTPUT_MD_NAME 留空时，按 '序号_时分秒_Prompt摘要.md' 自动生成不冲突的文件名"""
    sub_folder = repo_name.strip() if repo_name else ""
    drive_base = Path(drive_out_dir) / sub_folder if sub_folder else Path(drive_out_dir)
    existing_count = len(list(drive_base.glob("*.md"))) if drive_base.exists() else 0
    seq = max(int(globals().get("MD_AUTO_SEQ", 1)), existing_count + 1)
    globals()["MD_AUTO_SEQ"] = seq + 1

    first_line = ""
    for line in (prompt or "").splitlines():
        if line.strip():
            first_line = line.strip()
            break
    slug = re.sub(r'[\\/:*?"<>|\s#`]+', "_", first_line).strip("_")[:18] or "对话记录"
    ts = datetime.now().strftime("%H%M%S")
    return f"{seq:02d}_{ts}_{slug}.md"


def resolve_output_paths(
    output_md_name: str,
    repo_name: str = None,
    local_out_dir: str = LOCAL_MD_OUTPUT_DIR,
    drive_out_dir: str = DRIVE_MD_OUTPUT_DIR,
    prompt_for_auto_name: str = "",
) -> tuple[Path, Path]:
    """解析本地高速盘与 Google Drive 的目标 Markdown 路径（支持留空自动命名）"""
    clean_name = (output_md_name or "").strip()
    if not clean_name:
        clean_name = _build_auto_md_name(prompt_for_auto_name, repo_name=repo_name, drive_out_dir=drive_out_dir)
    elif not clean_name.lower().endswith(".md"):
        clean_name += ".md"

    sub_folder = repo_name.strip() if repo_name else ""
    local_base = Path(local_out_dir) / sub_folder if sub_folder else Path(local_out_dir)
    drive_base = Path(drive_out_dir) / sub_folder if sub_folder else Path(drive_out_dir)

    local_base.mkdir(parents=True, exist_ok=True)
    drive_base.mkdir(parents=True, exist_ok=True)
    return local_base / clean_name, drive_base / clean_name


def load_context_mds_content(
    context_mds: list[str],
    repo_name: str = None,
    local_out_dir: str = LOCAL_MD_OUTPUT_DIR,
    drive_out_dir: str = DRIVE_MD_OUTPUT_DIR,
) -> str:
    """加载已生成的 Markdown 报告内容作为本轮对话的背景知识"""
    if not context_mds:
        return ""

    sub_folder = repo_name.strip() if repo_name else ""
    search_dirs = [
        Path(local_out_dir) / sub_folder if sub_folder else Path(local_out_dir),
        Path(drive_out_dir) / sub_folder if sub_folder else Path(drive_out_dir),
        Path(local_out_dir),
        Path(drive_out_dir),
    ]

    blocks = []
    for md_item in context_mds:
        if not md_item or not str(md_item).strip():
            continue
        raw_name = str(md_item).strip()
        cand_names = [raw_name, raw_name + ".md"] if not raw_name.lower().endswith(".md") else [raw_name]

        found_path = None
        direct_p = Path(raw_name)
        if direct_p.is_absolute() and direct_p.exists():
            found_path = direct_p
        else:
            for s_dir in search_dirs:
                for c_name in cand_names:
                    p = s_dir / c_name
                    if p.exists() and p.is_file():
                        found_path = p
                        break
                if found_path:
                    break

        if found_path:
            try:
                txt = found_path.read_text(encoding="utf-8", errors="replace").strip()
                blocks.append(f"### 参考文档: `{found_path.name}`\n{txt}")
                print(f"📎 已加载参考 MD 上下文: {found_path.name} ({len(txt)} 字符)")
            except OSError as e:
                print(f"⚠️ 读取参考 MD 失败 ({found_path}): {e}")
        else:
            print(f"⚠️ 未找到指定的参考 MD 文档: {raw_name}")

    if not blocks:
        return ""
    return "## 已生成的参考分析文档 (Context MDs)\n\n" + "\n\n---\n\n".join(blocks)


def save_and_sync_md(
    clean_markdown: str,
    output_md_name: str,
    repo_name: str = None,
    local_out_dir: str = LOCAL_MD_OUTPUT_DIR,
    drive_out_dir: str = DRIVE_MD_OUTPUT_DIR,
    prompt_for_auto_name: str = "",
) -> tuple[Path, Path]:
    """
    将清洗后的纯净 Markdown 正文（不含 <think> 思考过程）写入本地高速盘，
    并立即通过原子复制 + os.sync() 同步至 Google Drive。
    """
    local_md_path, drive_md_path = resolve_output_paths(
        output_md_name=output_md_name,
        repo_name=repo_name,
        local_out_dir=local_out_dir,
        drive_out_dir=drive_out_dir,
        prompt_for_auto_name=prompt_for_auto_name,
    )

    content_to_write = clean_markdown.strip() + "\n"
    local_md_path.write_text(content_to_write, encoding="utf-8")

    # 同步至 Google Drive（若挂载了云盘）
    shutil.copy2(local_md_path, drive_md_path)
    try:
        os.sync()
    except AttributeError:
        pass

    print(f"💾 [MD 已保存并同步至 Google Drive]")
    print(f"   ├─ Colab 高速盘: {local_md_path}")
    print(f"   └─ Google Drive: {drive_md_path} ({drive_md_path.stat().st_size / 1024.0:.1f} KB)")
    return local_md_path, drive_md_path


# ==============================================================================
# 5. 模块四：Qwen3.8-27B 进程内直接推理核心引擎
# ==============================================================================
def _run_qwen_inprocess_once(
    messages: list[dict],
    tools: list[dict] = None,
    enable_thinking: bool = True,
    reasoning_effort: str = DEFAULT_CODE_REASONING_EFFORT,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = None,
    stream_to_console: bool = True,
) -> tuple[str, str, list[dict], int, float]:
    """
    在 Colab 进程内直接调用 GPU 上的 Qwen3.8-27B 执行单轮推理。
    返回: (reasoning_content, clean_markdown_content, tool_calls, generated_tokens, elapsed_seconds)
    """
    _touch_keepalive()
    if "install_model_concurrent_hook" in globals() and callable(globals()["install_model_concurrent_hook"]):
        globals()["install_model_concurrent_hook"](model)

    # 优先复用 Cell 4 的 prepare_prompt_and_inputs_sync
    if "prepare_prompt_and_inputs_sync" in globals() and callable(globals()["prepare_prompt_and_inputs_sync"]):
        prompt_text, prompt_opened_think, cpu_inputs, _ = globals()["prepare_prompt_and_inputs_sync"](
            messages,
            enable_thinking=enable_thinking,
            preserve_thinking=True,
            reasoning_effort=reasoning_effort,
            tools=tools,
        )
    else:
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": enable_thinking,
        }
        if enable_thinking:
            template_kwargs["reasoning_effort"] = reasoning_effort
        if tools:
            template_kwargs["tools"] = tools
        try:
            prompt_text = tokenizer.apply_chat_template(messages, **template_kwargs)
        except Exception:
            template_kwargs.pop("tools", None)
            prompt_text = tokenizer.apply_chat_template(messages, **template_kwargs)
        prompt_opened_think = enable_thinking and prompt_text.rstrip().endswith(_TAG_THINK_OPEN)
        cpu_inputs = tokenizer([prompt_text], return_tensors="pt", add_special_tokens=False)

    prompt_tokens = int(cpu_inputs["input_ids"].shape[1])

    if enable_thinking:
        temp = float(temperature) if temperature is not None else 1.0
        sampling_kwargs = dict(do_sample=True, temperature=temp, top_p=0.95, top_k=20) if temp > 0 else dict(do_sample=False)
    else:
        temp = float(temperature) if temperature is not None else 0.7
        sampling_kwargs = (
            dict(do_sample=True, temperature=temp, top_p=0.80, top_k=20, repetition_penalty=1.05)
            if temp > 0
            else dict(do_sample=False)
        )

    has_scheduler = "g4_scheduler" in globals() and globals()["g4_scheduler"] is not None
    if has_scheduler:
        globals()["g4_scheduler"].inc_session()

    model_inputs = {
        k: (
            v.to(model.device, dtype=model.dtype, non_blocking=True)
            if isinstance(v, torch.Tensor) and v.is_floating_point()
            else (v.to(model.device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
        )
        for k, v in cpu_inputs.items()
    }
    del cpu_inputs

    streamer = TokenCountingStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
    generation_kwargs = dict(
        **model_inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        **sampling_kwargs,
    )

    print(
        f"🧠 [Qwen3.8 本地推理启动] Prompt={prompt_tokens} tok | "
        f"thinking={enable_thinking}({reasoning_effort if enable_thinking else 'off'}) | "
        f"max_new_tokens={max_new_tokens}"
    )

    t0 = time.time()

    def _gen_worker():
        with torch.inference_mode():
            model.generate(**generation_kwargs)

    thread = threading.Thread(target=_gen_worker, daemon=True)
    thread.start()

    chunks = []
    in_think_print = prompt_opened_think
    if stream_to_console and in_think_print:
        print("💭 [思考中...] ", end="", flush=True)

    try:
        for piece in streamer:
            _touch_keepalive()
            if not piece:
                continue
            chunks.append(piece)
            if stream_to_console:
                display_piece = piece.replace("<|im_end|>", "").replace("<|endoftext|>", "")
                if _TAG_THINK_CLOSE in display_piece:
                    before_c, after_c = display_piece.split(_TAG_THINK_CLOSE, 1)
                    print(before_c, end="", flush=True)
                    print("\n" + "-" * 60 + "\n📝 [开始输出正文]\n", flush=True)
                    print(after_c.lstrip("\n"), end="", flush=True)
                    in_think_print = False
                else:
                    print(display_piece, end="", flush=True)
    finally:
        thread.join(timeout=5.0)
        _touch_keepalive()
        del model_inputs, generation_kwargs
        if has_scheduler:
            rem = globals()["g4_scheduler"].dec_session()
            if rem == 0 and "smart_vram_cleanup_if_needed" in globals():
                globals()["smart_vram_cleanup_if_needed"]()
        else:
            gc.collect()
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    raw_output = "".join(chunks)
    reasoning_content, clean_content, tool_calls = parse_qwen_clean_output(
        raw_output, prompt_opened_think=prompt_opened_think
    )
    gen_tokens = streamer.generated_tokens
    speed = gen_tokens / max(elapsed, 1e-5)
    print(f"\n📊 [本轮推理完成] 生成: {gen_tokens} tok | 耗时: {elapsed:.2f}s | 速率: {speed:.1f} tok/s")
    return reasoning_content, clean_content, tool_calls, gen_tokens, elapsed


# ==============================================================================
# 6. 面向用户的三大模式与统一对话接口 ask_qwen (支持 Agent / Pack / Chat + 多轮记忆 + 引用历史 MD)
# ==============================================================================
CHINESE_OUTPUT_SYSTEM_RULE = (
    "\n\n【强制语言规范】\n"
    "1. 无论代码库中的源码、注释、文档或工具返回内容是英文还是其他语言，你的最终回答与生成的 Markdown 报告必须全程使用**简体中文**撰写。\n"
    "2. 专有名词、文件路径、类名、函数名、变量名、命令行指令及引用代码片段保留原始英文标识符，其余所有分析、架构解读、流程说明、图表节点描述与总结均必须使用自然、专业的简体中文。"
)

CHINESE_OUTPUT_USER_REMINDER = (
    "\n\n（注意：请务必全程使用**简体中文**输出最终的 Markdown 分析报告正文，仅保留代码标识符与路径为原英文。）"
)


def _ensure_chinese_system_prompt(sys_prompt: str, force_chinese: bool = True) -> str:
    if not force_chinese:
        return sys_prompt
    if "强制语言规范" in sys_prompt or "全程使用**简体中文**" in sys_prompt:
        return sys_prompt
    return sys_prompt.rstrip() + CHINESE_OUTPUT_SYSTEM_RULE


def _pack_repo_files(repo_root: Path, include_patterns: list[str], exclude_patterns: list[str] = None) -> list[tuple[str, Path]]:
    """按相对路径或 Glob 通配符匹配本地高速盘中的源码文件"""
    if not include_patterns:
        return []
    exc_list = exclude_patterns or []
    packed_files = []
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in sorted(dirs) if not _should_ignore_dir(d)]
        for fname in sorted(files):
            fpath = Path(root) / fname
            if _should_ignore_file(fpath):
                continue
            rel_str = fpath.relative_to(repo_root).as_posix()
            matched_inc = any(
                fnmatch.fnmatch(fname, pat) or fnmatch.fnmatch(rel_str, pat) or rel_str.startswith(pat.rstrip("/"))
                for pat in include_patterns
            )
            if not matched_inc:
                continue
            matched_exc = any(
                fnmatch.fnmatch(fname, pat) or fnmatch.fnmatch(rel_str, pat)
                for pat in exc_list
            )
            if not matched_exc:
                packed_files.append((rel_str, fpath))
    return packed_files


def analyze_code_pack(
    task_prompt: str,
    output_md_name: str = "",
    local_repo_dir: str | Path = None,
    include_patterns: list[str] = None,
    exclude_patterns: list[str] = None,
    include_tree: bool = True,
    tree_max_depth: int = 4,
    keep_history: bool = False,
    context_mds: list[str] = None,
    force_chinese: bool = True,
    system_prompt: str = (
        "你是一位资深软件架构师与源码分析专家。请基于提供的代码库目录结构、参考文档与源码文件内容，"
        "使用简体中文给出结构清晰、深入详实的 Markdown 技术分析文档（包含必要的 Mermaid 架构/时序图、核心类/函数职责与关键代码路径引用）。"
    ),
    enable_thinking: bool = True,
    reasoning_effort: str = DEFAULT_CODE_REASONING_EFFORT,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    overwrite: bool = True,
    drive_out_dir: str = DRIVE_MD_OUTPUT_DIR,
) -> str:
    """
    【模式 A：目录树 + 按模块/Glob 路径打包源码投喂】
    直接从 Colab 本地高速盘读取匹配的源码文件并打包进上下文，生成的最终正文（剥离思考过程）保存为独立 .md 并同步至 Google Drive。
    """
    target_repo = local_repo_dir or globals().get("LOCAL_REPO_PATH")
    if not target_repo:
        raise RuntimeError("❌ 未指定 local_repo_dir 且全局 LOCAL_REPO_PATH 未初始化，请先运行 sync_drive_repo_to_local！")
    repo_root = Path(target_repo).resolve()
    repo_name = repo_root.name

    explicit_name = (output_md_name or "").strip()
    if explicit_name and not overwrite:
        _, check_drive_path = resolve_output_paths(
            output_md_name=explicit_name,
            repo_name=repo_name,
            drive_out_dir=drive_out_dir,
        )
        if check_drive_path.exists() and check_drive_path.stat().st_size > 0:
            print(f"⏭️ [断点跳过] 目标报告已存在于 Google Drive: {check_drive_path}")
            return check_drive_path.read_text(encoding="utf-8")

    if not keep_history:
        globals()["CHAT_HISTORY"] = []
    history_msgs = list(globals().get("CHAT_HISTORY", []))

    context_sections = []
    ctx_md_text = load_context_mds_content(context_mds, repo_name=repo_name, drive_out_dir=drive_out_dir)
    if ctx_md_text:
        context_sections.append(ctx_md_text)

    if include_tree:
        tree_str = build_repo_tree(repo_root, max_depth=tree_max_depth)
        context_sections.append(f"## 代码库目录结构 (`{repo_name}`)\n```text\n{tree_str}\n```")

    packed_files = _pack_repo_files(repo_root, include_patterns, exclude_patterns)
    if packed_files:
        file_blocks = []
        for rel_str, fpath in packed_files:
            try:
                code_text = fpath.read_text(encoding="utf-8", errors="replace")
                ext = fpath.suffix.lstrip(".")
                file_blocks.append(f"### 文件: `{rel_str}`\n```{ext}\n{code_text}\n```")
            except OSError:
                continue
        context_sections.append(f"## 源码文件内容（共打包 {len(file_blocks)} 个文件）\n\n" + "\n\n".join(file_blocks))

    prefix_ctx = ("\n\n".join(context_sections) + "\n\n---\n\n") if context_sections else ""
    lang_tail = CHINESE_OUTPUT_USER_REMINDER if force_chinese else ""
    full_user_content = f"{prefix_ctx}## 分析任务要求\n{task_prompt}{lang_tail}"
    final_sys_prompt = _ensure_chinese_system_prompt(system_prompt, force_chinese=force_chinese)

    messages = [{"role": "system", "content": final_sys_prompt}] + history_msgs + [
        {"role": "user", "content": full_user_content},
    ]

    print("\n" + "=" * 72)
    print(
        f"🚀 [模式: PACK 源码打包分析] 打包文件: {len(packed_files)} | "
        f"延续历史: {keep_history} ({len(history_msgs) // 2} 轮) | 引用MD: {len(context_mds or [])} | 中文输出: {force_chinese}"
    )
    print("=" * 72)

    _, clean_md, _, _, _ = _run_qwen_inprocess_once(
        messages=messages,
        tools=None,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
        max_new_tokens=max_new_tokens,
    )

    globals()["CHAT_HISTORY"].append({"role": "user", "content": task_prompt})
    globals()["CHAT_HISTORY"].append({"role": "assistant", "content": clean_md})

    save_and_sync_md(
        clean_markdown=clean_md,
        output_md_name=explicit_name,
        repo_name=repo_name,
        drive_out_dir=drive_out_dir,
        prompt_for_auto_name=task_prompt,
    )
    return clean_md


def explore_codebase_agent(
    task_prompt: str,
    output_md_name: str = "",
    local_repo_dir: str | Path = None,
    max_turns: int = 15,
    keep_history: bool = False,
    context_mds: list[str] = None,
    force_chinese: bool = True,
    system_prompt: str = (
        "你是一位具备自主代码库检索能力的资深软件架构师。你可以使用提供的只读工具（list_dir, find_files, grep_search, read_file）"
        "在本地高速代码库中自主搜索和阅读关键代码。\n"
        "工作原则：\n"
        "1. 先通过目录树、find_files 或 grep_search 定位核心模块与关键类/函数定义，再用 read_file 按需精读关键代码段。\n"
        "2. 收集到足够证据后，停止调用工具，直接输出一份使用简体中文撰写的、结构完整、逻辑严密、可直接保存为 Markdown 文档的深度技术报告。"
    ),
    enable_thinking: bool = True,
    reasoning_effort: str = DEFAULT_CODE_REASONING_EFFORT,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    overwrite: bool = True,
    drive_out_dir: str = DRIVE_MD_OUTPUT_DIR,
) -> str:
    """
    【模式 B：内置本地工具循环（Agent 自主检索阅读代码库）】
    为 Qwen3.8-27B 挂载本地高速盘只读工具 (list_dir / find_files / grep_search / read_file)，
    多轮自主查阅源码后，仅将最终汇总的纯净 Markdown 正文保存并同步至 Google Drive。
    """
    target_repo = local_repo_dir or globals().get("LOCAL_REPO_PATH")
    if not target_repo:
        raise RuntimeError("❌ 未指定 local_repo_dir 且全局 LOCAL_REPO_PATH 未初始化，请先运行 sync_drive_repo_to_local！")
    repo_root = Path(target_repo).resolve()
    repo_name = repo_root.name

    explicit_name = (output_md_name or "").strip()
    if explicit_name and not overwrite:
        _, check_drive_path = resolve_output_paths(
            output_md_name=explicit_name,
            repo_name=repo_name,
            drive_out_dir=drive_out_dir,
        )
        if check_drive_path.exists() and check_drive_path.stat().st_size > 0:
            print(f"⏭️ [断点跳过] 目标报告已存在于 Google Drive: {check_drive_path}")
            return check_drive_path.read_text(encoding="utf-8")

    if not keep_history:
        globals()["CHAT_HISTORY"] = []
    history_msgs = list(globals().get("CHAT_HISTORY", []))

    ctx_md_text = load_context_mds_content(context_mds, repo_name=repo_name, drive_out_dir=drive_out_dir)
    initial_tree = build_repo_tree(repo_root, max_depth=3, max_entries=150)
    lang_tail = CHINESE_OUTPUT_USER_REMINDER if force_chinese else ""

    user_parts = [
        f"当前代码库名称：`{repo_name}`\n代码库顶层目录树概览：\n```text\n{initial_tree}\n```"
    ]
    if ctx_md_text:
        user_parts.append(ctx_md_text)
    user_parts.append(
        f"## 你的分析任务\n{task_prompt}\n\n"
        f"请先使用工具在代码库中检索并阅读相关源码，完成研究后直接输出完整的简体中文 Markdown 报告正文。{lang_tail}"
    )
    first_user_msg = "\n\n".join(user_parts)
    final_sys_prompt = _ensure_chinese_system_prompt(system_prompt, force_chinese=force_chinese)

    messages = [{"role": "system", "content": final_sys_prompt}] + history_msgs + [
        {"role": "user", "content": first_user_msg},
    ]

    print("\n" + "=" * 72)
    print(
        f"🤖 [模式: AGENT 自主探索代码库] 最大轮数: {max_turns} | "
        f"延续历史: {keep_history} ({len(history_msgs) // 2} 轮) | 引用MD: {len(context_mds or [])} | 中文输出: {force_chinese}"
    )
    print("=" * 72)

    final_clean_md = ""
    for turn in range(1, max_turns + 1):
        print(f"\n🔄 --- [Agent 第 {turn}/{max_turns} 轮] ---")
        active_tools = CODEBASE_AGENT_TOOLS if turn < max_turns else None
        if turn == max_turns:
            messages.append({
                "role": "user",
                "content": (
                    "工具探索轮数已达上限，请立即根据目前已掌握的所有代码信息，"
                    "全程使用**简体中文**直接输出最终的完整 Markdown 分析报告（仅代码标识符与路径保留原英文）。"
                ),
            })

        reasoning_content, clean_md, tool_calls, _, _ = _run_qwen_inprocess_once(
            messages=messages,
            tools=active_tools,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            max_new_tokens=max_new_tokens,
        )

        if not tool_calls:
            final_clean_md = clean_md
            break

        assistant_msg = {
            "role": "assistant",
            "content": clean_md if clean_md else "",
            "tool_calls": tool_calls,
        }
        if reasoning_content:
            assistant_msg["reasoning_content"] = reasoning_content
        messages.append(assistant_msg)

        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            fn_args = tc["function"]["arguments"]
            tc_id = tc["id"]
            print(f"🛠️ [执行本地高速工具] {fn_name}({fn_args})")
            tool_output = _execute_local_code_tool(repo_root, fn_name, fn_args)
            if force_chinese:
                tool_output += "\n\n[系统提示：若已完成源码取证准备输出报告，请务必使用简体中文撰写最终 Markdown 报告正文。]"
            messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "name": fn_name,
                "content": tool_output,
            })

    if not final_clean_md.strip():
        raise RuntimeError("❌ 任务未能生成有效 Markdown 正文。")

    globals()["CHAT_HISTORY"].append({"role": "user", "content": task_prompt})
    globals()["CHAT_HISTORY"].append({"role": "assistant", "content": final_clean_md})

    save_and_sync_md(
        clean_markdown=final_clean_md,
        output_md_name=explicit_name,
        repo_name=repo_name,
        drive_out_dir=drive_out_dir,
        prompt_for_auto_name=task_prompt,
    )
    return final_clean_md


def chat_with_codebase(
    task_prompt: str,
    output_md_name: str = "",
    local_repo_dir: str | Path = None,
    keep_history: bool = True,
    context_mds: list[str] = None,
    files: list[str] = None,
    include_tree: bool = False,
    force_chinese: bool = True,
    system_prompt: str = (
        "你是一位资深软件架构师与技术专家。请结合已有的对话上下文或参考文档，使用简体中文深入、准确地回答用户问题，"
        "并以结构清晰的 Markdown 格式输出。"
    ),
    enable_thinking: bool = True,
    reasoning_effort: str = DEFAULT_CODE_REASONING_EFFORT,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    overwrite: bool = True,
    drive_out_dir: str = DRIVE_MD_OUTPUT_DIR,
) -> str:
    """
    【模式 C：纯对话 / 多轮追问 / 基于历史 MD 讨论模式】
    不强制重传整个目录树，适合基于上一轮对话 (keep_history=True) 或指定的历史 MD (context_mds) 进行追问和方案设计。
    """
    return analyze_code_pack(
        task_prompt=task_prompt,
        output_md_name=output_md_name,
        local_repo_dir=local_repo_dir,
        include_patterns=files,
        include_tree=include_tree,
        keep_history=keep_history,
        context_mds=context_mds,
        force_chinese=force_chinese,
        system_prompt=system_prompt,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
        max_new_tokens=max_new_tokens,
        overwrite=overwrite,
        drive_out_dir=drive_out_dir,
    )


def ask_qwen(
    prompt: str,
    mode: str = "agent",
    output_md_name: str = "",
    files: list[str] = None,
    exclude_files: list[str] = None,
    include_tree: bool = True,
    keep_history: bool = False,
    context_mds: list[str] = None,
    force_chinese: bool = True,
    reasoning_effort: str = DEFAULT_CODE_REASONING_EFFORT,
    enable_thinking: bool = True,
    max_turns: int = 15,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    overwrite: bool = True,
    local_repo_dir: str | Path = None,
    drive_out_dir: str = DRIVE_MD_OUTPUT_DIR,
) -> str:
    """
    【Cell 7 统一调用入口】
    - mode="agent": 模型使用本地高速工具 (list_dir/find_files/grep_search/read_file) 自主检索阅读代码库
    - mode="pack" : 将 files 通配符匹配的源码文件（及可选目录树）一次性打包投喂
    - mode="chat" : 纯对话/方案追问模式（默认不重复附带目录树，适合配合 keep_history=True 或 context_mds 追问）
    每次生成的最终正文（剔除 <think>）均保存为独立 .md 并实时同步至 Google Drive。
    """
    m = (mode or "agent").strip().lower()
    if m == "agent":
        return explore_codebase_agent(
            task_prompt=prompt,
            output_md_name=output_md_name,
            local_repo_dir=local_repo_dir,
            max_turns=max_turns,
            keep_history=keep_history,
            context_mds=context_mds,
            force_chinese=force_chinese,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            max_new_tokens=max_new_tokens,
            overwrite=overwrite,
            drive_out_dir=drive_out_dir,
        )
    elif m == "pack":
        return analyze_code_pack(
            task_prompt=prompt,
            output_md_name=output_md_name,
            local_repo_dir=local_repo_dir,
            include_patterns=files,
            exclude_patterns=exclude_files,
            include_tree=include_tree,
            keep_history=keep_history,
            context_mds=context_mds,
            force_chinese=force_chinese,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            max_new_tokens=max_new_tokens,
            overwrite=overwrite,
            drive_out_dir=drive_out_dir,
        )
    elif m == "chat":
        return chat_with_codebase(
            task_prompt=prompt,
            output_md_name=output_md_name,
            local_repo_dir=local_repo_dir,
            keep_history=keep_history,
            context_mds=context_mds,
            files=files,
            include_tree=False,
            force_chinese=force_chinese,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            max_new_tokens=max_new_tokens,
            overwrite=overwrite,
            drive_out_dir=drive_out_dir,
        )
    else:
        raise ValueError(f"❌ 不支持的 mode='{mode}'，可选值为: 'agent' | 'pack' | 'chat'")


# ==============================================================================
# 7. 初始化同步：运行本 Cell 时自动将 Google Drive 代码库导入 Colab 高速盘
# ==============================================================================
AUTO_SYNC_ON_LOAD = True

if AUTO_SYNC_ON_LOAD:
    LOCAL_REPO_PATH = sync_drive_repo_to_local(
        drive_source_path=DRIVE_REPO_PATH,
        local_workspace_root=LOCAL_WORKSPACE_ROOT,
        force_clean=False,  # False: 增量秒级同步；True: 清空重拷
    )
    print("\n" + "=" * 72)
    print("🎉 Cell 6 核心引擎与本地高速代码库已就绪！")
    print(f"📂 本地高速代码库路径: {LOCAL_REPO_PATH}")
    print(f"☁️ 云盘 MD 自动同步目录: {DRIVE_MD_OUTPUT_DIR}/{LOCAL_REPO_PATH.name}")
    print("👉 现在请直接在【Cell 7】中修改 PROMPT 并反复运行以与模型对话！")
    print("=" * 72)
