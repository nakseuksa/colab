# Cell 6 (H100 vLLM 本地高速代码库深度理解工作台 - Qwen3.8-27B OpenAI/SSE 版):
# 1. 高速导入：将指定 Google Drive 代码库文件夹智能过滤（剔除 .git / node_modules 等）并多线程同步至 Colab 本地 NVMe 高速盘 (/content)
# 2. 双通道流式推理：通过本地回环 HTTP SSE 直连 Cell 4 网关 (8000) 或自动回退至 Cell 2 vLLM 后端 (8001)
#    - 充分释放 H100 vLLM 原生 Automatic Prefix Caching、MTP 投机解码与真正并行并发能力
#    - 模式 A [analyze_code_pack]: 自动生成目录树 + 按模块/Glob 通配符打包源码投喂
#    - 模式 B [explore_codebase_agent]: 内置本地只读工具循环 (list_dir / find_files / grep_search / read_file)，Prefix Caching 加速多轮自主检索
#    - 模式 C [chat_with_codebase]: 纯对话 / 历史 MD 上下文追问
# 3. 纯净 MD 实时同步：双通道解析 (reasoning_content / delta.tool_calls + <think>/<tool_call> 兜底剥离)，仅将纯净正文写入 .md 并容错同步至 Google Drive

import os
import re
import time
import uuid
import json
import fnmatch
import shutil
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

# ==============================================================================
# 0. 核心路径与全局默认配置（请根据实际 Google Drive 目录修改此处）
# ==============================================================================
DEFAULT_DRIVE_REPO_PATH = "/content/drive/MyDrive/my_project"

# Google Drive 上的源代码库文件夹路径（支持普通代码目录，也兼容直接填入 .zip / .tar.gz 压缩包路径）
DRIVE_REPO_PATH = DEFAULT_DRIVE_REPO_PATH

# 导入到 Colab 本地 NVMe 高速磁盘的目标路径
LOCAL_WORKSPACE_ROOT = "/content/local_codebase"

# 生成的 Markdown 报告在 Google Drive 上的持久化同步目录
DRIVE_MD_OUTPUT_DIR = "/content/drive/MyDrive/qwen_code_reports"

# 生成的 Markdown 报告在 Colab 本地高速盘的缓存目录
LOCAL_MD_OUTPUT_DIR = "/content/local_outputs"

# 手动覆写本地 API Base URL（留空则自动探测：优先 Cell 4 网关 8000，其次 Cell 2 vLLM 8001）
LOCAL_API_BASE_URL = ""

# 跨 Cell 约定默认端点与模型名（Cell 2 / Cell 4 启动后会在 globals 中注入同名变量）
DEFAULT_GATEWAY_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_VLLM_BASE_URL = "http://127.0.0.1:8001/v1"
DEFAULT_SERVED_MODEL_NAME = "Qwen/Qwen3.8-27B"

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
# 1. 保活时间戳、端点自动探测与兼容解析层
# ==============================================================================
if "LAST_API_REQUEST_TIME" not in globals():
    globals()["LAST_API_REQUEST_TIME"] = time.time()


def _touch_keepalive():
    """刷新全局活跃时间戳，防止 Cell 5 的保活守护误判闲置"""
    globals()["LAST_API_REQUEST_TIME"] = time.time()


def resolve_served_model_name(model_name: str = None) -> str:
    """获取当前后端暴露的统一模型名称（默认 Qwen/Qwen3.8-27B）"""
    if model_name and str(model_name).strip():
        return str(model_name).strip()
    global_name = globals().get("SERVED_MODEL_NAME")
    if global_name and str(global_name).strip():
        return str(global_name).strip()
    return DEFAULT_SERVED_MODEL_NAME


def probe_openai_endpoint(
    base_url: str,
    timeout: float = 1.5,
    http_client: httpx.Client = None,
) -> bool:
    """探测本地 OpenAI 兼容端点 (/models 或 /health) 是否处于监听状态"""
    clean_base = (base_url or "").strip().rstrip("/")
    if not clean_base:
        return False

    owns_client = http_client is None
    client = http_client or httpx.Client(timeout=timeout)
    try:
        resp = client.get(f"{clean_base}/models")
        if resp.status_code < 500:
            return True
        if clean_base.endswith("/v1"):
            root_base = clean_base[:-3]
            h_resp = client.get(f"{root_base}/health")
            return h_resp.status_code < 500
        return False
    except (httpx.HTTPError, OSError):
        return False
    finally:
        if owns_client:
            client.close()


def resolve_api_base_url(
    api_base_url: str = None,
    http_client: httpx.Client = None,
    probe_timeout: float = 1.5,
) -> str:
    """
    自动探测可用的本地推理端点：
    1. 若显式传入 api_base_url 参数，直接使用（便于单测或定向调用）；
    2. 若顶部 LOCAL_API_BASE_URL 非空，探测并使用该手动覆写端点；
    3. 否则优先探测 Cell 4 网关 GATEWAY_BASE_URL (8000)，未启动则无缝回退至 Cell 2 vLLM VLLM_BASE_URL (8001)；
    4. 若均未监听，抛出清晰友好的 RuntimeError。
    """
    if api_base_url and str(api_base_url).strip():
        return str(api_base_url).strip().rstrip("/")

    manual_override = (globals().get("LOCAL_API_BASE_URL") or "").strip().rstrip("/")
    if manual_override:
        if probe_openai_endpoint(manual_override, timeout=probe_timeout, http_client=http_client):
            return manual_override
        raise RuntimeError(
            f"❌ 手动配置的 LOCAL_API_BASE_URL ({manual_override}) 无法连接！\n"
            f"请检查对应端口服务是否已启动，或清空 LOCAL_API_BASE_URL 以启用自动探测。"
        )

    gateway_url = (globals().get("GATEWAY_BASE_URL") or DEFAULT_GATEWAY_BASE_URL).strip().rstrip("/")
    vllm_url = (globals().get("VLLM_BASE_URL") or DEFAULT_VLLM_BASE_URL).strip().rstrip("/")

    if probe_openai_endpoint(gateway_url, timeout=probe_timeout, http_client=http_client):
        return gateway_url
    if vllm_url != gateway_url and probe_openai_endpoint(vllm_url, timeout=probe_timeout, http_client=http_client):
        return vllm_url

    raise RuntimeError(
        "❌ 未检测到运行中的 H100 本地推理服务！\n"
        f"   ├─ 已探测 Cell 4 网关: {gateway_url} (未监听)\n"
        f"   └─ 已探测 Cell 2 vLLM: {vllm_url} (未监听)\n"
        "👉 请先运行 Cell 2 启动 vLLM 后端服务（推荐同时运行 Cell 4 开启高并发网关）！"
    )


_TAG_THINK_OPEN = "<" + "think>"
_TAG_THINK_CLOSE = "</" + "think>"
_TAG_TC_OPEN = "<tool_" + "call>"
_TAG_TC_CLOSE = "</tool_" + "call>"
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
    """从 XML (<function=...><parameter=...>) 或 JSON 片段中兜底提取工具调用"""
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
            if isinstance(fn_name, dict):
                fn_args = fn_name.get("arguments", fn_name.get("parameters", {}))
                fn_name = fn_name.get("name", "")
            else:
                fn_args = tc_obj.get("arguments", tc_obj.get("parameters", {}))
            if not fn_name:
                return None
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


def parse_qwen_clean_output(raw_text: str, prompt_opened_think: bool = False):
    """
    兜底解析模型文本输出，剥离 <think>...</think> 思考过程与 <tool_call>...</tool_call> 标签，
    返回 (reasoning_content, clean_markdown, tool_calls)。
    """
    text = raw_text or ""
    for eos_tok in ("<|im_end|>", "<|endoftext|>"):
        text = text.replace(eos_tok, "")

    reasoning_parts = []
    body_text = text

    # 1. 剥离完整的 <think>...</think> 块（支持多个）
    think_matches = list(_REGEX_THINK_BLOCK.finditer(body_text))
    if think_matches:
        for m in think_matches:
            inner = m.group(1).strip()
            if inner:
                reasoning_parts.append(inner)
        body_text = _REGEX_THINK_BLOCK.sub("", body_text)

    # 2. 处理仅含关闭标签 </think> 的情况（模板预置了 <think>）
    if _TAG_THINK_CLOSE in body_text:
        think_part, rest_part = body_text.split(_TAG_THINK_CLOSE, 1)
        think_part = _REGEX_THINK_LEADING.sub("", think_part).strip()
        if think_part:
            reasoning_parts.append(think_part)
        body_text = rest_part
    # 3. 处理未闭合的 <think>...（例如思考阶段触发 max_tokens 截断）
    elif _TAG_THINK_OPEN in body_text:
        before_think, unclosed_think = body_text.split(_TAG_THINK_OPEN, 1)
        if unclosed_think.strip():
            reasoning_parts.append(unclosed_think.strip())
        body_text = before_think
    elif prompt_opened_think and not think_matches and body_text.strip():
        reasoning_parts.append(_REGEX_THINK_LEADING.sub("", body_text).strip())
        body_text = ""

    # 4. 提取兜底 <tool_call>...</tool_call> 并从正文中彻底剥离
    tool_calls = []
    for idx, match in enumerate(_REGEX_TC_BLOCK.finditer(body_text)):
        parsed_tc = _extract_tool_call_fallback(match.group(1), idx)
        if parsed_tc:
            tool_calls.append(parsed_tc)
    body_text = _REGEX_TC_BLOCK.sub("", body_text)

    if _TAG_TC_OPEN in body_text:
        before_tc, unclosed_part = body_text.split(_TAG_TC_OPEN, 1)
        parsed_tc = _extract_tool_call_fallback(unclosed_part, len(tool_calls))
        if parsed_tc:
            tool_calls.append(parsed_tc)
        body_text = before_tc

    reasoning_content = "\n".join(p for p in reasoning_parts if p).strip() or None
    clean_content = body_text.strip()
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
def _is_within_root(target: Path, root: Path) -> bool:
    """安全校验路径是否位于代码库根目录内，避免路径穿越或前缀误判"""
    try:
        return target.is_relative_to(root)
    except AttributeError:
        try:
            target.relative_to(root)
            return True
        except ValueError:
            return False


def build_repo_tree(
    repo_dir: Path,
    rel_subpath: str = ".",
    max_depth: int = 4,
    max_entries: int = 300,
) -> str:
    """生成代码库的清晰 ASCII 目录树（供 Prompt 上下文或 list_dir 工具使用）"""
    repo_root = Path(repo_dir).resolve()
    base = (repo_root / rel_subpath.lstrip("/\\")).resolve() if rel_subpath != "." else repo_root
    if not _is_within_root(base, repo_root) or not base.exists():
        return f"❌ 路径不存在或越界: {rel_subpath}"

    lines = [f"📁 {base.relative_to(repo_root).as_posix() if base != repo_root else repo_root.name}/"]
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
    search_base = (repo_root / subpath.lstrip("/\\")).resolve() if subpath != "." else repo_root
    if not _is_within_root(search_base, repo_root) or not search_base.exists():
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
    if not _is_within_root(target, repo_root) or not target.exists() or not target.is_file():
        return f"❌ 文件不存在或路径非法: {file_path}"

    try:
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as e:
        return f"❌ 读取文件失败 ({file_path}): {e}"

    total_lines = len(lines)
    if total_lines == 0:
        rel_str = target.relative_to(repo_root).as_posix()
        return f"📄 文件: {rel_str} (空文件，共 0 行)"

    s_idx = max(1, int(start_line))
    e_idx = min(total_lines, max(s_idx, int(end_line)))
    selected = [f"{i:4d} | {lines[i - 1]}" for i in range(s_idx, e_idx + 1)]
    rel_str = target.relative_to(repo_root).as_posix()
    header = f"📄 文件: {rel_str} (显示第 {s_idx}~{e_idx} 行 / 共 {total_lines} 行)\n"
    return header + "```" + target.suffix.lstrip(".") + "\n" + "\n".join(selected) + "\n```"


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
# 4. 模块三：Markdown 智能命名、纯净正文落盘与 Google Drive 容错同步
# ==============================================================================
if "CHAT_HISTORY" not in globals():
    globals()["CHAT_HISTORY"] = []

if "MD_AUTO_SEQ" not in globals():
    globals()["MD_AUTO_SEQ"] = 1


def reset_chat_history():
    """手动清空多轮对话历史记忆"""
    globals()["CHAT_HISTORY"] = []
    print("🧹 已清空多轮对话历史 (CHAT_HISTORY)。")


def _is_drive_target_writable(drive_out_dir: str | Path) -> bool:
    """
    判断 Google Drive 输出目录是否可用：
    - 若目标路径位于 /content/drive/... 下，必须确认 /content/drive 已挂载（避免未挂载时建目录报错或写入容器假目录）
    - 若为自定义本地路径（如单元测试临时目录），直接允许尝试创建
    """
    raw_posix = Path(drive_out_dir).as_posix()
    if raw_posix == "/content/drive" or raw_posix.startswith("/content/drive/"):
        return os.path.ismount("/content/drive") or Path("/content/drive/MyDrive").exists()
    return True


def _build_auto_md_name(
    prompt: str,
    repo_name: str = None,
    drive_out_dir: str = DRIVE_MD_OUTPUT_DIR,
    local_out_dir: str = LOCAL_MD_OUTPUT_DIR,
) -> str:
    """当 OUTPUT_MD_NAME 留空时，按 '序号_时分秒_Prompt摘要.md' 自动生成不冲突的文件名"""
    sub_folder = repo_name.strip() if repo_name else ""
    drive_base = Path(drive_out_dir) / sub_folder if sub_folder else Path(drive_out_dir)
    local_base = Path(local_out_dir) / sub_folder if sub_folder else Path(local_out_dir)

    drive_count = len(list(drive_base.glob("*.md"))) if (_is_drive_target_writable(drive_out_dir) and drive_base.exists()) else 0
    local_count = len(list(local_base.glob("*.md"))) if local_base.exists() else 0
    seq = max(int(globals().get("MD_AUTO_SEQ", 1)), drive_count + 1, local_count + 1)
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
    """解析本地高速盘与 Google Drive 的目标 Markdown 路径（支持留空自动命名与云盘未挂载容错）"""
    clean_name = (output_md_name or "").strip()
    if not clean_name:
        clean_name = _build_auto_md_name(
            prompt_for_auto_name,
            repo_name=repo_name,
            drive_out_dir=drive_out_dir,
            local_out_dir=local_out_dir,
        )
    elif not clean_name.lower().endswith(".md"):
        clean_name += ".md"

    sub_folder = repo_name.strip() if repo_name else ""
    local_base = Path(local_out_dir) / sub_folder if sub_folder else Path(local_out_dir)
    drive_base = Path(drive_out_dir) / sub_folder if sub_folder else Path(drive_out_dir)

    local_base.mkdir(parents=True, exist_ok=True)
    if _is_drive_target_writable(drive_out_dir):
        try:
            drive_base.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
    return local_base / clean_name, drive_base / clean_name


def load_context_mds_content(
    context_mds: list[str],
    repo_name: str = None,
    local_out_dir: str = LOCAL_MD_OUTPUT_DIR,
    drive_out_dir: str = DRIVE_MD_OUTPUT_DIR,
) -> str:
    """加载已生成的 Markdown 报告内容作为本轮对话的背景知识（自动兼容云盘未挂载场景）"""
    if not context_mds:
        return ""

    sub_folder = repo_name.strip() if repo_name else ""
    search_dirs = [
        Path(local_out_dir) / sub_folder if sub_folder else Path(local_out_dir),
        Path(local_out_dir),
    ]
    if _is_drive_target_writable(drive_out_dir):
        search_dirs.extend([
            Path(drive_out_dir) / sub_folder if sub_folder else Path(drive_out_dir),
            Path(drive_out_dir),
        ])

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
    将清洗后的纯净 Markdown 正文（不含思考过程）写入本地高速盘；
    若 Google Drive 已挂载且可写，则同步至云盘，否则安全降级为仅本地落盘而不崩溃。
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
    local_size_kb = local_md_path.stat().st_size / 1024.0

    synced_to_drive = False
    drive_err = None
    if _is_drive_target_writable(drive_out_dir):
        try:
            drive_md_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_md_path, drive_md_path)
            try:
                os.sync()
            except AttributeError:
                pass
            synced_to_drive = True
        except OSError as e:
            drive_err = e

    if synced_to_drive:
        print("💾 [MD 已保存并同步至 Google Drive]")
        print(f"   ├─ Colab 高速盘: {local_md_path}")
        print(f"   └─ Google Drive: {drive_md_path} ({drive_md_path.stat().st_size / 1024.0:.1f} KB)")
    else:
        reason = f"({drive_err})" if drive_err else "(Google Drive 未挂载)"
        print(f"💾 [MD 已保存至本地] {reason}")
        print(f"   └─ 本地路径: {local_md_path} ({local_size_kb:.1f} KB)")
    return local_md_path, drive_md_path


# ==============================================================================
# 5. 模块四：H100 vLLM HTTP SSE 流式推理核心引擎
# ==============================================================================
def build_vllm_chat_payload(
    messages: list[dict],
    model_name: str = None,
    tools: list[dict] = None,
    enable_thinking: bool = True,
    reasoning_effort: str = DEFAULT_CODE_REASONING_EFFORT,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = None,
) -> dict:
    """构建符合 H100 vLLM & Cell 4 网关契约的 /chat/completions 流式请求体"""
    payload = {
        "model": resolve_served_model_name(model_name),
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": int(max_new_tokens),
    }

    if enable_thinking:
        temp = float(temperature) if temperature is not None else 1.0
        payload["temperature"] = temp
        if temp > 0:
            payload["top_p"] = 0.95
            payload["top_k"] = 20
        effort = (reasoning_effort or DEFAULT_CODE_REASONING_EFFORT).strip().lower()
        payload["reasoning_effort"] = effort
        payload["chat_template_kwargs"] = {
            "enable_thinking": True,
            "preserve_thinking": True,
            "reasoning_effort": effort,
        }
    else:
        temp = float(temperature) if temperature is not None else 0.7
        payload["temperature"] = temp
        if temp > 0:
            payload["top_p"] = 0.80
            payload["top_k"] = 20
            payload["repetition_penalty"] = 1.05
        # 关思考时不传递 reasoning_effort
        payload["chat_template_kwargs"] = {
            "enable_thinking": False,
            "preserve_thinking": True,
        }

    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    return payload


class VLLMStreamAccumulator:
    """
    双通道 SSE 流式聚合与控制台打印器：
    1. 原生结构化通道：delta.reasoning_content / delta.reasoning 与增量 delta.tool_calls (按 index 拼接)
    2. 兜底文本通道：当 delta.content 夹带 <think>...</think> 或 <tool_call>...</tool_call> 时自动剥离并提取
    3. 统计指标：TTFT、Prompt/Completion Tokens（优先取 usage，无 usage 时合理估算）
    """

    def __init__(self, stream_to_console: bool = True, start_perf: float = None):
        self.stream_to_console = stream_to_console
        self.start_perf = start_perf if start_perf is not None else time.perf_counter()
        self.first_token_perf = None

        self.native_reasoning_parts = []
        self.content_parts = []
        self.tool_calls_by_index = {}
        self.usage = None
        self.non_empty_token_chunks = 0

        self._display_state = "init"  # "init" | "thinking" | "content"
        self._in_content_think_tag = False
        self._tag_buf = ""

    def _mark_first_token(self):
        if self.first_token_perf is None:
            self.first_token_perf = time.perf_counter()

    def _print_console(self, text: str):
        if self.stream_to_console and text:
            print(text, end="", flush=True)

    def _enter_thinking_display(self):
        if self._display_state != "thinking":
            self._print_console("💭 [思考中...] ")
            self._display_state = "thinking"

    def _enter_content_display(self):
        if self._display_state == "thinking":
            self._print_console("\n" + "-" * 60 + "\n📝 [开始输出正文]\n")
            self._display_state = "content"
        elif self._display_state == "init":
            self._print_console("📝 [开始输出正文]\n")
            self._display_state = "content"

    def _emit_content_streaming(self, piece: str):
        """跨 chunk 检测 <think> / </think> 标签，保持控制台流式输出整洁"""
        clean = piece.replace("<|im_end|>", "").replace("<|endoftext|>", "")
        if not clean:
            return
        self._tag_buf += clean

        while self._tag_buf:
            idx_open = self._tag_buf.find(_TAG_THINK_OPEN)
            idx_close = self._tag_buf.find(_TAG_THINK_CLOSE)

            candidates = [(i, tag) for i, tag in ((idx_open, _TAG_THINK_OPEN), (idx_close, _TAG_THINK_CLOSE)) if i != -1]
            if candidates:
                pos, tag = min(candidates, key=lambda x: x[0])
                before = self._tag_buf[:pos]
                if before:
                    if not self._in_content_think_tag and self._display_state != "content" and before.strip():
                        self._enter_content_display()
                    self._print_console(before)
                if tag == _TAG_THINK_OPEN:
                    self._in_content_think_tag = True
                    self._enter_thinking_display()
                else:
                    self._in_content_think_tag = False
                    self._enter_content_display()
                self._tag_buf = self._tag_buf[pos + len(tag):]
                continue

            # 若末尾可能存在未接收完的 '<think>' 或 '</think>' 前缀，暂留到下一块
            last_lt = self._tag_buf.rfind("<")
            if last_lt != -1:
                tail = self._tag_buf[last_lt:]
                if len(tail) < len(_TAG_THINK_CLOSE) and (
                    _TAG_THINK_OPEN.startswith(tail) or _TAG_THINK_CLOSE.startswith(tail)
                ):
                    head = self._tag_buf[:last_lt]
                    if head:
                        if not self._in_content_think_tag and self._display_state != "content" and head.strip():
                            self._enter_content_display()
                        self._print_console(head)
                    self._tag_buf = tail
                    break

            if not self._in_content_think_tag and self._display_state != "content" and self._tag_buf.strip():
                self._enter_content_display()
            self._print_console(self._tag_buf)
            self._tag_buf = ""

    def feed_chunk(self, chunk: dict):
        """消费单个解析后的 OpenAI/vLLM SSE JSON chunk"""
        if not isinstance(chunk, dict):
            return

        if isinstance(chunk.get("usage"), dict):
            self.usage = chunk["usage"]

        choices = chunk.get("choices") or []
        if not choices:
            return
        delta = choices[0].get("delta") or {}
        if not isinstance(delta, dict):
            return

        # 1. 原生结构化推理通道 (reasoning_content 优先，其次 reasoning，绝不重复累计)
        r_piece = delta.get("reasoning_content")
        if not r_piece:
            r_piece = delta.get("reasoning")
        if r_piece:
            self._mark_first_token()
            self.non_empty_token_chunks += 1
            self.native_reasoning_parts.append(r_piece)
            self._enter_thinking_display()
            self._print_console(r_piece.replace("<|im_end|>", "").replace("<|endoftext|>", ""))

        # 2. 正文通道 (含可能的 <think>/<tool_call> 兜底标签)
        c_piece = delta.get("content")
        if c_piece:
            self._mark_first_token()
            self.non_empty_token_chunks += 1
            self.content_parts.append(c_piece)
            self._emit_content_streaming(c_piece)

        # 3. 原生结构化增量 tool_calls 通道 (按 index 拼接)
        tc_deltas = delta.get("tool_calls")
        if isinstance(tc_deltas, list) and tc_deltas:
            self._mark_first_token()
            self.non_empty_token_chunks += 1
            for tc in tc_deltas:
                if not isinstance(tc, dict):
                    continue
                idx = int(tc.get("index", 0))
                entry = self.tool_calls_by_index.setdefault(
                    idx,
                    {
                        "index": idx,
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    },
                )
                if tc.get("id"):
                    entry["id"] = tc["id"]
                if tc.get("type"):
                    entry["type"] = tc["type"]
                fn_obj = tc.get("function")
                if isinstance(fn_obj, dict):
                    if fn_obj.get("name"):
                        entry["function"]["name"] += fn_obj["name"]
                    if fn_obj.get("arguments") is not None:
                        entry["function"]["arguments"] += fn_obj["arguments"]

    def finalize(self, elapsed: float = None) -> dict:
        """结束流式聚合，返回完整结构化结果与指标"""
        if self._tag_buf:
            if not self._in_content_think_tag and self._display_state != "content" and self._tag_buf.strip():
                self._enter_content_display()
            self._print_console(self._tag_buf)
            self._tag_buf = ""

        total_elapsed = elapsed if elapsed is not None else (time.perf_counter() - self.start_perf)
        ttft = (self.first_token_perf - self.start_perf) if self.first_token_perf is not None else total_elapsed

        raw_content = "".join(self.content_parts)
        fb_reasoning, clean_content, fb_tool_calls = parse_qwen_clean_output(
            raw_content, prompt_opened_think=False
        )

        native_reasoning = "".join(self.native_reasoning_parts).strip()
        if native_reasoning and fb_reasoning and fb_reasoning not in native_reasoning:
            reasoning_content = f"{native_reasoning}\n{fb_reasoning}".strip()
        else:
            reasoning_content = native_reasoning or fb_reasoning

        native_tool_calls = []
        for idx in sorted(self.tool_calls_by_index.keys()):
            item = self.tool_calls_by_index[idx]
            native_tool_calls.append({
                "index": idx,
                "id": item["id"] or f"call_{uuid.uuid4().hex[:12]}",
                "type": item.get("type") or "function",
                "function": {
                    "name": item["function"]["name"],
                    "arguments": item["function"]["arguments"],
                },
            })

        tool_calls = native_tool_calls if native_tool_calls else fb_tool_calls

        prompt_tokens = None
        gen_tokens = None
        used_fallback_count = False
        if isinstance(self.usage, dict):
            if self.usage.get("prompt_tokens") is not None:
                prompt_tokens = int(self.usage["prompt_tokens"])
            if self.usage.get("completion_tokens") is not None:
                gen_tokens = int(self.usage["completion_tokens"])

        if gen_tokens is None or (gen_tokens == 0 and (reasoning_content or clean_content or tool_calls)):
            used_fallback_count = True
            total_chars = len(native_reasoning) + len(raw_content) + sum(
                len(tc["function"]["name"]) + len(tc["function"]["arguments"]) for tc in native_tool_calls
            )
            if total_chars == 0 and self.non_empty_token_chunks == 0:
                gen_tokens = 0
            else:
                gen_tokens = max(self.non_empty_token_chunks, total_chars // 3, 1)

        return {
            "reasoning_content": reasoning_content,
            "clean_content": clean_content,
            "tool_calls": tool_calls,
            "prompt_tokens": prompt_tokens,
            "gen_tokens": int(gen_tokens),
            "used_fallback_count": used_fallback_count,
            "ttft": ttft,
            "elapsed": total_elapsed,
        }


def _run_qwen_vllm_once(
    messages: list[dict],
    tools: list[dict] = None,
    enable_thinking: bool = True,
    reasoning_effort: str = DEFAULT_CODE_REASONING_EFFORT,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = None,
    stream_to_console: bool = True,
    api_base_url: str = None,
    model_name: str = None,
    http_client: httpx.Client = None,
) -> tuple[str | None, str, list[dict], int, float]:
    """
    通过本地高速回环 HTTP SSE 流式请求 H100 vLLM 后端（或 Cell 4 网关）执行单轮推理。
    返回: (reasoning_content, clean_markdown_content, tool_calls, generated_tokens, elapsed_seconds)
    """
    _touch_keepalive()
    base_url = resolve_api_base_url(api_base_url=api_base_url, http_client=http_client)
    payload = build_vllm_chat_payload(
        messages=messages,
        model_name=model_name,
        tools=tools,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    api_key = os.environ.get("OPENAI_API_KEY") or "sk-colab"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    endpoint_url = f"{base_url}/chat/completions"

    print(
        f"🧠 [H100 vLLM 推理启动] Endpoint={base_url} | Model={payload['model']} | "
        f"thinking={enable_thinking}({reasoning_effort if enable_thinking else 'off'}) | "
        f"max_tokens={max_new_tokens}"
    )

    t0 = time.perf_counter()
    acc = VLLMStreamAccumulator(stream_to_console=stream_to_console, start_perf=t0)

    owns_client = http_client is None
    client = http_client or httpx.Client(timeout=None)
    try:
        try:
            with client.stream("POST", endpoint_url, json=payload, headers=headers, timeout=None) as resp:
                if resp.status_code >= 400:
                    err_bytes = resp.read()
                    err_text = err_bytes.decode("utf-8", errors="replace")[:500]
                    raise RuntimeError(f"❌ H100 推理服务返回错误 HTTP {resp.status_code} ({endpoint_url}): {err_text}")

                for raw_line in resp.iter_lines():
                    _touch_keepalive()
                    if not raw_line:
                        continue
                    line = raw_line.strip()
                    if line.startswith(":"):
                        # 兼容 Cell 4 网关的 SSE ': keep-alive' 心跳
                        continue
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if not data_str:
                        continue
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk_obj = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    acc.feed_chunk(chunk_obj)
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"❌ 无法连接到本地推理服务 {endpoint_url} ({e})。\n"
                "请确认 Cell 2 (vLLM) 或 Cell 4 (Gateway) 正在运行！"
            ) from e
    finally:
        _touch_keepalive()
        if owns_client:
            client.close()

    elapsed = time.perf_counter() - t0
    res = acc.finalize(elapsed=elapsed)

    gen_tokens = res["gen_tokens"]
    speed = gen_tokens / max(res["elapsed"], 1e-5)
    prompt_info = f"{res['prompt_tokens']} tok" if res["prompt_tokens"] is not None else "N/A"
    est_suffix = " (估算)" if res["used_fallback_count"] else ""
    print(
        f"\n📊 [本轮推理完成] TTFT: {res['ttft']:.2f}s | Prompt: {prompt_info} | "
        f"生成: {gen_tokens} tok{est_suffix} | 总耗时: {res['elapsed']:.2f}s | 速率: {speed:.1f} tok/s"
    )
    return (
        res["reasoning_content"],
        res["clean_content"],
        res["tool_calls"],
        gen_tokens,
        res["elapsed"],
    )


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
    local_out_dir: str = LOCAL_MD_OUTPUT_DIR,
    api_base_url: str = None,
    http_client: httpx.Client = None,
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
        check_local_path, check_drive_path = resolve_output_paths(
            output_md_name=explicit_name,
            repo_name=repo_name,
            local_out_dir=local_out_dir,
            drive_out_dir=drive_out_dir,
        )
        if _is_drive_target_writable(drive_out_dir) and check_drive_path.exists() and check_drive_path.stat().st_size > 0:
            print(f"⏭️ [断点跳过] 目标报告已存在于 Google Drive: {check_drive_path}")
            return check_drive_path.read_text(encoding="utf-8")
        if check_local_path.exists() and check_local_path.stat().st_size > 0:
            print(f"⏭️ [断点跳过] 目标报告已存在于本地缓存: {check_local_path}")
            return check_local_path.read_text(encoding="utf-8")

    if not keep_history:
        globals()["CHAT_HISTORY"] = []
    history_msgs = list(globals().get("CHAT_HISTORY", []))

    context_sections = []
    ctx_md_text = load_context_mds_content(
        context_mds,
        repo_name=repo_name,
        local_out_dir=local_out_dir,
        drive_out_dir=drive_out_dir,
    )
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

    _, clean_md, _, _, _ = _run_qwen_vllm_once(
        messages=messages,
        tools=None,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
        max_new_tokens=max_new_tokens,
        api_base_url=api_base_url,
        http_client=http_client,
    )

    globals()["CHAT_HISTORY"].append({"role": "user", "content": task_prompt})
    globals()["CHAT_HISTORY"].append({"role": "assistant", "content": clean_md})

    save_and_sync_md(
        clean_markdown=clean_md,
        output_md_name=explicit_name,
        repo_name=repo_name,
        local_out_dir=local_out_dir,
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
    local_out_dir: str = LOCAL_MD_OUTPUT_DIR,
    api_base_url: str = None,
    http_client: httpx.Client = None,
) -> str:
    """
    【模式 B：内置本地工具循环（Agent 自主检索阅读代码库 + Prefix Caching 加速）】
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
        check_local_path, check_drive_path = resolve_output_paths(
            output_md_name=explicit_name,
            repo_name=repo_name,
            local_out_dir=local_out_dir,
            drive_out_dir=drive_out_dir,
        )
        if _is_drive_target_writable(drive_out_dir) and check_drive_path.exists() and check_drive_path.stat().st_size > 0:
            print(f"⏭️ [断点跳过] 目标报告已存在于 Google Drive: {check_drive_path}")
            return check_drive_path.read_text(encoding="utf-8")
        if check_local_path.exists() and check_local_path.stat().st_size > 0:
            print(f"⏭️ [断点跳过] 目标报告已存在于本地缓存: {check_local_path}")
            return check_local_path.read_text(encoding="utf-8")

    if not keep_history:
        globals()["CHAT_HISTORY"] = []
    history_msgs = list(globals().get("CHAT_HISTORY", []))

    ctx_md_text = load_context_mds_content(
        context_mds,
        repo_name=repo_name,
        local_out_dir=local_out_dir,
        drive_out_dir=drive_out_dir,
    )
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
        if turn == max_turns and turn > 1:
            messages.append({
                "role": "user",
                "content": (
                    "工具探索轮数已达上限，请立即根据目前已掌握的所有代码信息，"
                    "全程使用**简体中文**直接输出最终的完整 Markdown 分析报告（仅代码标识符与路径保留原英文）。"
                ),
            })

        reasoning_content, clean_md, tool_calls, _, _ = _run_qwen_vllm_once(
            messages=messages,
            tools=active_tools,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            max_new_tokens=max_new_tokens,
            api_base_url=api_base_url,
            http_client=http_client,
        )

        if not tool_calls:
            final_clean_md = clean_md
            break

        sanitized_tool_calls = [{k: v for k, v in tc.items() if k != "index"} for tc in tool_calls]
        assistant_msg = {
            "role": "assistant",
            "content": clean_md if clean_md else "",
            "tool_calls": sanitized_tool_calls,
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
        local_out_dir=local_out_dir,
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
    local_out_dir: str = LOCAL_MD_OUTPUT_DIR,
    api_base_url: str = None,
    http_client: httpx.Client = None,
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
        local_out_dir=local_out_dir,
        api_base_url=api_base_url,
        http_client=http_client,
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
    local_out_dir: str = LOCAL_MD_OUTPUT_DIR,
    api_base_url: str = None,
    http_client: httpx.Client = None,
) -> str:
    """
    【Cell 7 统一调用入口】
    - mode="agent": 模型使用本地高速工具 (list_dir/find_files/grep_search/read_file) 自主检索阅读代码库
    - mode="pack" : 将 files 通配符匹配的源码文件（及可选目录树）一次性打包投喂
    - mode="chat" : 纯对话/方案追问模式（默认不重复附带目录树，适合配合 keep_history=True 或 context_mds 追问）
    每次生成的最终正文（剔除思考过程）均保存为独立 .md 并容错同步至 Google Drive。
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
            local_out_dir=local_out_dir,
            api_base_url=api_base_url,
            http_client=http_client,
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
            local_out_dir=local_out_dir,
            api_base_url=api_base_url,
            http_client=http_client,
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
            local_out_dir=local_out_dir,
            api_base_url=api_base_url,
            http_client=http_client,
        )
    else:
        raise ValueError(f"❌ 不支持的 mode='{mode}'，可选值为: 'agent' | 'pack' | 'chat'")


# ==============================================================================
# 7. 初始化同步：仅在直接作为 Cell 运行时安全执行，确保模块 import 零副作用
# ==============================================================================
AUTO_SYNC_ON_LOAD = True

if "LOCAL_REPO_PATH" not in globals():
    LOCAL_REPO_PATH = None


def _auto_sync_on_cell_load():
    """在 Colab Cell 直接运行时执行安全自检与代码库同步；默认占位路径不存在时不中断函数注册"""
    if not AUTO_SYNC_ON_LOAD:
        return

    src_path = Path(DRIVE_REPO_PATH)
    if src_path.exists():
        synced_path = sync_drive_repo_to_local(
            drive_source_path=DRIVE_REPO_PATH,
            local_workspace_root=LOCAL_WORKSPACE_ROOT,
            force_clean=False,
        )
        globals()["LOCAL_REPO_PATH"] = synced_path
        print("\n" + "=" * 72)
        print("🎉 Cell 6 H100 vLLM 工作台引擎与本地高速代码库已就绪！")
        print(f"📂 本地高速代码库路径: {synced_path}")
        print(f"☁️ 云盘 MD 自动同步目录: {DRIVE_MD_OUTPUT_DIR}/{synced_path.name}")
        print("👉 现在请直接在【Cell 7】中修改 PROMPT 并反复运行以与模型对话！")
        print("=" * 72)
    elif DRIVE_REPO_PATH == DEFAULT_DRIVE_REPO_PATH:
        print("\n" + "=" * 72)
        print("ℹ️ [Cell 6 引擎函数已注册就绪]")
        print(f"检测到 DRIVE_REPO_PATH 仍为默认占位路径 ({DRIVE_REPO_PATH}) 且目录尚不存在，已跳过自动同步。")
        print("👉 请将顶部 DRIVE_REPO_PATH 修改为你的 Google Drive 代码库路径后重跑 Cell 6，")
        print("   或在 Cell 7 中传入 local_repo_dir='...' / 手动调用 sync_drive_repo_to_local('...')。")
        print("=" * 72)
    else:
        # 用户已修改为自定义路径但目录不存在，给出明确路径错误
        sync_drive_repo_to_local(
            drive_source_path=DRIVE_REPO_PATH,
            local_workspace_root=LOCAL_WORKSPACE_ROOT,
            force_clean=False,
        )


if __name__ == "__main__":
    _auto_sync_on_cell_load()
