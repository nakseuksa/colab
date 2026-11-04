# ==============================================================================
# Cell 7: Qwen3.8-27B 交互式代码库对话执行入口（每次修改下方参数并运行本 Cell 即可）
# ==============================================================================
# 前置条件：请确保已运行过一次 Cell 6（完成云盘代码库导入至 LOCAL_REPO_PATH 及引擎注册）

if "ask_qwen" not in globals():
    raise RuntimeError("❌ 未检测到 Cell 6 核心引擎函数 ask_qwen，请先运行一次 Cell 6！")

# ------------------------------------------------------------------------------
# 1. 选择对话/分析模式 (MODE):
#    - "agent": 模型使用本地高速工具 (list_dir / find_files / grep_search / read_file) 自主检索阅读代码库
#    - "pack" : 手动在下方 FILES 列表中指定要打包投喂的源码文件或 Glob 通配符
#    - "chat" : 纯对话/方案追问模式（不重复注入完整目录树，适合配合 KEEP_HISTORY=True 或 CONTEXT_MDS 追问）
# ------------------------------------------------------------------------------
MODE = "agent"

# ------------------------------------------------------------------------------
# 2. 上下文衔接与多轮追问配置:
#    - KEEP_HISTORY = True : 保留并承接上一轮（或多轮）的对话记忆，可直接改 PROMPT 追问细节
#    - KEEP_HISTORY = False: 清空历史对话记忆，开启全新的独立分析话题
#    - CONTEXT_MDS         : 可填入之前已生成的 .md 报告文件名作为前置参考背景，例如 ["01_全局架构.md"]
# ------------------------------------------------------------------------------
KEEP_HISTORY = False
CONTEXT_MDS = [
    # "01_全局架构.md",
]

# ------------------------------------------------------------------------------
# 3. 源码打包配置 (在 MODE = "pack" 时生效，或在 MODE = "chat" 时附加指定源码):
# ------------------------------------------------------------------------------
FILES = [
    # "src/core/**/*.py",
    # "src/api/*.py",
    # "README.md",
]
EXCLUDE_FILES = []
INCLUDE_TREE = True           # MODE = "pack" 时是否在头部自动附带代码库目录树

# ------------------------------------------------------------------------------
# 4. 输出 Markdown 命名与模型推理参数:
#    - OUTPUT_MD_NAME = ""  : 留空则自动按 "序号_时分秒_Prompt摘要.md" 生成独立文件，绝不覆盖历史记录
#    - OUTPUT_MD_NAME = "xx": 手动指定文件名（如 "01_项目架构总览.md"），配合 OVERWRITE 控制是否覆盖
# ------------------------------------------------------------------------------
OUTPUT_MD_NAME = ""
OVERWRITE = True              # 手动指定 OUTPUT_MD_NAME 时：True 覆盖更新同名文件 | False 已存在则直接跳过
FORCE_CHINESE = True          # 强制模型全程使用简体中文输出报告（仅代码标识符/路径保留原英文）
ENABLE_THINKING = True        # 是否开启 Qwen3.8 思考模式
REASONING_EFFORT = "medium"   # 思考强度档位: "low" | "medium" | "xhigh"
MAX_TURNS = 15                # MODE = "agent" 时的最大工具探索轮数
MAX_NEW_TOKENS = 16384        # 单轮最大生成 Token 数

# ------------------------------------------------------------------------------
# 5. 编写本轮与模型对话的 Prompt（每次修改此处文本后直接点击运行 Cell 即可）:
# ------------------------------------------------------------------------------
PROMPT = """
请全面探索当前代码库，撰写一份《项目全局架构与核心模块解析》文档，包含：
1. 项目核心定位、技术栈与整体目录分层结构
2. 核心模块/包之间的依赖关系（使用 Mermaid 流程图展示）
3. 关键启动入口、主流程控制链路与核心数据结构说明
""".strip()

# ==============================================================================
# 🚀 执行对话并将纯净 Markdown 正文（剥离 <think>）保存及同步至 Google Drive
# ==============================================================================
latest_markdown_report = ask_qwen(
    prompt=PROMPT,
    mode=MODE,
    output_md_name=OUTPUT_MD_NAME,
    files=FILES,
    exclude_files=EXCLUDE_FILES,
    include_tree=INCLUDE_TREE,
    keep_history=KEEP_HISTORY,
    context_mds=CONTEXT_MDS,
    force_chinese=FORCE_CHINESE,
    reasoning_effort=REASONING_EFFORT,
    enable_thinking=ENABLE_THINKING,
    max_turns=MAX_TURNS,
    max_new_tokens=MAX_NEW_TOKENS,
    overwrite=OVERWRITE,
)
