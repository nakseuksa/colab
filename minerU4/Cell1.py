# 1. 挂载 Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. 安装 uv 及 MinerU 4.0 [full]（包含 Torch + vLLM 加速引擎）
!pip install -U uv
!uv pip install --system -U "mineru[full]>=4.0,<5"