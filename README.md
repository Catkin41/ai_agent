# AI Agent Demo

基于 LangGraph + DeepSeek 的简单 AI Agent，支持工具调用（计算器、代码辅助）。

## 功能
- 计算器：计算数学表达式（如 `123*456`）
- 代码辅助：解释或生成代码（如 `用 Python 实现冒泡排序`）

## 运行
1. 安装依赖：`pip install -r requirements.txt`
2. 设置环境变量 `DEEPSEEK_API_KEY` 或在根目录创建 `.env` 文件
3. 运行：`streamlit run app.py`

## 技术栈
- LangChain / LangGraph
- DeepSeek API
- Streamlit