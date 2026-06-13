import math
from langchain.tools import tool

@tool
def calculator(expression: str) -> str:
    """计算数学表达式，例如 '2+3*4'"""
    try:
        # 安全评估
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

@tool
def code_assistant(query: str) -> str:
    """解释或生成简单代码片段（模拟）"""
    # 实际可以调用 LLM 进行更复杂的代码辅助
    if "解释" in query or "explain" in query.lower():
        return "我可以帮你解释 Python 代码。例如：`sorted([3,1,2])` 会返回 `[1,2,3]`。"
    else:
        return "我可以生成代码片段。需要什么语言的代码？请具体说明。"