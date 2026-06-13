import streamlit as st
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.tools import tool
import os
from dotenv import load_dotenv

load_dotenv()

# ==================== DeepSeek API 配置 ====================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    st.error("请设置环境变量 DEEPSEEK_API_KEY，或在项目根目录创建 .env 文件并写入 DEEPSEEK_API_KEY=你的密钥")
    st.stop()

llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base="https://api.deepseek.com/v1",
    temperature=0
)

# ==================== 工具定义 ====================
@tool
def calculator(expression: str) -> str:
    """计算数学表达式，例如 '2+3*4'"""
    try:
        import math
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

@tool
def code_assistant(query: str) -> str:
    """解释或生成简单代码片段"""
    return "我可以帮你解释或生成代码。请具体说明需求（例如：用 Python 实现冒泡排序）。"

tools = [calculator, code_assistant]
tools_dict = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

# ==================== LangGraph 工作流 ====================
class AgentState(dict):
    pass

def call_model(state):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def call_tool(state):
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls
    new_messages = []
    for tc in tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]
        tool_func = tools_dict.get(tool_name)
        if tool_func:
            result = tool_func.invoke(tool_args)
            tool_msg = ToolMessage(content=str(result), tool_call_id=tc["id"])
            new_messages.append(tool_msg)
        else:
            new_messages.append(ToolMessage(content=f"工具 {tool_name} 未找到", tool_call_id=tc["id"]))
    return {"messages": new_messages}

def should_continue(state):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", call_tool)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
workflow.add_edge("tools", "agent")
graph = workflow.compile()

# ==================== Streamlit 界面 ====================
st.set_page_config(page_title="AI Agent Demo (DeepSeek)", page_icon="🤖")
st.title("🤖 AI Agent 演示")
st.markdown("**后端模型**：DeepSeek · 支持计算器、代码辅助 · LangGraph 工作流")

# 显示 API 余额不足的友好提示（如果已经发生）
if "api_error_shown" not in st.session_state:
    st.session_state.api_error_shown = False

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("问我任何问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            final_state = graph.invoke({"messages": [HumanMessage(content=prompt)]})
            final_message = final_state["messages"][-1].content
            placeholder.markdown(final_message)
            st.session_state.messages.append({"role": "assistant", "content": final_message})
        except Exception as e:
            error_msg = str(e)
            if "402" in error_msg or "Insufficient Balance" in error_msg:
                friendly_error = "⚠️ DeepSeek API 余额不足，请充值后重试。"
                placeholder.error(friendly_error)
                st.session_state.messages.append({"role": "assistant", "content": friendly_error})
            else:
                placeholder.error(f"调用 DeepSeek API 失败：{error_msg}")
                st.session_state.messages.append({"role": "assistant", "content": f"错误：{error_msg}"})