import os
from typing import TypedDict, List, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from .tools import calculator, code_assistant
from .rag import search_docs
from .memory import short_term_memory

# 智谱 API 配置
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
llm = ChatOpenAI(
    model="glm-4",                       # 或 glm-4-plus, glm-4-air
    openai_api_key=ZHIPU_API_KEY,
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0
)

tools = [calculator, code_assistant, search_docs]
tools_dict = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: List[Any]

def call_model(state: AgentState):
    messages = state["messages"]
    # 注入短期记忆（对话历史）
    if short_term_memory.chat_memory.messages:
        history = short_term_memory.chat_memory.messages[-6:]  # 最近6条
        messages = history + messages
    response = llm_with_tools.invoke(messages)
    # 将模型响应存入短期记忆
    short_term_memory.chat_memory.add_message(response)
    return {"messages": [response]}

def call_tool(state: AgentState):
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
    for msg in new_messages:
        short_term_memory.chat_memory.add_message(msg)
    return {"messages": new_messages}

def should_continue(state: AgentState):
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