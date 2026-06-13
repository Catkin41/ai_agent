from langchain.memory import ConversationBufferMemory

# 短期记忆（对话缓冲）
short_term_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
long_term_memory = None  # 暂不使用长期记忆，避免额外依赖