import os
from langchain_zhipu import ZhipuAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import tool

# 从环境变量获取 API Key
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")

# 初始化智谱嵌入模型
embeddings = ZhipuAIEmbeddings(
    model="embedding-2",
    zhipuai_api_key=ZHIPU_API_KEY,
)

persist_dir = "./chroma_db"

def init_vectorstore():
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    
    documents = []
    docs_dir = "data/docs"
    if os.path.exists(docs_dir):
        for fname in os.listdir(docs_dir):
            if fname.endswith(".txt"):
                loader = TextLoader(os.path.join(docs_dir, fname))
                documents.extend(loader.load())
    
    if not documents:
        from langchain_core.documents import Document
        documents = [Document(page_content="这是公司内部知识库示例：储能电池的最佳工作温度是15-25°C。")]
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=persist_dir)
    vectorstore.persist()
    return vectorstore

def retrieve(query, k=3):
    vectorstore = init_vectorstore()
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in docs])

@tool
def search_docs(query: str) -> str:
    """从公司内部知识库检索相关信息。查询可以是问题或关键词。"""
    results = retrieve(query)
    if not results.strip():
        return "未找到相关信息。"
    return f"检索结果：\n{results}"