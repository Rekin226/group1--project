import os
from typing import TypedDict, List
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage, SystemMessage
from langgraph.graph import StateGraph, END

# ==========================================
# 1. Knowledge Base and Vector Search Setup (RAG Setup)
# ==========================================
def setup_vectorstore(db_path="./chroma_db", kb_path="./knowledge_base"):
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    
    # Load existing vector database if it exists
    if os.path.exists(db_path):
        print(f"Loading existing vector database from {db_path}")
        return Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    print("Initializing and building vector database...")
# 1. 修正標題切分層級：拔除 ####，強制保留整個疾病/異常狀況的完整上下文
    headers_to_split_on = [
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    documents = []
    if os.path.exists(kb_path):
        for filename in os.listdir(kb_path):
            if filename.endswith(".md"):
                with open(os.path.join(kb_path, filename), "r", encoding="utf-8") as f:
                    splits = markdown_splitter.split_text(f.read())
                    documents.extend(splits)
    
    # 2. 放寬字元限制，確保整個 ### 區塊能完整塞入單一 Chunk，但又不會大到包含兩個不同的疾病
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    print(f"Split into {len(chunks)} semantic chunks.")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_path)
    return vectorstore

# Initialize database and retriever
vectorstore = setup_vectorstore()
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10})

# Initialize LLM model
llm = ChatOllama(model="llama3", temperature=0)

# ==========================================
# 2. Define Graph State and Nodes (LangGraph Node)
# ==========================================
class AquaponicsState(TypedDict):
    messages: List[AnyMessage]
    user_query: str
    original_problem: str       # 新增：用於鎖定最初的症狀，避免檢索失憶
    mode: str
    follow_up_count: int

# 4. 定義路由分類節點 (Router Node)
def router_node(state: AquaponicsState):
    query = state.get("user_query", "")
    
    # 改寫為全英文的高精度二元分類提示詞
    system_prompt = """You are an Aquaponics query classifier.
Your ONLY task is to classify the user's input into one of two categories: 'simple' or 'complex'.
Output EXACTLY one word. Do NOT output any other text, punctuation, or formatting.

CLASSIFICATION RULES:
- Output 'simple' IF the user is asking a general knowledge, definitional, or factual question (e.g., "What is the ideal pH?", "Explain the nitrogen cycle", "How many stages are there?").
- Output 'complex' IF the user is reporting a system problem, asking for a diagnosis, troubleshooting a symptom, or asking "what is wrong?" (e.g., "My plants have yellow leaves", "My fish are dying", "The water is cloudy").
"""
    
    messages_to_llm = [SystemMessage(content=system_prompt)] + state.get("messages", [])
    response = llm.invoke(messages_to_llm)
    decision = response.content.strip().lower()
    
    # 強化防呆機制
    if "complex" in decision:
        mode = "complex"
    else:
        mode = "simple"
        
    print(f"\n[Internal] Router decided mode: {mode}")
    
    return {"mode": mode}

def simple_node(state: AquaponicsState):
    query = state.get("user_query", "")
    docs = retriever.invoke(query)
    # 將原本的 context = "\n\n".join([doc.page_content for doc in docs]) 刪除
        # 替換為以下帶有 Metadata 解析的寫法：
# 將原本的 context = "\n\n".join([doc.page_content for doc in docs]) 刪除
        # 替換為以下帶有 Metadata 解析的寫法：
    context_parts = []
    for doc in docs:
        # 提取並組合 metadata 中的標題 (如：Header 3 > Header 4)
        headers = " > ".join([str(v) for k, v in doc.metadata.items() if "Header" in k])
        # 將標題與內文重新結合
        context_parts.append(f"SECTION: [{headers}]\nCONTENT:\n{doc.page_content}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    system_prompt = f"""You are a strict and rigorous Aquaponics expert.
TASK: Answer the user's factual question based ONLY on the provided Context.

CRITICAL RULES:
1. If the Context does not contain the answer, reply EXACTLY with: "Based on the current knowledge base, I cannot provide an exact answer."
2. Do NOT hallucinate, guess, or bring in outside knowledge.
3. If the Context contains a list or specific steps, extract them EXACTLY as they appear.
4. Output entirely in English.

Context:
{context}
"""
    
    # 將 SystemMessage 與對話歷史合併
    messages_to_llm = [SystemMessage(content=system_prompt)] + state.get("messages", [])
    response = llm.invoke(messages_to_llm)
    
    return {"messages": state.get("messages", []) + [AIMessage(content=response.content)]}

def complex_node(state: AquaponicsState):
    query = state.get("user_query", "")
    original_problem = state.get("original_problem", "")
    count = state.get("follow_up_count", 0)
    
    MAX_FOLLOW_UP = 1
    
    if count >= MAX_FOLLOW_UP:
        # 【關鍵修復】將原始症狀與最新數據合併，確保能抓到對應的異常排除文本
        combined_query = f"Problem: {original_problem}. Additional Info: {query}"
        docs = retriever.invoke(combined_query)
        # 將原本的 context = "\n\n".join([doc.page_content for doc in docs]) 刪除
        # 替換為以下帶有 Metadata 解析的寫法：
        context_parts = []
        for doc in docs:
            # 提取並組合 metadata 中的標題 (如：Header 3 > Header 4)
            headers = " > ".join([str(v) for k, v in doc.metadata.items() if "Header" in k])
            # 將標題與內文重新結合
            context_parts.append(f"SECTION: [{headers}]\nCONTENT:\n{doc.page_content}")
        
        context = "\n\n---\n\n".join(context_parts)
        
# 將原本的 prompt 改為 system_prompt，並移除結尾的 "Follow-up Info: {query}"
        system_prompt = f"""You are an Aquaponics troubleshooting diagnostic AI.
TASK: Provide a FINAL diagnosis and actionable solution based ONLY on the Context.

CRITICAL RULES:
1. REASONING PROCESS: Match the user's symptoms to a specific SECTION in the Context.
2. STRICT BINDING RULE: Once you identify the correct condition, you MUST extract the "Root Cause" and "Actionable Solution" EXCLUSIVELY from that exact same SECTION.
3. EXACT EXTRACTION: Do not paraphrase. Extract the exact steps directly from the matched section.
4. Output entirely in English.

Context:
{context}

Original Problem: {original_problem}

You MUST structure your response EXACTLY in this format:
**REASONING**: [Explain which symptoms match which specific section header]
**DIAGNOSIS**: [Exact name of the disease/issue from that section]
**ROOT CAUSE**: [Extract the cause ONLY from the exact same section]
**SOLUTION**: [Extract the actionable steps ONLY from the exact same section]
"""
        messages_to_llm = [SystemMessage(content=system_prompt)] + state.get("messages", [])
        response = llm.invoke(messages_to_llm)
        return {"messages": state.get("messages", []) + [AIMessage(content=response.content)], "follow_up_count": 0, "mode": "simple"}
    
    else:
        # 第一次進入，只使用 query 檢索
        docs = retriever.invoke(query)
        # 將原本的 context = "\n\n".join([doc.page_content for doc in docs]) 刪除
        # 替換為以下帶有 Metadata 解析的寫法：
        context_parts = []
        for doc in docs:
            # 提取並組合 metadata 中的標題 (如：Header 3 > Header 4)
            headers = " > ".join([str(v) for k, v in doc.metadata.items() if "Header" in k])
            # 將標題與內文重新結合
            context_parts.append(f"SECTION: [{headers}]\nCONTENT:\n{doc.page_content}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # 將原本的 prompt 改為 system_prompt，並移除結尾的 "User Problem: {query}"
        system_prompt = f"""You are an Aquaponics troubleshooting diagnostic AI.
TASK: The user has reported a system abnormality. Gather missing critical data.

CRITICAL RULES:
1. Identify missing vital parameters (e.g., pH level, Ammonia/Nitrite ppm).
2. Ask 1 to 3 highly specific, bulleted follow-up questions.
3. Do NOT provide any solutions at this stage.
4. Output entirely in English.

Context:
{context}
"""
        messages_to_llm = [SystemMessage(content=system_prompt)] + state.get("messages", [])
        response = llm.invoke(messages_to_llm)
        return {"messages": state.get("messages", []) + [AIMessage(content=response.content)], "follow_up_count": count + 1}

# ==========================================
# 3. Build and Compile State Machine (StateGraph Execution)
# ==========================================
def route_entry(state: AquaponicsState):
    return "complex_node" if state.get("follow_up_count", 0) > 0 else "router_node"

def route_after_router(state: AquaponicsState):
    return "complex_node" if state.get("mode") == "complex" else "simple_node"

workflow = StateGraph(AquaponicsState)
workflow.add_node("router_node", router_node)
workflow.add_node("simple_node", simple_node)
workflow.add_node("complex_node", complex_node)

workflow.set_conditional_entry_point(route_entry, {"router_node": "router_node", "complex_node": "complex_node"})
workflow.add_conditional_edges("router_node", route_after_router, {"simple_node": "simple_node", "complex_node": "complex_node"})
workflow.add_edge("simple_node", END)
workflow.add_edge("complex_node", END)

app = workflow.compile()

# ==========================================
# 4. CLI Interactive Loop
# ==========================================
if __name__ == "__main__":
    print("\nLangGraph Aquaponics Expert System initialized. Type 'exit' to quit.")
    print("-" * 40)
    
# 初始化加入 original_problem
    current_state = {"messages": [], "user_query": "", "original_problem": "", "mode": "simple", "follow_up_count": 0}
    
    while True:
        if current_state["follow_up_count"] > 0:
            user_input = input("\n[Follow-up needed] Please provide more details: ")
        else:
            user_input = input("\nPlease enter your question: ")
            # 只有在新問題(追問次數為0)時，才將輸入記錄為原始症狀
            current_state["original_problem"] = user_input
            
        if user_input.lower() in ['exit', 'quit']:
                    print("System shutting down.")
                    break
                    
                # 【新增此區塊】重置記憶體狀態
        if user_input.lower() in ['reset', 'clear']:
            current_state = {"messages": [], "user_query": "", "original_problem": "", "mode": "simple", "follow_up_count": 0}
            print("[System] Memory cleared. Starting a new session.")
            continue
                    
        if not user_input.strip():
            continue
            
        print("Expert is thinking...\n")
        # 找到這行：
        current_state["user_query"] = user_input
        
        # 【新增這行】將使用者的輸入正式寫入狀態記憶體
        current_state["messages"].append(HumanMessage(content=user_input))
        
        try:
            current_state = app.invoke(current_state)
            last_message = current_state["messages"][-1]
            print(f"Expert Answer:\n{last_message.content}")
        except Exception as e:
            print(f"Error occurred: {e}")