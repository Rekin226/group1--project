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
# 找到這行並刪除：
# retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 15})

# 替換為標準的相似度檢索：
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})

# Initialize LLM model
llm = ChatOllama(model="llama3", temperature=0)

# ==========================================
# 2. Define Graph State and Nodes (LangGraph Node)
# ==========================================
class AquaponicsState(TypedDict):
    messages: List[AnyMessage]
    user_query: str
    original_problem: str
    mode: str
    task_type: str              # 【新增】用於區分 diagnostic 或 design
    follow_up_count: int

# 4. 定義路由分類節點 (Router Node)
def router_node(state: AquaponicsState):
    # 改寫為高精度的三元分類提示詞
    system_prompt = """You are an Aquaponics query classifier.
Your ONLY task is to classify the user's input into one of three categories: 'simple', 'diagnostic', or 'design'.
Output EXACTLY one word. Do NOT output any other text.

CLASSIFICATION RULES:
- Output 'simple' IF the user is asking a general factual question (e.g., "What is the ideal pH?").
- Output 'diagnostic' IF the user is reporting a system problem or asking for troubleshooting (e.g., "My fish are dying", "Roots are mushy").
- Output 'design' IF the user is asking about system architecture, building, dimensions, component selection, or cost estimation (e.g., "I want to build a DWC system", "How much does it cost?").
"""
    
    messages_to_llm = [SystemMessage(content=system_prompt)] + state.get("messages", [])
    response = llm.invoke(messages_to_llm)
    decision = response.content.strip().lower()
    
    # 根據分類結果決定 mode (決定是否需要追問) 與 task_type (決定套用哪種專家 Prompt)
    if "diagnostic" in decision:
        mode = "complex"
        task_type = "diagnostic"
    elif "design" in decision:
        mode = "complex"
        task_type = "design"
    else:
        mode = "simple"
        task_type = "simple"
        
    print(f"\n[Internal] Router decided mode: {mode} | task_type: {task_type}")
    
    return {"mode": mode, "task_type": task_type}

def simple_node(state: AquaponicsState):
    query = state.get("user_query", "")
    docs = retriever.invoke(query)
    
    context_parts = []
    for doc in docs:
        headers = " > ".join([str(v) for k, v in doc.metadata.items() if "Header" in k])
        context_parts.append(f"SECTION: [{headers}]\nCONTENT:\n{doc.page_content}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    system_prompt = f"""You are a strict and rigorous Aquaponics expert.
TASK: Answer the user's factual question based ONLY on the provided Context.

CRITICAL RULES:
1. DIRECT ANSWER (NO FLUFF): Do NOT start with phrases like "Based on the context" or "According to the text". Provide the facts directly.
2. ZERO HALLUCINATION: If the entire answer is missing, output EXACTLY: "Based on the current knowledge base, I cannot provide an exact answer."
3. PARTIAL KNOWLEDGE: If you can only answer part of a multi-part question, answer the part you know. For the missing part, explicitly append: "Note: The current knowledge base does not contain information regarding [missing topic]."
4. TRACEABILITY: You MUST conclude your response with a "**SOURCES**:" section, listing the EXACT SECTION headers from the Context used to formulate your answer.

Context:
{context}

You MUST structure your response EXACTLY as follows (unless completely missing):

**ANSWER**: 
[Direct, professional answer. Use bullet points for steps or multiple facts.]

**SOURCES**: 
- [Exact SECTION header 1]
- [Exact SECTION header 2]
"""
    
    messages_to_llm = [SystemMessage(content=system_prompt)] + state.get("messages", [])
    response = llm.invoke(messages_to_llm)
    
    return {"messages": state.get("messages", []) + [AIMessage(content=response.content)]}

def complex_node(state: AquaponicsState):
    query = state.get("user_query", "")
    original_problem = state.get("original_problem", "")
    count = state.get("follow_up_count", 0)
    task_type = state.get("task_type", "diagnostic")
    
    # 提取歷史紀錄中所有 HumanMessage，確保檢索時不會遺漏如 "rooftop" 這樣的中間線索
    user_messages = [msg.content for msg in state.get("messages", []) if isinstance(msg, HumanMessage)]
    search_query = " | ".join(user_messages)
    
    docs = retriever.invoke(search_query)
    
    print("\n[Debug] Retrieved Contexts:")
    context_parts = []
    for doc in docs:
        headers = " > ".join([str(v) for k, v in doc.metadata.items() if "Header" in k])
        print(f"  - {headers}")
        context_parts.append(f"SECTION: [{headers}]\nCONTENT:\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    # ==========================================
    # 2. 針對「診斷」任務的動態多輪 Prompt
    # ==========================================
    if task_type == "diagnostic":
        system_prompt = f"""You are a strict Aquaponics troubleshooting diagnostic AI.
TASK: Assess the user's symptoms against the Context using the process of strict ELIMINATION.

CRITICAL RULES:
1. MATCHING: Identify conditions in the Context that share the user's symptoms.
2. ELIMINATION (CRITICAL): You MUST actively eliminate conditions that CONTRADICT the user's specific symptoms. For example, if the user states "dark brown gills", you MUST eliminate conditions that specify "pale gills" (like Flukes) or do not mention brown gills.
3. COUNT: Count the REMAINING conditions that perfectly align with ALL provided specific symptoms without any contradictions.
4. STATUS DECISION:
   - If the Remaining Count is 2 or more, or 0, output **STATUS**: ASK.
   - ONLY if the Remaining Count is EXACTLY 1, output **STATUS**: SOLVE.

Context:
{context}

Original Problem: {original_problem}
Current User Input: {query}

You MUST structure your response EXACTLY as follows:

**ANALYSIS**:
- Confirmed Symptoms: [List the user's exact stated symptoms]
- Eliminated Conditions: [List conditions you removed and briefly explain WHY. e.g., "Monogenean Flukes eliminated because user reported dark brown gills, whereas Flukes cause pale gills."]
- Remaining Conditions: [List the exact names of the surviving conditions]
- Remaining Count: [Number of conditions listed in Remaining Conditions]

**STATUS**: [ASK if Remaining Count is not 1. SOLVE if Remaining Count is exactly 1]

[If STATUS is ASK, output:]
**QUESTIONS**:
- [Ask 1-2 highly specific questions to differentiate the remaining conditions]

[If STATUS is SOLVE, output:]
**DIAGNOSIS**: [Exact name of the disease/issue]
**ROOT CAUSE**: [Extract from context]
**SOLUTION**: [Extract from context]
"""
        messages_to_llm = [SystemMessage(content=system_prompt)] + state.get("messages", [])
        response = llm.invoke(messages_to_llm)
        
        # 攔截邏輯保持不變
        if "**STATUS**: SOLVE" in response.content:
            return {"messages": state.get("messages", []) + [AIMessage(content=response.content)], "follow_up_count": 0, "mode": "simple"}
        else:
            return {"messages": state.get("messages", []) + [AIMessage(content=response.content)], "follow_up_count": count + 1}
            
    # ==========================================
    # 3. 針對「設計」任務 (保留舊有寫死次數的邏輯)
    # ==========================================
    elif task_type == "design":
        system_prompt = f"""You are a strict Aquaponics System Architect AI.
TASK: Recommend the optimal system architecture (DWC, NFT, or Media-Based) based ONLY on the Context.

CRITICAL RULES:
1. MANDATORY CONSTRAINTS: You CANNOT design a system without knowing BOTH the 'Target Crop' and the 'Space/Weight Limitations'.
2. STRICT EVALUATION: If the user's input does NOT explicitly name a crop (e.g., tomatoes, lettuce) or a space/weight constraint (e.g., balcony, backyard), you MUST output 'No'. Do not guess or assume.
3. ROOFTOP RULE: If the user mentions "rooftop", "balcony", or "weight limit", you MUST eliminate Media-Based and DWC.
4. FRUIT RULE: If the user mentions "fruiting plants" or "tomatoes", you MUST eliminate NFT.

Context:
{context}

Original Request: {original_problem}
Current User Input: {query}

You MUST structure your response EXACTLY as follows (DO NOT use brackets around Yes/No):

**ANALYSIS**:
- Target Crop Known: Yes or No
- Space/Weight Known: Yes or No
- Rooftop Rule Triggered: Yes or No
- Fruit Rule Triggered: Yes or No
- Remaining Architectures: [List surviving architectures. If none, write None]

**STATUS**: [ASK / CONFLICT / SOLVE]

[If STATUS is ASK, output:]
**QUESTIONS**:
- [Ask specific questions to gather the missing constraints]

[If STATUS is CONFLICT, output:]
**EXPLANATION**: [Explain why constraints contradict based on context]
**SUGGESTION**: Please adjust either your target crop or your location limitations.

[If STATUS is SOLVE, output:]
**RECOMMENDED ARCHITECTURE**: [Exact name]
**KEY MECHANICS**: [Extract functional details]
**ECONOMIC VARIABLES & COSTS**: [Extract exact prices, costs, and lifespans]
"""
        messages_to_llm = [SystemMessage(content=system_prompt)] + state.get("messages", [])
        response = llm.invoke(messages_to_llm)
        content = response.content
        
        # ==========================================
        # 程式碼護欄 (Programmatic Guardrail) - 終極防線
        # ==========================================
        # 清理可能被 LLM 誤加的中括號，確保字串比對準確
        clean_content = content.replace("[", "").replace("]", "")
        
        missing_crop = "Target Crop Known: No" in clean_content
        missing_space = "Space/Weight Known: No" in clean_content
        
        rooftop_triggered = "Rooftop Rule Triggered: Yes" in clean_content
        fruit_triggered = "Fruit Rule Triggered: Yes" in clean_content
        
        # 1. 攔截：零解狀態 (Zero-Solution Conflict)
        if rooftop_triggered and fruit_triggered:
            print("\n[System Guardrail] 攔截到物理條件衝突！發布 CONFLICT。")
            conflict_msg = "**STATUS**: CONFLICT\n\n"
            conflict_msg += "**EXPLANATION**: Based on the aquaponics engineering constraints, your requirements are physically incompatible.\n"
            conflict_msg += "1. Growing heavy fruiting plants requires Media-Based systems. NFT must be eliminated due to root clogging risks.\n"
            conflict_msg += "2. Strict weight limits eliminate Media-Based systems and DWC systems.\n"
            conflict_msg += "**SUGGESTION**: You must make a design compromise. Either switch your target crop to 'leafy greens' or move to a ground-level space."
            return {"messages": state.get("messages", []) + [AIMessage(content=conflict_msg)], "follow_up_count": 0, "mode": "simple"}

        # 2. 攔截：資訊不足 (Missing Constraints)
        if missing_crop or missing_space:
            print("\n[System Guardrail] 攔截到資訊不足，啟動強制提問。")
            dynamic_questions = []
            if missing_crop:
                dynamic_questions.append("- What is your specific target crop (e.g., leafy greens, fruiting plants)?")
            if missing_space:
                dynamic_questions.append("- What are your space and weight limitations (e.g., rooftop, backyard)?")
            
            questions_str = "\n".join(dynamic_questions)
            # 覆寫 LLM 內容，強制轉換為 ASK 狀態
            override_content = content.split("**STATUS**: ")[0] + f"**STATUS**: ASK\n\n**QUESTIONS**:\n{questions_str}"
            
            return {"messages": state.get("messages", []) + [AIMessage(content=override_content)], "follow_up_count": count + 1}
        
        # 3. 正常流程路由
        if "**STATUS**: SOLVE" in content:
            return {"messages": state.get("messages", []) + [AIMessage(content=content)], "follow_up_count": 0, "mode": "simple"}
        else:
            return {"messages": state.get("messages", []) + [AIMessage(content=content)], "follow_up_count": count + 1}
        
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