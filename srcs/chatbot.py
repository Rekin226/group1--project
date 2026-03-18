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
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 12})

# Initialize LLM model
llm = ChatOllama(model="qwen2.5:7b", temperature=0)

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
    system_prompt = """You are an Aquaponics query classifier.
Your ONLY task is to classify the user's input into: 'simple', 'diagnostic', 'design', or 'cost'.
Output EXACTLY one word. Do NOT output any other text.

CLASSIFICATION RULES:
- 'diagnostic': Reporting a problem, physical symptoms, or troubleshooting (e.g., "Fish gasping", "yellow leaves").
- 'design': Asking about system architecture, planning a build, dimensions, or material requirements (e.g., "I want to build a system", "Which system is best for tomatoes?").
- 'cost': Asking about price, budget, financial estimates, or component costs (e.g., "How much does it cost?", "What is the price of grow media?").
- 'simple': Factual questions, definitions, or theory without reporting an active problem (e.g., "What is the ideal pH?").
"""
    
    messages_to_llm = [SystemMessage(content=system_prompt)] + state.get("messages", [])
    response = llm.invoke(messages_to_llm)
    decision = response.content.strip().lower()
    
    # 根據分類結果決定 mode 與 task_type
    if "diagnostic" in decision:
        mode, task_type = "complex", "diagnostic"
    elif "design" in decision:
        mode, task_type = "complex", "design"
    elif "cost" in decision:
        mode, task_type = "complex", "cost"
    else:
        mode, task_type = "simple", "simple"
        
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
TASK: Answer the user's factual question based STRICTLY AND ONLY on the provided Context.

CRITICAL RULES:
1. ZERO HALLUCINATION (CRITICAL): You are explicitly FORBIDDEN from using any external knowledge. If the exact specific entity (e.g., a specific fish species like 'Koi', a specific plant, or a specific metric) asked by the user is NOT explicitly mentioned in the Context, you MUST reject the question. Output EXACTLY: "Based on the current knowledge base, I cannot provide an exact answer." Do NOT guess, approximate, or substitute.
2. DIRECT ANSWER: If the exact answer exists in the Context, provide the facts directly. Do NOT start with phrases like "According to the text".
3. PARTIAL KNOWLEDGE: If you can only answer part of a multi-part question, answer the part you know. For the missing part, explicitly append: "Note: The current knowledge base does not contain information regarding [missing topic]."
4. TRACEABILITY: You MUST conclude your response with a "**SOURCES**:" section, listing the EXACT SECTION headers from the Context.

Context:
{context}

You MUST structure your response EXACTLY as follows (unless completely missing):

**ANSWER**: 
[Direct, professional answer using bullet points.]

**SOURCES**: 
- [Exact SECTION header]
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
    # 2. 泛用型「診斷」任務 Prompt
    # ==========================================

    if task_type == "diagnostic":
        system_prompt = f"""You are a strict Aquaponics troubleshooting diagnostic AI.
TASK: Assess the user's symptoms against the Context using strict diagnostic logic.

CRITICAL RULES FOR ELIMINATION (MUST FOLLOW STRICTLY):
1. CANDIDATE MATCHING (EXHAUSTIVE SEARCH): You MUST meticulously scan EVERY SINGLE section in the Context. Identify and list ALL conditions that share ANY of the user's stated symptom(s). Missing a matching condition is a fatal error. Do not stop early.
2. THE "UNSTATED" RULE (KEEP): Do NOT eliminate a condition just because the user failed to mention every symptom listed in the Context. 
3. EXPLICIT CONTRADICTION (ELIMINATE): You MUST eliminate a condition IF AND ONLY IF the user's stated symptom logically or physically contradicts the condition's description in the Context (e.g., mutually exclusive physical states, opposite behaviors, or conflicting environmental triggers).
4. ZERO HALLUCINATION: Do NOT attribute symptoms to a condition if they are not explicitly written in the Context.
5. COUNT: Count the remaining Candidate conditions.
6. STATUS DECISION:
   - If Count > 1: output **STATUS**: ASK.
   - If Count == 0: output **STATUS**: UNKNOWN.
   - ONLY if Count == 1: output **STATUS**: SOLVE.

Context:
{context}

Original Problem: {original_problem}
Current User Input: {query}

You MUST structure your response EXACTLY as follows:

**ANALYSIS**:
- Confirmed Symptoms: [List user's exact symptoms]
- Candidate Conditions: 
  * [Condition name]: Kept because [Reason].
- Eliminated Conditions: 
  * [Condition name]: Eliminated ONLY because [State the exact logical contradiction based on Context].
- Candidate Count: [Number]

**STATUS**: [ASK / SOLVE / UNKNOWN]

[If STATUS is ASK, output:]
**QUESTIONS**:
[Ask 1-3 highly specific diagnostic questions to differentiate the remaining Candidates based on their unique traits in the Context.]

[If STATUS is SOLVE, output:]
**DIAGNOSIS**: [Exact name]
**ROOT CAUSE**: [Extract from context]
**SOLUTION**: [Extract from context]

[If STATUS is UNKNOWN, output:]
**RESPONSE**: Based on the knowledge base, I cannot match these exact symptoms to a single condition.
"""
        messages_to_llm = [SystemMessage(content=system_prompt)] + state.get("messages", [])
        response = llm.invoke(messages_to_llm)
        
        if "**STATUS**: SOLVE" in response.content or "**STATUS**: UNKNOWN" in response.content:
            return {"messages": state.get("messages", []) + [AIMessage(content=response.content)], "follow_up_count": 0, "mode": "simple"}
        else:
            return {"messages": state.get("messages", []) + [AIMessage(content=response.content)], "follow_up_count": count + 1}
            
    # ==========================================
    # 3. 專精型「設計」任務 Prompt (通用物理維度比對，無硬編碼)
    # ==========================================
    elif task_type == "design":
        system_prompt = f"""You are an expert Aquaponics System Architect AI.
TASK: Recommend the optimal system architecture based on user constraints and the Context.

STEP 1: STRUCTURED EXTRACTION
Extract specific constraints. If not mentioned, write 'None'.
- Location: (e.g., warehouse, balcony, outdoors)
- Limits: (e.g., weight limitations, space constraints)
- Crops: (e.g., tomatoes, leafy greens)

STEP 2: UNIVERSAL ARCHITECTURE EVALUATION
Evaluate DWC, NFT, and Media-Based strictly against the Context. Check for physical incompatibilities:
1. Load Check: Compare the user's Location/Limits against the architecture's weight properties in the Context. If the Context states the architecture requires "structural reinforcement", has "substantial static load", "extreme static loads", or is "restricted to ground-level", it MUST be marked 'No Fit' for elevated or weight-limited locations.
2. Biology Check: Compare the user's Crops against the architecture's plant support properties in the Context. If the Context states the architecture lacks "structural support" or has "clogging risks", it MUST be marked 'No Fit' for heavy/fruiting crops.
Mark 'No Fit' if it fails ANY check, and quote the exact reason from the Context. Otherwise, mark 'Fit'. 
Count the remaining 'Fit' architectures.

STEP 3: STATUS ROUTING
- If Fit Count > 1: **STATUS**: ASK (You MUST ask questions to narrow it down to exactly 1).
- If Fit Count == 0: **STATUS**: CONFLICT
- If Fit Count == 1: **STATUS**: SOLVE_DESIGN

STEP 4: ZERO HALLUCINATION OUTPUT
Format your response exactly based on STATUS. Do not mix formats. If materials/steps are missing in Context, write "Data missing in knowledge base."

Context:
{context}

Current User Input: {query}

Output EXACTLY in this format:

**ANALYSIS**:
- Location: [Extracted]
- Limits: [Extracted]
- Crops: [Extracted]
- Evaluation:
  * DWC: [Fit / No Fit] because [Context quote]
  * NFT: [Fit / No Fit] because [Context quote]
  * Media-Based: [Fit / No Fit] because [Context quote]
- Fit Count: [Number]

**STATUS**: [ASK / CONFLICT / SOLVE_DESIGN]

**FINAL OUTPUT**:
[CRITICAL: Choose ONLY ONE format below based on the STATUS. DO NOT output the other formats.]
- If ASK: Write "QUESTIONS:" followed by specific questions to gather missing constraints.
- If CONFLICT: Write "EXPLANATION: [reason]" and "SUGGESTION: [alternative]".
- If SOLVE_DESIGN: Write "RECOMMENDED ARCHITECTURE:", "WHY IT FITS:", "REQUIRED MATERIALS:", and "CONSTRUCTION STEPS:" strictly from Context.
"""
        messages_to_llm = [SystemMessage(content=system_prompt)] + state.get("messages", [])
        response = llm.invoke(messages_to_llm)
        content = response.content
        
        if "**STATUS**: CONFLICT" in content or "**STATUS**: SOLVE_DESIGN" in content:
            return {"messages": state.get("messages", []) + [AIMessage(content=content)], "follow_up_count": 0, "mode": "simple"}
        else:
            return {"messages": state.get("messages", []) + [AIMessage(content=content)], "follow_up_count": count + 1}

    # ==========================================
    # 4. 專精型「成本估算」任務 Prompt (完全獨立，防止格式污染)
    # ==========================================
    elif task_type == "cost":
        system_prompt = f"""You are an expert Aquaponics System Architect AI.
TASK: Recommend the optimal system architecture and provide a cost estimate based on the user's intent and Context.

CRITICAL RULES:
1. SHORT-CIRCUIT BLOCK (CRITICAL): If the user provides ZERO specific constraints (e.g., no crop type, no space limits), you MUST extract 'None'. 
   -> If Extracted Constraints is 'None', ALL architectures MUST be marked as 'Pending' and you MUST output **STATUS**: ASK. Do NOT guess or assume any Fit.
2. STRICT EVALUATION & PHYSICAL RULES:
   - You MUST evaluate constraints against the Context. Narrow down to EXACTLY ONE architecture (DWC, NFT, or Media-Based).
   - ROOFTOP/WEIGHT RULE: If the user mentions "rooftop", "balcony", "space-constrained", or "weight limit", you MUST eliminate DWC and Media-Based. You MUST choose NFT.
   - FRUIT RULE: If the user mentions "fruiting plants" or "tomatoes", you MUST eliminate NFT.
3. INTENT DETECTION: Determine if the user is asking "How much it costs" (Intent: COST).
4. ZERO HALLUCINATION (CRITICAL): 
   - For SOLVE_COST: Extract EXACT prices from 'Economic Variables'. Do NOT guess prices or use external data.
5. STATUS DECISION:
   - If constraints are 'None' or insufficient: output **STATUS**: ASK.
   - If requirements are physically incompatible: output **STATUS**: CONFLICT.
   - If EXACTLY ONE architecture fits AND user wants a cost estimate: output **STATUS**: SOLVE_COST.

Context:
{context}

Original Request: {original_problem}
Current User Input: {query}

You MUST structure your response EXACTLY as follows:

**ANALYSIS**:
- Extracted Constraints: [List extracted requirements. Write 'None' if none]
- User Intent: COST
- Architecture Evaluation:
  * DWC: [Fit / No Fit / Pending] because [reason]
  * NFT: [Fit / No Fit / Pending] because [reason]
  * Media-Based: [Fit / No Fit / Pending] because [reason]
  (Note: If constraints are 'None', all MUST be 'Pending')

**STATUS**: [ASK / CONFLICT / SOLVE_COST]

[If STATUS is ASK, output:]
**QUESTIONS**:
- [Ask 1-3 specific questions to gather missing constraints]

[If STATUS is CONFLICT, output:]
**EXPLANATION**: [Explain why constraints contradict based on physical rules]
**SUGGESTION**: [Suggest a compromise]

[If STATUS is SOLVE_COST, output:]
**RECOMMENDED ARCHITECTURE**: [Exact name]
**WHY IT FITS YOU**: [Explain why based on characteristics]
**COST ESTIMATE TABLE**:
[Generate a Markdown table listing each component, expected lifespan, and cost range STRICTLY from 'Economic Variables'.]
**TOTAL ESTIMATE SUMMARY**:
[Summarize the estimated total capital investment and performance benchmarks based ONLY on the Context.]
"""
        messages_to_llm = [SystemMessage(content=system_prompt)] + state.get("messages", [])
        response = llm.invoke(messages_to_llm)
        content = response.content
        
        return {"messages": state.get("messages", []) + [AIMessage(content=content)], "follow_up_count": 0, "mode": "simple"}
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