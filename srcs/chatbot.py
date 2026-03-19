import os
from typing import TypedDict, List
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage, SystemMessage
from langgraph.graph import StateGraph, END

# ==========================================
# 1. Knowledge Base and Vector Search Setup (RAG Setup)
# ==========================================
def setup_vectorstore(db_path="./chroma_db", url_path="urls.txt"):
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    
    # 如果已存在向量資料庫則直接讀取
    if os.path.exists(db_path):
        print(f"Loading existing vector database from {db_path}")
        return Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    print("Initializing and building vector database from URLs...")
    
    documents = []
    # 修改部分：從 urls.txt 讀取網址並抓取內容
    if os.path.exists(url_path):
        with open(url_path, "r", encoding="utf-8") as f:
            # 讀取每一行，過濾掉空白行，並處理掉可能存在的 前綴
            urls = []
            for line in f:
                clean_line = line.strip()
                if clean_line:
                    # 如果行內包含 http，則提取網址部分
                    if "http" in clean_line:
                        url = clean_line.split(" ")[-1] # 取得空格後的網址
                        urls.append(url)
            
        if urls:
            print(f"Fetching content from {len(urls)} URLs...")
            # 使用 WebBaseLoader 抓取網頁內容
            loader = WebBaseLoader(urls)
            documents = loader.load()
    
    # 由於網頁內容為 HTML/純文字，不再適合使用 MarkdownHeaderTextSplitter
    # 直接使用 RecursiveCharacterTextSplitter 進行切分
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
    
    print("\n[Debug] Retrieved Contexts:")
    context_parts = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown Source")
        title = doc.metadata.get("title", "No Title").strip().replace("\n", "")
        print(f"  - Source: {source} | Title: {title[:30]}...")
        context_parts.append(f"SOURCE: [{source}]\nCONTENT:\n{doc.page_content}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    system_prompt = f"""You are a strict and rigorous Aquaponics expert.
TASK: Answer the user's factual question based STRICTLY AND ONLY on the provided Context.

CRITICAL RULES:
1. FACT VERIFICATION (CRITICAL): Before answering, you MUST scan the Context for the exact entities requested (e.g., specific pH numbers, commercial brands). You are STRICTLY FORBIDDEN from using outside or pre-trained knowledge.
2. DIRECT ANSWER (NO FLUFF): Provide the facts directly. Use bullet points. Do not start with "Based on the context".
3. ZERO HALLUCINATION: If the entire answer is missing, output EXACTLY: "Based on the current knowledge base, I cannot provide an exact answer."
4. PARTIAL KNOWLEDGE: If you can answer part of the question, but a specific requested detail (e.g., a specific commercial brand) is NOT explicitly named in the Context, you MUST NOT guess. Answer what is available, and explicitly append: "Note: The current knowledge base does not contain information regarding [missing topic]."
5. TRACEABILITY: Conclude your response with a "**SOURCES**:" section, listing ONLY the exact SOURCE URLs from the Context.

Context:
{context}

Current User Input: {query}

You MUST structure your response EXACTLY as follows:

**FACT VERIFICATION**: [Briefly state which requested facts are explicitly found in the Context, and which are missing.]

**ANSWER**: 
- [Direct fact 1 from Context]
- [Direct fact 2 from Context]

Note: The current knowledge base does not contain information regarding [missing topic, if applicable].

**SOURCES**: 
- [Source URL 1]
- [Source URL 2]
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
        source = doc.metadata.get("source", "Unknown Source")
        title = doc.metadata.get("title", "No Title").strip().replace("\n", "")
        print(f"  - Source: {source} | Title: {title[:30]}...")
        context_parts.append(f"SOURCE: [{source}]\nCONTENT:\n{doc.page_content}")
    
    context = "\n\n---\n\n".join(context_parts)

    # ==========================================
    # 2. 泛用型「診斷」任務 Prompt
    # ==========================================

    if task_type == "diagnostic":
        system_prompt = f"""You are a strict Aquaponics troubleshooting diagnostic AI.
TASK: Assess the user's symptoms against the Context using strict diagnostic logic.

CRITICAL RULES FOR ELIMINATION (MUST FOLLOW STRICTLY):
1. STRICT DEFINITION OF 'CONDITION' (CRITICAL): A "Condition" MUST be an explicitly named disease, toxicity, nutrient deficiency, or specific parameter failure (e.g., 'Ammonia Toxicity', 'Iron Deficiency', 'High pH'). A symptom itself (e.g., 'Cloudy Water', 'Yellow Leaves') is STRICTLY FORBIDDEN from being listed as a Condition.
2. CANDIDATE MATCHING (EXHAUSTIVE SEARCH): Scan the Context. Identify ALL Conditions that share ANY of the user's stated symptom(s). If the Context states a symptom can be caused by multiple distinct factors (e.g., cloudy water caused by high ammonia OR poor filtration), EACH factor MUST be listed as a separate Candidate Condition.
3. THE "UNSTATED" RULE (KEEP): Do NOT eliminate a condition just because the user failed to mention every symptom listed in the Context. 
4. EXPLICIT CONTRADICTION (ELIMINATE): You MUST eliminate a condition IF AND ONLY IF the user's stated symptom logically or physically contradicts the condition's description in the Context.
5. ZERO HALLUCINATION (STRICT): Do NOT attribute symptoms to a condition if they are not explicitly written in the Context. If the Context contains NO explicitly matching conditions, your Candidate Count MUST be 0.
6. NO UNSOLICITED ADVICE (CRITICAL): If Candidate Count is 0, you are STRICTLY FORBIDDEN from offering general advice, generating a 'RECOMMENDATION' section, or guessing troubleshooting steps.
7. STATUS DECISION:
   - If Candidate Count > 1: output **STATUS**: ASK.
   - If Candidate Count == 0: output **STATUS**: UNKNOWN.
   - ONLY if Candidate Count == 1: output **STATUS**: SOLVE.

Context:
{context}

Original Problem: {original_problem}
Current User Input: {query}

You MUST structure your response EXACTLY as follows:

**ANALYSIS**:
- Confirmed Symptoms: [List user's exact symptoms]
- Candidate Conditions: 
  * [Explicit Condition name 1]: Kept because [Reason based on Context].
  * [Explicit Condition name 2]: Kept because [Reason based on Context].
  (Write 'None' if 0 candidates)
- Eliminated Conditions: 
  * [Condition name]: Eliminated ONLY because [State the exact logical contradiction based on Context]. (Write 'None' if none eliminated)
- Candidate Count: [Number]

**STATUS**: [ASK / SOLVE / UNKNOWN]

[If STATUS is ASK, output:]
**QUESTIONS**:
[Ask 1-3 highly specific diagnostic questions (e.g., asking for specific water parameter readings or visual details) to differentiate the remaining Candidates based on their unique traits in the Context.]

[If STATUS is SOLVE, output:]
**DIAGNOSIS**: [Exact Condition name from Context]
**ROOT CAUSE**: [Extract from Context]
**SOLUTION**: [Extract from Context]

[If STATUS is UNKNOWN, output EXACTLY AND ONLY the following sentence. DO NOT append any advice or recommendations:]
**RESPONSE**: Based on the knowledge base, I cannot match these exact symptoms to a specific condition.
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
- Limits/Capacity: Extract ANY mention of space size, weight limits, or the LACK thereof (e.g., "strict weight limit", "no weight limits", "plenty of space"). Do NOT write 'None' if the user explicitly states their capacity.
- Crops: (e.g., tomatoes, leafy greens)

STEP 2: UNIVERSAL ARCHITECTURE EVALUATION
Evaluate DWC, NFT, and Media-Based strictly against the Context. Check for physical incompatibilities:
1. Load Check: Compare the user's Location/Limits against the architecture's weight properties in the Context. If the Context implies the architecture is heavy (e.g., deep water volume in DWC, gravel/clay pebbles in Media-Based) and the user has strict weight limits or elevated locations (balcony/rooftop), you MUST mark it 'No Fit'.
2. Biology Check: Compare the user's Crops against the architecture's plant support properties in the Context. Heavy/fruiting crops (like tomatoes) require root space and physical support. If the system lacks this (e.g., standard NFT), you MUST mark it 'No Fit'.
Mark 'No Fit' if it fails ANY check, and quote the exact reason from the Context. Otherwise, mark 'Fit'. 
Count the remaining 'Fit' architectures.

STEP 3: STATUS ROUTING
- If Fit Count > 1: **STATUS**: ASK (You MUST ask questions to narrow it down to exactly 1).
- If Fit Count == 0: **STATUS**: CONFLICT
- If Fit Count == 1: **STATUS**: SOLVE_DESIGN

STEP 4: ZERO HALLUCINATION OUTPUT & LOGICAL CONSISTENCY
Format your response exactly based on STATUS. Do not mix formats.
CRITICAL RULE: If STATUS is CONFLICT, your SUGGESTION MUST NOT recommend an architecture you just marked as 'No Fit'. Instead, suggest modifying the user's constraints (e.g., "Change crop to leafy greens to utilize lightweight NFT") strictly based on Aquaponics principles in the Context.

Context:
{context}

Current User Input: {query}

Output EXACTLY in this format:

**ANALYSIS**:
- Location: [Extracted]
- Limits: [Extracted]
- Crops: [Extracted]
- Evaluation:
  * DWC: [Fit / No Fit] because [Context quote or physical constraint]
  * NFT: [Fit / No Fit] because [Context quote or physical constraint]
  * Media-Based: [Fit / No Fit] because [Context quote or physical constraint]
- Fit Count: [Number]

**STATUS**: [ASK / CONFLICT / SOLVE_DESIGN]

**FINAL OUTPUT**:
[CRITICAL: Choose ONLY ONE format below based on the STATUS. DO NOT output the other formats.]
- If ASK: Write "QUESTIONS:" followed by specific questions to gather missing constraints.
- If CONFLICT: Write "EXPLANATION: [Explain why all failed based on physics/Context]" and "SUGGESTION: [Suggest which user constraint to change to make a system viable. DO NOT invent hybrid systems]."
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
TASK: Recommend the optimal system architecture and provide a cost estimate strictly based on the Context.

STEP 1: CONSTRAINT EXTRACTION
Extract specific constraints from BOTH the Original Request and Current User Input. 
Valid constraints include: location (e.g., backyard, balcony), space/weight limits, and crop types (e.g., lettuce, tomatoes).
If NO valid constraints are found, you MUST write 'None'.

STEP 2: SHORT-CIRCUIT BLOCK
If Extracted Constraints is 'None', ALL architectures MUST be marked as 'Pending' and you MUST output **STATUS**: ASK.

STEP 3: STRICT EVALUATION & PHYSICAL RULES
Evaluate constraints against the Context. Narrow down to EXACTLY ONE architecture (DWC, NFT, or Media-Based).
- ROOFTOP/WEIGHT RULE: If location implies weight limits (rooftop, balcony), eliminate DWC and Media-Based.
- FRUIT RULE: If growing heavy/fruiting plants, eliminate NFT.

STEP 4: ZERO HALLUCINATION (CRITICAL FOR COST)
You are STRICTLY FORBIDDEN from using your pre-trained knowledge to guess prices. 
If the Context does NOT contain explicit numerical prices (e.g., "$50") for the required materials, you MUST output "Data missing in knowledge base" in the cost column. Do NOT invent numbers.

STEP 5: STATUS DECISION
- If constraints are 'None' or insufficient: **STATUS**: ASK
- If constraints physically contradict: **STATUS**: CONFLICT
- If EXACTLY ONE architecture fits AND user wants a cost estimate: **STATUS**: SOLVE_COST

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

**STATUS**: [ASK / CONFLICT / SOLVE_COST]

[If STATUS is ASK, output:]
**QUESTIONS**:
- [Ask 1-3 specific questions to gather missing constraints]

[If STATUS is CONFLICT, output:]
**EXPLANATION**: [Explain why constraints contradict based on physics]
**SUGGESTION**: [Suggest a compromise]

[If STATUS is SOLVE_COST, output:]
**RECOMMENDED ARCHITECTURE**: [Exact name]
**WHY IT FITS YOU**: [Explain why based on characteristics]
**PRICE VERIFICATION**: [CRITICAL: Write "I have checked the Context. Specific numerical prices are missing." or "Prices are present."]
**COST ESTIMATE TABLE**:
| Component | Cost Estimate |
|---|---|
| [List component] | [Exact price from Context OR "Data missing in knowledge base"] |
**TOTAL ESTIMATE SUMMARY**:
[Summarize based ONLY on the Context. If prices are missing, explicitly state that a total cost cannot be calculated.]
"""
        messages_to_llm = [SystemMessage(content=system_prompt)] + state.get("messages", [])
        response = llm.invoke(messages_to_llm)
        content = response.content
        # 新增條件判斷，確保追問狀態能正確傳遞
        if "**STATUS**: ASK" in content or "**STATUS**: CONFLICT" in content:
            return {"messages": state.get("messages", []) + [AIMessage(content=content)], "follow_up_count": count + 1}
        else:
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