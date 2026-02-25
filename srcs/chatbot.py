"""Aquaponics AI - v7.9.0 (Full English Version)
- Language: Strictly English only.
- Formatting: Next questions are now formatted as Q1:, Q2:, etc.
- RAG: Restored loading from urls.txt and local knowledge files.
"""

from __future__ import annotations
import json
import logging
import pathlib
import re
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import requests_cache
from langchain.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# =============================================================================
# Configuration
# =============================================================================
URL_FILE = "urls.txt"
KNOWLEDGE_DIR = "knowledge"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
TOP_K = 3

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# =============================================================================
# LLM & State Initialization
# =============================================================================
llm = OllamaLLM(model="llama3", temperature=0)

class ThreadState:
    def __init__(self): self.reset()
    def reset(self):
        self.known_facts = {}
        self.asked_questions = []

state = ThreadState()

# =============================================================================
# Prompts - English Only
# =============================================================================

router_system = """You are a router. Classify input:
1. 'FAQ': General knowledge (e.g., "What is aquaponics?"). 
2. 'DIAGNOSIS': Troubleshooting an issue or wanting to BUILD/DESIGN a system.
Return ONLY 'FAQ' or 'DIAGNOSIS'."""

faq_system_en = """You are a professional aquaponics expert. Respond in English.
1. Answer the question based on the provided context.
2. Use friendly analogies and be concise.
3. End by asking if the user needs help designing or troubleshooting their own system."""

diagnostic_system = """You are an expert system designer. Return ONLY JSON.
If the user wants to DESIGN, BUILD, or FIX a system, set mode to 'ASK_MORE'.
Ask 1-3 critical questions to gather missing info (e.g., tank size, budget, fish species).

JSON structure:
{{
  "mode": "ASK_MORE" | "FINAL_PLAN",
  "known_facts_update": {{}},
  "next_questions": [],
  "immediate_actions": [],
  "followup_offer": "One sentence offer"
}}
All text values MUST be in English."""

# =============================================================================
# RAG Core (Restored URL & File Loader)
# =============================================================================

def load_all_documents():
    documents = []
    
    # 1. Load from urls.txt
    url_p = pathlib.Path(URL_FILE)
    if url_p.exists():
        urls = [line.strip() for line in url_p.read_text().splitlines() if line.strip() and not line.startswith("#")]
        for url in urls:
            try:
                loader = WebBaseLoader(url)
                documents.extend(loader.load())
            except Exception as e:
                logging.warning(f"Failed to load URL {url}: {e}")

    # 2. Load from knowledge/
    base = pathlib.Path(KNOWLEDGE_DIR)
    if base.exists():
        for fp in base.rglob("*"):
            if fp.suffix.lower() in {".md", ".txt"}:
                try:
                    loader = TextLoader(str(fp), encoding="utf-8")
                    documents.extend(loader.load())
                except Exception as e:
                    logging.warning(f"Failed to load file {fp}: {e}")
    
    return documents

def build_vector_store():
    docs = load_all_documents()
    if not docs:
        logging.warning("No documents found. Running without RAG.")
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return FAISS.from_documents(splitter.split_documents(docs), HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))

# =============================================================================
# Interaction Handler
# =============================================================================

def handle_turn(user: str, vs: Optional[FAISS]):
    # 1. Intent Routing
    intent = llm.invoke(ChatPromptTemplate.from_messages([("system", router_system), ("human", "{u}")]).format_messages(u=user)).strip().upper()
    
    # Force Diagnosis mode for specific keywords
    if any(kw in user.lower() for kw in ["design", "build", "setup", "recommend", "size", "create"]):
        intent = "DIAGNOSIS"

    context = ""
    if vs:
        docs = vs.similarity_search(user, k=TOP_K)
        context = "\n\n".join(d.page_content for d in docs)

    # --- Mode A: FAQ Knowledge ---
    if "FAQ" in intent:
        ans = llm.invoke(ChatPromptTemplate.from_messages([("system", faq_system_en), ("human", "Context: {c}\n\nQuestion: {u}")]).format_messages(c=context, u=user))
        print(f"\n[AI Knowledge] {ans}\n")
        return

    # --- Mode B: Diagnosis & Design (Numbered Question Mode) ---
    raw_json = llm.invoke(ChatPromptTemplate.from_messages([("system", diagnostic_system), ("human", "Facts: {f}\nContext: {c}\nUser: {u}")]).format_messages(
        f=json.dumps(state.known_facts), c=context, u=user
    ))

    try:
        # Extract JSON safely
        clean_json = re.search(r"({.*})", raw_json, re.DOTALL).group(1)
        data = json.loads(clean_json)
        state.known_facts.update(data.get("known_facts_update", {}))
        
        # Output Actions
        if data.get("immediate_actions"):
            print(f"\n[Immediate Actions]")
            for action in data["immediate_actions"]:
                print(f" - {action}")

        # Output Questions with Q1, Q2, Q3 format
        if data.get("next_questions"):
            print(f"\n[To help you further, please answer the following:]")
            q_idx = 1
            for q in data["next_questions"]:
                if q not in state.asked_questions:
                    print(f" Q{q_idx}: {q}")
                    state.asked_questions.append(q)
                    q_idx += 1
        
        if data.get("followup_offer"):
            print(f"\n{data['followup_offer']}")

    except Exception:
        print(f"\n[AI] I need more details to provide a specific recommendation. Could you describe your goals or space?")

# =============================================================================
# Main Entry
# =============================================================================
def main():
    requests_cache.install_cache("aquaponics_cache", expire_after=86400)
    print("Loading knowledge base (URLs and local files)...")
    vs = build_vector_store()
    print(f"--- Aquaponics Chatbot ---")
    print(f"Status: English Mode | RAG: {'Active' if vs else 'Offline'}")
    
    while True:
        try:
            u = input("\nYou> ")
            if u.lower() in ["exit", "quit"]: break
            handle_turn(u, vs)
        except KeyboardInterrupt: break

if __name__ == "__main__":
    main()