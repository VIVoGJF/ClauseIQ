# rag/qa.py
import re
import json
import ast
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from rag.store import get_retriever

load_dotenv()

# -------------------------
# LLM
# -------------------------
_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_retries=2,
    api_key=os.getenv("GROQ_API_KEY"),
)

# -------------------------
# Prompt
# -------------------------
_PROMPT = ChatPromptTemplate.from_template("""
You are a legal assistant specializing in Indian law.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I could not find this in the document."
Do not make up information.
Fix any OCR errors in the context that might affect the answer
ONLY return valid JSON. Do not include triple backticks or any extra text.

Context:
{context}

Question:
{question}

Return a single JSON with exactly these keys:
- "answer": clear and concise plain English answer (2-4 sentences)
- "confidence": exactly one of: high, medium, low
- "caveat": one line limitation or uncertainty, or "None"
""")

# -------------------------
# Helpers — same pattern as summarization.py
# -------------------------
def _remove_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    return text

def _extract_first_json_block(text: str) -> str:
    match = re.search(r"\{[\s\S]*\}", text)
    return match.group(0) if match else text

def _try_parse_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    try:
        repaired = re.sub(r"(?<!\\)\'", '"', text)
        return json.loads(repaired)
    except Exception:
        return None

def _format_docs(docs) -> str:
    parts = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        parts.append(f"[Page {page}]\n{doc.page_content}")
    return "\n\n".join(parts)

def _extract_pages(docs) -> list[int]:
    pages = [doc.metadata.get("page") for doc in docs if doc.metadata.get("page")]
    return sorted(set(pages))


# -------------------------
# Public API
# -------------------------
def answer(
    question: str,
    doc_id: str,
    top_k: int = 5,
    filter: dict = None,
) -> dict:
    if not question.strip():
        return {
            "answer": "Please provide a question.",
            "relevant_pages": [],
            "confidence": "low",
            "caveat": "Empty question.",
            "chunks_used": 0,
        }

    retriever = get_retriever(doc_id=doc_id, top_k=top_k, filter=filter)
    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        return {
            "answer": "No relevant content found in the document for this question.",
            "relevant_pages": [],
            "confidence": "low",
            "caveat": "No chunks retrieved from Pinecone.",
            "chunks_used": 0,
        }

    context = _format_docs(retrieved_docs)
    pages = _extract_pages(retrieved_docs)

    # RAG chain
    chain = _PROMPT | _llm | StrOutputParser()
    raw = chain.invoke({"context": context, "question": question})

    # Parse — same as summarization.py
    cleaned = _remove_code_fences(raw)
    cleaned = _extract_first_json_block(cleaned).strip()
    parsed = _try_parse_json(cleaned)

    if parsed:
        confidence = parsed.get("confidence", "medium").lower()
        if confidence not in ["high", "medium", "low"]:
            confidence = "medium"
        return {
            "answer": parsed.get("answer", raw).strip(),
            "relevant_pages": pages,
            "confidence": confidence,
            "caveat": parsed.get("caveat", "None"),
            "chunks_used": len(retrieved_docs),
        }

    # fallback if JSON parsing fails
    return {
        "answer": raw.strip(),
        "relevant_pages": pages,
        "confidence": "medium",
        "caveat": "None",
        "chunks_used": len(retrieved_docs),
    }