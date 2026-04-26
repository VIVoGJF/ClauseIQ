# rag/qa.py
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
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

Context:
{context}

Question:
{question}

Respond with:
- answer: clear and concise answer (2-4 sentences)
- relevant_pages: which pages the answer was found on
- confidence: high / medium / low
- caveat: any limitation or uncertainty in the answer (or "None")
""")

# -------------------------
# Helpers
# -------------------------
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
    """
    Answer a question about a specific document using RAG.

    Args:
        question: Natural language question from the user.
        doc_id:   Pinecone namespace — identifies the document.
        top_k:    Number of chunks to retrieve (default 5).
        filter:   Optional Pinecone metadata filter.
                  e.g. {"risks": {"$ne": "None"}}

    Returns:
        {
            "answer":          str,
            "relevant_pages":  list[int],
            "confidence":      str,
            "caveat":          str,
            "chunks_used":     int,
        }
    """
    if not question.strip():
        return {
            "answer": "Please provide a question.",
            "relevant_pages": [],
            "confidence": "low",
            "caveat": "Empty question.",
            "chunks_used": 0,
        }

    retriever = get_retriever(doc_id=doc_id, top_k=top_k, filter=filter)

    # Retrieve docs separately so we can extract page metadata
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

    # ── RAG chain ─────────────────────────────────────────────────────────
    chain = _PROMPT | _llm | StrOutputParser()
    raw_answer = chain.invoke({"context": context, "question": question})

    return {
        "answer": raw_answer.strip(),
        "relevant_pages": pages,
        "confidence": _parse_field(raw_answer, "confidence"),
        "caveat": _parse_field(raw_answer, "caveat"),
        "chunks_used": len(retrieved_docs),
    }


def _parse_field(text: str, field: str) -> str:
    """Extract a labeled field from the LLM response if present."""
    import re
    match = re.search(rf"{field}:\s*(.+)", text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else "Not specified"