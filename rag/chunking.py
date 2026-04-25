# rag/chunking.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.entities import extract_entities_clean
from core.risks import extract_risk_categories

# -------------------------
# Splitter config
# Tuned for legal prose:
# - 1000 chars fits one legal clause comfortably
# - 150 overlap preserves context across cuts
# -------------------------
_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_pages(pages_text: list[str], doc_id: str) -> list[dict]:
    """
    Split pages into chunks and enrich each chunk with entity metadata.

    Args:
        pages_text: Output of extraction.extract_text_pages() — one string per page.
        doc_id:     Unique document identifier (used as Pinecone namespace).

    Returns:
        List of dicts:
        {
            "text":     str,
            "metadata": {
                "doc_id":      str,
                "page":        int,
                "chunk_index": int,
                "preview":     str,       # first 80 chars of chunk
                "parties":     str,       # comma-joined or "Not specified"
                "dates":       str,
                "money":       str,
                "obligations": str,
            }
        }
    """
    chunks = []
    chunk_index = 0

    for page_num, page_text in enumerate(pages_text, start=1):
        if not page_text.strip():
            continue

        # ── Split ────────────────────────────────────────────────────────
        page_chunks = _SPLITTER.split_text(page_text)

        for chunk_text in page_chunks:
            if not chunk_text.strip():
                continue

            # ── Entity extraction per chunk ───────────────────────────────
            try:
                entities = extract_entities_clean(chunk_text)
            except Exception as e:
                print(f"[Chunking] Entity extraction failed on chunk {chunk_index}: {e}")
                entities = {
                    "Parties": ["Not specified"],
                    "Dates": ["Not specified"],
                    "Money/Penalties": ["Not specified"],
                    "Obligations": ["Not specified"],
                }
            
            
            risks = extract_risk_categories(chunk_text)
            # ── Flatten list fields to strings for Pinecone metadata ──────
            # Pinecone metadata values must be str / int / float / bool
            def _join(val):
                if isinstance(val, list):
                    return ", ".join(str(v) for v in val) if val else "Not specified"
                return str(val) if val else "Not specified"

            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "doc_id":      doc_id,
                    "page":        page_num,
                    "chunk_index": chunk_index,
                    "preview":     chunk_text[:80].replace("\n", " "),
                    "parties":     _join(entities.get("Parties")),
                    "dates":       _join(entities.get("Dates")),
                    "money":       _join(entities.get("Money/Penalties")),
                    "obligations": _join(entities.get("Obligations")),
                    "risks":       risks
                },
            })

            chunk_index += 1

    return chunks