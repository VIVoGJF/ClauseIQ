# rag/store.py
import os
import time
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# -------------------------
# Constants
# -------------------------
INDEX_NAME = "legal-docs"
  # text-embedding-3-small output dimension

# -------------------------
# Clients
# -------------------------
_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", 
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    output_dimensionality=1536
    )

_pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


# -------------------------
# Index management
# -------------------------
def _ensure_index():
    """Create Pinecone index if it doesn't exist."""
    existing = _pc.list_indexes().names()
    if INDEX_NAME not in existing:
        _pc.create_index(
            INDEX_NAME,
            dimension=3072,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not _pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)


# -------------------------
# Upsert
# -------------------------
def upsert_chunks(chunks: list[dict], doc_id: str):
    """
    Embed and upsert chunks into Pinecone under namespace=doc_id.

    Args:
        chunks: Output of chunking.chunk_pages() —
                list of {"text": str, "metadata": dict}
        doc_id: Used as Pinecone namespace for isolation per document.
    """
    _ensure_index()

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    PineconeVectorStore.from_texts(
        texts=texts,
        embedding=_embeddings,
        metadatas=metadatas,
        index_name=INDEX_NAME,
        namespace=doc_id,
    )


# -------------------------
# Retrieval
# -------------------------
def get_retriever(doc_id: str, top_k: int = 5, filter: dict = None):
    """
    Returns a LangChain retriever scoped to a single document namespace.

    Args:
        doc_id: Pinecone namespace to query within.
        top_k:  Number of chunks to retrieve.
        filter: Optional Pinecone metadata filter dict.
                e.g. {"risks": {"$ne": "None"}}

    Returns:
        LangChain VectorStoreRetriever
    """
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=_embeddings,
        namespace=doc_id,
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
    )

    search_kwargs = {"k": top_k}
    if filter:
        search_kwargs["filter"] = filter

    return vectorstore.as_retriever(search_kwargs=search_kwargs)


def delete_doc(doc_id: str):
    """
    Delete all vectors for a document by dropping its namespace.
    Useful for cleanup after session ends.
    """
    index = _pc.Index(INDEX_NAME)
    index.delete(delete_all=True, namespace=doc_id)