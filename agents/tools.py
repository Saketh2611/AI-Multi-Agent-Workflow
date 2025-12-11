from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
import os
import json
from typing import List, Dict, Optional

# PDF + Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# FAISS + POSTGRES
try:
    import faiss
    import numpy as np
except ImportError:
    faiss = None
    np = None

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    psycopg2 = None

# Global Store Variable
_global_store = None

from dotenv import load_dotenv
load_dotenv()

# -------------------------------------------------------
# HELPER: AUTO-INIT LOGIC
# -------------------------------------------------------
def _initialize_store_internal(dim: int = 768):
    """
    Internal helper to initialize the store without going through the tool interface.
    """
    global _global_store
    pg_conn_str = os.getenv("DATABASE_URL")
    
    if not pg_conn_str:
        raise ValueError("DATABASE_URL environment variable is not set.")
        
    if faiss is None or np is None or psycopg2 is None:
        raise ImportError("Missing dependencies: faiss-cpu, numpy, or psycopg2")

    print(f"⚙️ Auto-initializing FAISS with dim={dim}...")
    _global_store = FaissPostgresStore(dim, pg_conn_str)
    return _global_store

# -------------------------------------------------------
# WEB SEARCH TOOL
# -------------------------------------------------------
@tool
def web_search(query: str):
    """
    Searches the web using Tavily Search and returns summarized text results.
    """
    try:
        search_tool = TavilySearchResults(max_results=3)
        results = search_tool.invoke({"query": query})

        output = []
        for res in results:
            output.append(f"Source: {res.get('url')}\nContent: {res.get('content')}\n")

        return "\n".join(output)

    except Exception as e:
        return f"Error executing search: {str(e)}"

# -------------------------------------------------------
# FAISS + POSTGRES STORE CLASS
# -------------------------------------------------------
class FaissPostgresStore:
    def __init__(self, dim: int, pg_conn_str: str):
        self.dim = dim
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))

        self.conn = psycopg2.connect(pg_conn_str)
        self.conn.autocommit = True
        self._ensure_table()

    def _ensure_table(self):
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS vector_metadata (
                    id BIGINT PRIMARY KEY,
                    metadata JSONB
                );
            """)

    def add(self, ids, embeddings, metadatas=None):
        arr = np.asarray(embeddings, dtype=np.float32)
        id_arr = np.asarray(ids, dtype=np.int64)
        self.index.add_with_ids(arr, id_arr)

        if metadatas:
            with self.conn.cursor() as cur:
                for i, md in zip(ids, metadatas):
                    cur.execute("""
                        INSERT INTO vector_metadata (id, metadata)
                        VALUES (%s, %s)
                        ON CONFLICT (id) DO UPDATE SET metadata = EXCLUDED.metadata;
                    """, (i, json.dumps(md)))

    def search(self, query_embedding, k):
        q = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        distances, ids = self.index.search(q, k)

        ids_list = [int(v) for v in ids[0] if v != -1]
        distances_list = distances[0][:len(ids_list)]

        if not ids_list:
            return []

        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, metadata FROM vector_metadata WHERE id = ANY(%s);",
                (ids_list,)
            )
            rows = cur.fetchall()

        meta_map = {int(r["id"]): r["metadata"] for r in rows}

        return [
            {"id": _id, "distance": float(distances_list[i]), "metadata": meta_map.get(_id)}
            for i, _id in enumerate(ids_list)
        ]

# -------------------------------------------------------
# INIT FAISS TOOL (FIXED DOCSTRING)
# -------------------------------------------------------
@tool
def init_faiss(dim: int = 768):
    """
    Manually initializes the FAISS + Postgres vector store.
    """
    try:
        _initialize_store_internal(dim)
        return f"FAISS initialized with dimension={dim}"
    except Exception as e:
        return f"FAISS initialization error: {e}"

# -------------------------------------------------------
# FAISS ADD TOOL
# -------------------------------------------------------
@tool
def faiss_add(ids: List[int], embeddings: List[List[float]], metadatas: Optional[List[Dict]] = None):
    """
    Adds embeddings + metadata into the FAISS store.
    """
    if _global_store is None:
        try:
            _initialize_store_internal(len(embeddings[0]))
        except Exception as e:
            return f"Failed to auto-initialize store: {e}"

    try:
        _global_store.add(ids, embeddings, metadatas)
        return f"Added {len(ids)} vectors."
    except Exception as e:
        return f"Vector add error: {e}"

# -------------------------------------------------------
# FAISS SEARCH TOOL
# -------------------------------------------------------
@tool
def faiss_search(query_embedding: List[float], k: int = 5):
    """
    Performs vector similarity search using FAISS.
    """
    if _global_store is None:
        try:
            _initialize_store_internal(len(query_embedding))
        except Exception as e:
            return f"Failed to auto-initialize store: {e}"

    try:
        return _global_store.search(query_embedding, k)
    except Exception as e:
        return f"Vector search error: {e}"

# -------------------------------------------------------
# PDF SEARCH TOOL (ASYNC + DOCSTRING)
# -------------------------------------------------------
@tool
async def pdf_search(query: str):
    """
    Searches inside embedded PDF documents using vector similarity.
    Requires FAISS to be initialized and PDFs already processed.
    """
    # 1. AUTO-INIT CHECK
    if _global_store is None:
        try:
            # Google GenAI embeddings are 768 dimensions
            _initialize_store_internal(dim=768)
        except Exception as e:
            return f"System Error: Could not initialize PDF store. {e}"

    try:
        # Use Google GenAI Embeddings
        embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # FIX: Await the async version of embed_query
        q_emb = await embedder.aembed_query(query)

        # 2. Perform Search (Synchronous call to store is fine here)
        results = _global_store.search(q_emb, k=5)

        if not results:
            return "No matching PDF segments found."

        outputs = []
        for r in results:
            meta = r["metadata"]
            snippet = meta.get("text", "")[:300]
            source = meta.get("source", "PDF")
            outputs.append(f"📄 {source}\n{snippet}\n(distance={r['distance']:.4f})")

        return "\n\n".join(outputs)

    except Exception as e:
        return f"PDF search error: {e}"