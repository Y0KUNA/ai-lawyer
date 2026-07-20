from crewai.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Type
import chromadb
from sentence_transformers import SentenceTransformer
import os
from pathlib import Path
from ..services.chroma_service import ChromaService

class RAGQueryInput(BaseModel):
    queries: List[str] = Field(..., description="List of queries to search the RAG DB")
    n_results: int = Field(default=5, description="Number of results per query")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class RAGQueryTool(BaseTool):
    """RAGQueryTool: accepts a list of queries, searches ChromaDB for each, and returns merged chunks."""
    name : str  = "RAGQueryTool"
    description : str = "Searches the ChromaDB RAG store for each query in `queries` and returns merged chunks list."
    args_schema: Type[BaseModel]  = RAGQueryInput

    _embed_model: Optional[SentenceTransformer] = None
    _client: Optional[chromadb.PersistentClient] = None
    _collection: Optional[chromadb.Collection] = None
    _is_ready: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_chroma()

    def _init_chroma(self):
        try:
            if self._embed_model is None:
                BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
                CHROMA_DIR = Path(os.getenv("CHROMA_PATH", BASE_DIR / "chroma_db"))
                if not CHROMA_DIR.is_absolute():
                    CHROMA_DIR = (BASE_DIR / CHROMA_DIR).resolve()
                self._embed_model = SentenceTransformer("BAAI/bge-m3")
                ChromaService.initialize()
                self._collection = ChromaService.get_collection()
                self._is_ready = True
        except Exception:
            self._is_ready = False

    def _run(self, queries: List[str], n_results: int = 5) -> List[dict]:
        """Return a list of chunks (dicts) merged across queries.

        Each chunk: {"id": ..., "text": ..., "metadata": {...}, "source_query": ...}
        """
        if not self._is_ready or self._collection is None or self._embed_model is None:
            return []

        merged = []
        try:
            for q in queries:
                if not q:
                    continue
                q_vec = self._embed_model.encode(q).tolist()
                results = self._collection.query(
                    query_embeddings=[q_vec],
                    n_results=min(n_results, 10),
                    where={"status": {"$ne": "het_hieu_luc"}},
                )

                docs = results.get("documents", [[]])[0]
                metas = results.get("metadatas", [[]])[0]
                ids = results.get("ids", [[]])[0]

                for doc, meta, _id in zip(docs, metas, ids):
                    merged.append({
                        "id": _id,
                        "text": doc,
                        "metadata": meta,
                        "source_query": q,
                    })

            # Optionally dedupe by id
            seen = set()
            deduped = []
            for c in merged:
                cid = c.get("id") or c.get("metadata", {}).get("so_hieu") or c.get("text")[:120]
                if cid in seen:
                    continue
                seen.add(cid)
                deduped.append(c)

            return deduped
        except Exception:
            return []
