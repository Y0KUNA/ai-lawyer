from crewai.tools import BaseTool
from typing import Type, Optional
from pydantic import BaseModel, Field, ConfigDict
import chromadb
from sentence_transformers import SentenceTransformer
import os
from pathlib import Path


class RAGQueryInput(BaseModel):
    """Input schema for RAG Query Tool."""
    query: str = Field(
        ..., 
        description="Legal question or keywords to search in Vietnamese law database (e.g., 'civil contract breach', 'labor law violations')"
    )
    n_results: int = Field(
        default=5,
        description="Number of results to return (1-10)"
    )


class RAGQueryTool(BaseTool):
    """Tool to query Vietnamese legal database (ChromaDB RAG)"""
    name: str = "Vietnamese Legal Database (RAG)"
    description: str = (
        "Searches the Vietnamese legal database (ChromaDB) for applicable laws, decrees, ordinances, and legal articles. "
        "Use this tool FIRST before external search. Input should be a legal question or keywords related to the case. "
        "Returns the most relevant legal provisions from the database."
    )
    args_schema: Type[BaseModel] = RAGQueryInput
    
    # Pydantic config to allow arbitrary attributes
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='allow')
    
    # Initialize ChromaDB components at class level
    _embed_model: Optional[SentenceTransformer] = None
    _client: Optional[chromadb.PersistentClient] = None
    _collection: Optional[chromadb.Collection] = None
    _is_ready: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_chroma()

    def _init_chroma(self):
        """Initialize ChromaDB connection"""
        try:
            if self._embed_model is None:
                BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
                
                CHROMA_DIR = Path(os.getenv("CHROMA_PATH", BASE_DIR / "chroma_db"))
                print(CHROMA_DIR)
                if not CHROMA_DIR.is_absolute():
                    CHROMA_DIR = (BASE_DIR / CHROMA_DIR).resolve()
                
                self._embed_model = SentenceTransformer("BAAI/bge-m3")
                self._client = chromadb.PersistentClient(path=str(CHROMA_DIR))
                self._collection = self._client.get_collection("luat_vn")
                print("Collection size:", self._collection.count())
                self._is_ready = True
                print("✓ RAG tool initialized successfully")
        except Exception as e:
            print(f"⚠ RAG tool initialization error: {e}")
            self._is_ready = False

    def _run(self, query: str, n_results: int = 5) -> str:
        """Query ChromaDB for relevant laws"""
        if not self._is_ready or self._collection is None or self._embed_model is None:
            return "Vietnamese legal database is not available. Please try external search."
        
        try:
            # Encode query
            q_vec = self._embed_model.encode(query).tolist()
            
            # Query with filter (prioritize in-force laws)
            results = self._collection.query(
                query_embeddings=[q_vec],
                n_results=min(n_results, 10),
                where={"status": {"$ne": "het_hieu_luc"}},  # exclude expired laws
            )
            
            # Fallback if no results with filter
            if len(results["documents"][0]) < 2:
                results = self._collection.query(
                    query_embeddings=[q_vec],
                    n_results=min(n_results, 10)
                )
            
            # Format results
            output = "# Legal Provisions Found\n\n"
            for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]), 1):
                output += f"## Result {i}\n"
                output += f"**Law**: {meta.get('so_hieu', 'N/A')} - {meta.get('ten_luat', 'N/A')}\n"
                output += f"**Type**: {meta.get('loai_van_ban', 'N/A')}\n"
                output += f"**Section**: {meta.get('heading', 'N/A')}\n"
                output += f"**Status**: {meta.get('status', 'N/A')}\n"
                output += f"**Content**:\n{doc}\n\n"
                output += "---\n\n"
            
            return output if output != "# Legal Provisions Found\n\n" else "No relevant legal provisions found in database."
        except Exception as e:
            return f"Error querying legal database: {str(e)}"


