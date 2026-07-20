from pathlib import Path
import os
from typing import Optional
import chromadb
from chromadb.config import Settings


class ChromaService:
    _client: Optional[chromadb.PersistentClient] = None
    _collection: Optional[chromadb.Collection] = None
    _initialized: bool = False
    _collection_name: str = "luat_vn"

    @classmethod
    def initialize(cls, path: Optional[str] = None, collection_name: Optional[str] = None):
        if cls._initialized:
            return
        if collection_name:
            cls._collection_name = collection_name
        base_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
        chroma_dir = Path(path or os.getenv("CHROMA_PATH", base_dir / "chroma_db"))
        if not chroma_dir.is_absolute():
            chroma_dir = (base_dir / chroma_dir).resolve()
        cls._client = chromadb.PersistentClient(path=str(chroma_dir), settings=Settings(anonymized_telemetry=False))
        cls._collection = cls._client.get_collection(cls._collection_name)
        cls._initialized = True

    @classmethod
    def get_collection(cls) -> Optional[chromadb.Collection]:
        if not cls._initialized:
            cls.initialize()
        return cls._collection
