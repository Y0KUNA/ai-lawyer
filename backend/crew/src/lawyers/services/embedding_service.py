from pathlib import Path
import os
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    _models: Dict[str, SentenceTransformer] = {}
    _default_model_name = os.getenv(
        "EMBEDDING_MODEL",
        "BAAI/bge-m3"
    )

    @classmethod
    def initialize(cls, model_name: Optional[str] = None):
        name = model_name or cls._default_model_name
        if name in cls._models:
            return
        cls._models[name] = SentenceTransformer(name)

    @classmethod
    def encode(cls, texts: List[str], model_name: Optional[str] = None) -> List[List[float]]:
        name = model_name or cls._default_model_name
        if name not in cls._models:
            cls.initialize(name)
        return cls._models[name].encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    @classmethod
    def encode_single(cls, text: str, model_name: Optional[str] = None) -> List[float]:
        name = model_name or cls._default_model_name
        if name not in cls._models:
            cls.initialize(name)
        return cls._models[name].encode(
            text,
            normalize_embeddings=True,
        )
    @classmethod
    def preload(cls):
        cls.initialize("BAAI/bge-m3")