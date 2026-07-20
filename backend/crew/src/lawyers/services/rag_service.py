from typing import Dict, List, Optional
from .embedding_service import EmbeddingService
from .chroma_service import ChromaService
import logging

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        ChromaService.initialize()
        self.collection = ChromaService.get_collection()

    def search(self, queries: List[str], n_results: int = 5) -> List[Dict]:
        if not queries or self.collection is None:
            return []

        all_chunks: List[Dict] = []
        embeddings = EmbeddingService.encode(queries)
        logger.info(
            "Running %d semantic queries",
            len(queries),
        )
        try:
            
            results = self.collection.query(
                query_embeddings=embeddings,
                n_results=n_results,
                where={
                    "status": {
                        "$ne": "het_hieu_luc"
                    }
                }
            )
        except Exception:
            
            logger.exception(
                "Filtered search failed. Retry without filter."
            )
            results = self.collection.query(
                query_embeddings=embeddings,
                n_results=n_results,
            )
            
        logger.info(
            "Retrieved %d semantic result groups",
            len(results.get("documents", [])),
        )
        for i, query in enumerate(queries):

            documents = results.get("documents", [])
            metadatas = results.get("metadatas", [])
            ids_list = results.get("ids", [])
            distances_list = results.get("distances", [])

            if (
                i >= len(documents)
                or i >= len(metadatas)
                or i >= len(ids_list)
                or i >= len(distances_list)
            ):
                continue

            docs = documents[i]
            metas = metadatas[i]
            ids = ids_list[i]
            distances = distances_list[i]
            
            for rank, (doc, meta, _id, distance) in enumerate(
                zip(docs, metas, ids, distances),
                start=1,
            ):
                retrieval_score = (
                                1.0 - float(distance)
                                if distance is not None
                                else None
                            )   

                all_chunks.append({
                                "id": _id,
                                "text": doc,
                                "metadata": meta or {},
                                "source_query": query,
                                "retrieval_rank": rank,
                                "retrieval_score": retrieval_score,
                                "retrieval_source": "rag",
                })
           
        logger.info(
            "Retrieved %d chunks",
            len(all_chunks),
        )
        deduped = []
        seen = set()
        for chunk in all_chunks:
            meta = chunk["metadata"]
            cid = (
                meta.get("so_hieu"),
                meta.get("dieu"),
                meta.get("khoan"),
            )
            if cid == (None, None, None):
                cid = chunk.get("id") or chunk["text"][:100]
            if cid in seen:
                continue
            seen.add(cid)
            deduped.append(chunk)
        deduped.sort(
            key=lambda x: x.get("retrieval_score", 0.0),
            reverse=True,
        )
        return deduped
