from typing import Dict, List, Optional
from .evidence import Evidence
from .embedding_service import EmbeddingService
from .chroma_service import ChromaService
from .law_document_filter import LawDocumentFilter
import logging

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        ChromaService.initialize()
        self.collection = ChromaService.get_collection()

    def search(self, queries: List[str], n_results: int = 5) -> List[Dict]:
        if not queries or self.collection is None:
            return []

        all_chunks: List[Evidence] = []
        embeddings = EmbeddingService.encode(queries)
        logger.info(
            "Running %d semantic queries",
            len(queries),
        )
        try:
            results = self._query_with_law_filters(embeddings, n_results)
        except Exception:
            logger.exception("Semantic search failed")
            return []

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
                    else 0.0
                )

                chunk_dict = {
                    "id": _id,
                    "text": doc,
                    "metadata": meta or {},
                    "source_query": query,
                    "retrieval_rank": rank,
                    "retrieval_score": retrieval_score,
                    "retrieval_source": "rag",
                }
                all_chunks.append(Evidence.from_dict(chunk_dict, default_source="rag"))

        logger.info(
            "Retrieved %d chunks",
            len(all_chunks),
        )
        deduped = []
        seen = set()
        for evidence in all_chunks:
            meta = evidence.metadata or {}
            # Field names match what the chunker actually writes to Chroma:
            # law_code / article_number / clause_number (NOT so_hieu/dieu/khoan).
            cid = (
                meta.get("law_code"),
                meta.get("article_number"),
                meta.get("clause_number"),
            )
            if cid == (None, None, None):
                cid = evidence.source or evidence.text[:100]
            if cid in seen:
                continue
            seen.add(cid)
            deduped.append(evidence)
        deduped.sort(
            key=lambda x: x.score,
            reverse=True,
        )
        filtered = LawDocumentFilter.filter_chunks(deduped)
        if len(filtered) < len(deduped):
            logger.info(
                "Filtered out %d non-law/case-law chunks from RAG results",
                len(deduped) - len(filtered),
            )
        print("RAG results: ", filtered)
        return filtered

    def _query_with_law_filters(self, embeddings: List[List[float]], n_results: int) -> Dict:
        """
        Try each `where` clause from LawDocumentFilter.chroma_where_clauses(),
        from strictest to loosest. A clause is only accepted if it actually
        returned at least one document for at least one query; otherwise we
        fall through to the next (looser) clause. This fixes a bug where the
        loop returned the FIRST clause's result as long as it didn't raise
        an exception - even when that result was empty - so the intended
        `where=None` fallback was never reached.
        """
        last_error = None
        last_result = None

        for where in LawDocumentFilter.chroma_where_clauses():
            kwargs = {
                "query_embeddings": embeddings,
                "n_results": n_results,
            }
            if where is not None:
                kwargs["where"] = where

            try:
                result = self.collection.query(**kwargs)
            except Exception as exc:
                last_error = exc
                logger.warning("Chroma query failed with where=%s: %s", where, exc)
                continue

            last_result = result
            documents = result.get("documents", [])
            has_any_hit = any(len(group) > 0 for group in documents)

            if has_any_hit:
                if where is not None:
                    logger.info("Chroma query matched using where=%s", where)
                return result

            logger.info(
                "Chroma query with where=%s returned 0 documents, trying next filter",
                where,
            )

        if last_result is not None:
            # Every clause ran without error but all were empty - return the
            # last (loosest / where=None) result rather than raising.
            return last_result

        if last_error:
            raise last_error

        return self.collection.query(
            query_embeddings=embeddings,
            n_results=n_results,
        )