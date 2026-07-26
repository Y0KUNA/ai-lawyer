import unittest

from backend.crew.src.lawyers.services.evidence import Evidence
from backend.crew.src.lawyers.services.retrieval_pipeline_service import RetrievalPipelineService
from backend.crew.src.lawyers.services.reranker_service import RerankerService


class RetrievalPipelineServiceTests(unittest.TestCase):
    def test_compact_chunks_accepts_evidence_objects(self):
        service = RetrievalPipelineService.__new__(RetrievalPipelineService)

        chunk = Evidence(
            text="Điều 10. Người lao động có quyền ...",
            source="test",
            metadata={"law_code": "LUAT-1", "heading": "Điều 10"},
        )

        compacted = service._compact_chunks([chunk], "Điều 10 về quyền lao động")

        self.assertEqual(len(compacted), 1)
        self.assertIsInstance(compacted[0], Evidence)
        self.assertEqual(compacted[0].metadata.get("law_code"), "LUAT-1")
        self.assertIn("Điều 10", compacted[0].text)

    def test_reranker_accepts_evidence_objects(self):
        service = RerankerService.__new__(RerankerService)
        service.model = type("MockModel", (), {"predict": lambda self, pairs: [0.42]})()

        chunk = Evidence(
            text="Điều 10 về quyền lao động",
            source="test",
            metadata={"law_code": "LUAT-1", "heading": "Điều 10"},
        )

        reranked = service.rerank([chunk], "quyền lao động", top_k=1)

        self.assertEqual(len(reranked), 1)
        self.assertIsInstance(reranked[0], Evidence)
        self.assertEqual(reranked[0].text, "Điều 10 về quyền lao động")
        self.assertEqual(reranked[0].metadata.get("law_code"), "LUAT-1")

    def test_evidence_from_rag_metadata(self):
        chunk = {
            "text": "Điều 10. Người lao động có quyền ...",
            "metadata": {
                "law_code": "LUAT-10",
                "article_number": 10,
                "clause_number": 1,
                "point": "a",
                "law_name": "Bộ luật Lao động",
                "article_title": "Quyền lao động",
            },
        }

        evidence = Evidence.from_dict(chunk)

        self.assertEqual(evidence.law_id, "LUAT-10")
        self.assertEqual(evidence.article, 10)
        self.assertEqual(evidence.clause, "1")
        self.assertEqual(evidence.point, "a")


if __name__ == "__main__":
    unittest.main()
