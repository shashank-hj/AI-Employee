from unittest.mock import MagicMock

import pytest

from rag.schemas.documents import SearchResult


def _result(doc_id, title, content, score=0.9):
    return SearchResult(
        chunk_id=f"chunk-{doc_id}",
        document_id=doc_id,
        document_title=title,
        chunk_index=0,
        content=content,
        score=score,
    )


class _FakeLLM:
    def __init__(self, content="Answer text"):
        self._content = content
        self._calls = []

    async def generate(self, system_prompt, user_message):
        self._calls.append((system_prompt, user_message))
        return MagicMock(content=self._content, model="fake")


class TestAnswerGenerator:
    @pytest.mark.asyncio
    async def test_generate_returns_answer_and_deduplicated_sources(self):
        from rag.services.answer_generator import AnswerGenerator

        llm = _FakeLLM(content="According to 'r1.pdf', the refund policy is 30 days.")
        gen = AnswerGenerator(llm)
        results = [
            _result("d1", "r1.pdf", "Refund policy is 30 days."),
            _result("d1", "r1.pdf", "Another chunk from same doc."),
            _result("d2", "r2.pdf", "Shipping takes 5 business days."),
        ]

        answer = await gen.generate("What is the refund policy?", results)

        assert answer.answer == "According to 'r1.pdf', the refund policy is 30 days."
        assert [s.document_id for s in answer.sources] == ["d1", "d2"]
        assert answer.sources[0].document_title == "r1.pdf"

    @pytest.mark.asyncio
    async def test_prompt_labels_chunks_with_document_title(self):
        from rag.services.answer_generator import AnswerGenerator

        llm = _FakeLLM()
        gen = AnswerGenerator(llm)
        await gen.generate("policy?", [_result("d1", "r1.pdf", "Some content.")])

        system_prompt, user_message = llm._calls[0]
        assert "r1.pdf" in user_message
        assert "[doc:1]" in user_message
        assert "policy?" in user_message

    @pytest.mark.asyncio
    async def test_generate_with_no_results_returns_no_answer(self):
        from rag.services.answer_generator import AnswerGenerator

        gen = AnswerGenerator(_FakeLLM())
        answer = await gen.generate("anything", [])
        assert answer.sources == []
        assert "could not find" in answer.answer.lower()

    @pytest.mark.asyncio
    async def test_same_title_documents_collapse_in_sources(self):
        from rag.services.answer_generator import AnswerGenerator

        llm = _FakeLLM()
        gen = AnswerGenerator(llm)
        results = [
            _result("d1", "ind_geo.pdf", "Peninsular plateau is ancient rock.", score=0.9),
            _result("d2", "ind_geo.pdf", "Peninsular plateau is ancient rock.", score=0.85),
            _result("d3", "other.pdf", "Rivers of India.", score=0.6),
        ]

        answer = await gen.generate("plateau?", results)

        assert [s.document_title for s in answer.sources] == ["ind_geo.pdf", "other.pdf"]
        assert answer.sources[0].document_id == "d1"  # highest score wins

    @pytest.mark.asyncio
    async def test_duplicate_content_is_not_fed_to_llm(self):
        from rag.services.answer_generator import AnswerGenerator

        llm = _FakeLLM()
        gen = AnswerGenerator(llm)
        results = [
            _result("d1", "ind_geo.pdf", "Deccan plateau covers central India.", score=0.9),
            _result("d2", "ind_geo.pdf", "Deccan plateau covers central India.", score=0.8),
        ]

        await gen.generate("deccan?", results)

        _, user_message = llm._calls[0]
        assert user_message.count("Deccan plateau covers central India.") == 1
