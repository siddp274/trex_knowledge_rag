"""
Query pipeline orchestrator — ties all query steps together.

This is the single entry point for querying:
    pipeline = QueryPipeline(config)
    result = pipeline.query("What are the key strategic themes?")
    print(result["answer"])
"""
import logging

from config import TREXConfig
from retrieval.qdrant_retrieval import QdrantRetriever
from retrieval.context_assembler import ContextAssembler
from retrieval.answer_generator import AnswerGenerator

logger = logging.getLogger(__name__)


class QueryPipeline:
    def __init__(self, config: TREXConfig):
        self.config = config
        self.qdrant_retriever = QdrantRetriever(config)
        self.context_assembler = ContextAssembler(config)
        self.answer_generator = AnswerGenerator(config)

    def _retrieve_context(
        self,
        question: str,
        top_k: int | None,
        max_context_tokens: int,
    ) -> tuple[list[dict], dict]:
        """
        Shared retrieval logic for query() and query_stream().

        Returns:
            (qdrant_results, assembled_context)
        """
        qdrant_results = self.qdrant_retriever.search(question, top_k=top_k)
        assembled = self.context_assembler.assemble(
            qdrant_results=qdrant_results,
            max_context_tokens=max_context_tokens,
        )
        return qdrant_results, assembled

    def query(
        self,
        question: str,
        top_k: int | None = None,
        max_context_tokens: int = 3500,
    ) -> dict:
        logger.info(f"[Query] Question: {question}")
        qdrant_results, assembled = self._retrieve_context(question, top_k, max_context_tokens)
        result = self.answer_generator.generate(question, assembled)
        result["sources"] = list({r.get("source", "unknown") for r in qdrant_results})
        return result

    def query_stream(
        self,
        question: str,
        top_k: int | None = None,
        max_context_tokens: int = 3500,
    ):
        _, assembled = self._retrieve_context(question, top_k, max_context_tokens)
        yield from self.answer_generator.generate_stream(question, assembled)
