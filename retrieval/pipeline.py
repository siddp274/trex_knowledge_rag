"""
Query pipeline orchestrator — ties all 5 query steps together.

This is the single entry point for querying:
    pipeline = QueryPipeline(config, neo4j_manager)
    result = pipeline.query("What are the key strategic themes?")
    print(result["answer"])
"""
import logging
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
print(f"Current dir: {CURRENT_DIR} and project root: {PROJECT_ROOT}")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config import TREXConfig
from store.graph_store import Neo4jManager
from retrieval.qdrant_retrieval import QdrantRetriever
from retrieval.entity_linker import QueryEntityLinker
from retrieval.graph_context import GraphContextBuilder
from retrieval.context_assembler import ContextAssembler
from retrieval.answer_generator import AnswerGenerator

logger = logging.getLogger(__name__)

class QueryPipeline:
    def __init__(self, config: TREXConfig, neo4j_manager: Neo4jManager | None = None):
        self.config = config
        self.qdrant_retriever = QdrantRetriever(config)
        self.context_assembler = ContextAssembler(config)
        self.answer_generator = AnswerGenerator(config)

        # Only initialize graph components if enabled AND Neo4j is available
        if config.enable_graph and neo4j_manager is not None:
            self.entity_linker = QueryEntityLinker(config, neo4j_manager)
            self.graph_context_builder = GraphContextBuilder(config, neo4j_manager)
            self.graph_enabled = True
        else:
            self.entity_linker = None
            self.graph_context_builder = None
            self.graph_enabled = False

    def query(
        self,
        question: str,
        top_k: int | None = None,
        max_context_tokens: int = 3500,
    ) -> dict:

        logger.info(f"[Query] Question: {question}")

        # Step 1: Hybrid retrieval from Qdrant (always runs)
        qdrant_results = self.qdrant_retriever.search(question, top_k=top_k)

        # Steps 2-3: Graph context (only if enabled)
        if self.graph_enabled:
            matched_entities = self.entity_linker.extract_and_link(question)
            graph_context = self.graph_context_builder.build(matched_entities)
        else:
            matched_entities = []
            graph_context = {"context_text": "", "referenced_chunk_ids": set()}

        # Step 4: Assemble context
        assembled = self.context_assembler.assemble(
            qdrant_results=qdrant_results,
            graph_context=graph_context,
            max_context_tokens=max_context_tokens,
        )

        # Step 5: Generate answer
        result = self.answer_generator.generate(question, assembled)

        result["sources"] = list({r.get("source", "unknown") for r in qdrant_results})
        result["entities_found"] = [e["name"] for e in matched_entities]
        result["graph_enabled"] = self.graph_enabled

        return result
    
    def query_stream(
        self,
        question: str,
        top_k: int | None = None,
        max_context_tokens: int = 3500,
        ):
        qdrant_results = self.qdrant_retriever.search(question, top_k=top_k)

        if self.graph_enabled:
            matched_entities = self.entity_linker.extract_and_link(question)
            graph_context = self.graph_context_builder.build(matched_entities)
        else:
            graph_context = {"context_text": "", "referenced_chunk_ids": set()}

        assembled = self.context_assembler.assemble(
            qdrant_results=qdrant_results,
            graph_context=graph_context,
            max_context_tokens=max_context_tokens,
        )

        yield from self.answer_generator.generate_stream(question, assembled)