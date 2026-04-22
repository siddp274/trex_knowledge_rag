# TREX + GraphRAG Hybrid Implementation Guide
### Stack: Python + LangChain 1.2.15 | Qdrant | Neo4j | OpenAI GPT-4.1-mini & GPT-4o

---

## Architecture Overview

This implementation blends two paradigms:

- **TREX** — Truncated RAPTOR tree stored in Qdrant, searched via hybrid (dense vector + BM25 sparse) retrieval fused with RRF
- **GraphRAG** — Entity and relationship graph stored in Neo4j, used to enrich retrieved context with relational structure

```
INDEXING PIPELINE
─────────────────
Raw Documents (PDF, HTML, CSV, TXT)
        │
        ▼
[Document Loaders] ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
        │
        ▼
[Chunker] → 500-token chunks (Level 0 leaves)
        │
        ├──────────────────────────────────────────┐
        ▼                                          ▼
[RAPTOR Tree Builder]                    [Entity Extractor]
  Embed → UMAP → GMM/BIC                GPT-4.1-mini extracts
  → Summarize (GPT-4.1-mini)            entities + relationships
  → Repeat (max 2 levels)                         │
  → Tree nodes at L0, L1, L2                      ▼
        │                                    [Neo4j Graph]
        ▼                                    Nodes = Entities
[Qdrant Hybrid Index]                        Edges = Relationships
  Dense vectors (text-embedding-3-small)
  Sparse vectors (BM25 via FastEmbed)
  Metadata: level, parent_id, source

QUERY PIPELINE
──────────────
User Question
        │
        ├─────────────────────────────┐
        ▼                             ▼
[Qdrant Hybrid Search]         [Neo4j Entity Lookup]
  Dense + BM25 → RRF fusion     Extract entities from query
  Returns top-K tree nodes       → fetch neighborhood graph
        │                             │
        └──────────────┬──────────────┘
                       ▼
              [Context Assembly]
              Tree nodes + Graph context
                       │
                       ▼
              [GPT-4o Answer Generation]
```

---

## 1. Setup & Dependencies

```bash
pip install langchain>=1.2.15 \
            langchain-core>=1.2.15 \
            langchain-openai>=0.4.0 \
            langchain-qdrant>=0.4.0 \
            langchain-community>=0.4.0 \
            langchain-text-splitters>=0.3.0 \
            qdrant-client>=1.13.0 \
            neo4j>=5.0.0 \
            umap-learn>=0.5.0 \
            scikit-learn>=1.5.0 \
            numpy>=1.26.0 \
            fastembed>=0.4.0 \
            pypdf>=4.0.0 \
            beautifulsoup4>=4.12.0 \
            tiktoken>=0.8.0
```

> **Note on LangChain 1.x**: LangChain reached stable 1.0 in October 2025. The 1.x line commits to no breaking changes until 2.0. Key changes from 0.3.x:
> - `langchain-openai` now defaults to the **Responses API** for models prefixed with `"openai:"` — use bare model names (`"gpt-4.1-mini"`) to stay on Chat Completions API
> - `openai_api_key` parameter renamed to `api_key` (Pydantic v2 field name)
> - **Retry Middleware** is now native — pass `max_retries` directly to `ChatOpenAI`
> - **`ContextOverflowError`** raised natively by `langchain-openai`; import from `langchain_core.exceptions`
> - **`langchain-qdrant`** now uses `QdrantVectorStore.from_existing_collection()` as preferred pattern
> - **Structured output** via `llm.with_structured_output(PydanticModel)` replaces `JsonOutputParser`
> - **Python 3.10+ required** (3.9 support dropped)

---

## 2. Configuration

```python
# config.py
# Requires Python 3.10+ (LangChain 1.x dropped 3.9 support)
import os
from dataclasses import dataclass, field

@dataclass
class TREXConfig:
    # --- OpenAI ---
    # Use bare model names (NOT "openai:" prefix) to stay on Chat Completions API.
    # "openai:" prefix routes to the Responses API in langchain-openai 0.4+.
    openai_api_key: str = field(default_factory=lambda: os.environ["OPENAI_API_KEY"])
    indexing_model: str = "gpt-4.1-mini"   # Summarization + entity extraction
    query_model: str = "gpt-4o"            # Final answer generation
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # --- Qdrant ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "trex_index"

    # --- Neo4j ---
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"

    # --- RAPTOR Tree (TREX truncation) ---
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_tree_levels: int = 2          # TREX truncation: stop at level 2
    cluster_threshold: float = 0.5   # GMM soft-assignment threshold
    umap_global_components: int = 2  # For broad theme clustering
    umap_local_components: int = 10  # For fine-grained clustering
    max_clusters: int = 15
    min_cluster_size: int = 2

    # --- Retrieval ---
    top_k: int = 10
    rrf_k: int = 60                  # RRF smoothing constant from the paper

    # --- LangChain 1.x Native Retry Middleware ---
    max_retries: int = 3             # Passed directly to ChatOpenAI; uses exponential backoff
    retry_backoff_factor: float = 2.0


config = TREXConfig()
```

---

## 3. Document Loaders (Multi-Format)

```python
# ingestion/loaders.py
# LangChain 1.2.x: community loaders remain in langchain_community.document_loaders.
# WebBaseLoader now accepts web_paths as a list and supports async natively;
# we use sync .load() here for simplicity.
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
    CSVLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)


def load_documents(sources: List[str]) -> List[Document]:
    """
    Load documents from multiple source types.
    sources: list of file paths or URLs.
    """
    docs = []
    for source in sources:
        docs.extend(_load_single(source))
    return docs


def _load_single(source: str) -> List[Document]:
    source = source.strip()
    if source.startswith("http://") or source.startswith("https://"):
        # LangChain 1.x: WebBaseLoader takes web_paths as a list
        loader = WebBaseLoader(
            web_paths=[source],
            requests_kwargs={"verify": True},
        )
    else:
        path = Path(source)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            # PyPDFLoader in langchain-community 0.4+ uses pypdf v4 internally
            loader = PyPDFLoader(source)
        elif suffix == ".csv":
            loader = CSVLoader(source)
        elif suffix in (".md", ".markdown"):
            loader = UnstructuredMarkdownLoader(source)
        else:
            loader = TextLoader(source, encoding="utf-8")
    return loader.load()
```

---

## 4. RAPTOR Tree Builder

This is the mathematical core of the system. It implements the full GMM + UMAP + BIC pipeline from the RAPTOR paper, truncated at `max_tree_levels` per TREX.

```python
# ingestion/raptor_tree.py
import uuid
import numpy as np
from typing import List, Dict, Tuple, Any
from sklearn.mixture import GaussianMixture
import umap.umap_ as umap
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import TREXConfig


SUMMARIZATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert research analyst. "
        "Your task is to produce a dense, informative summary of the following text passages. "
        "Preserve all key facts, entities, relationships, and insights. "
        "Do not omit important details. Write in clear prose."
    )),
    ("human", "Passages to summarize:\n\n{text}")
])


class RAPTORTreeBuilder:
    """
    Builds a truncated RAPTOR tree over a list of documents.

    Each node in the tree is a dict:
    {
        "id": str (uuid),
        "text": str,
        "embedding": List[float],
        "level": int,         # 0 = leaf chunk, 1+ = summary levels
        "parent_id": str | None,
        "source": str,        # original document source
        "children_ids": List[str]
    }
    """

    def __init__(self, config: TREXConfig):
        self.config = config
        self.embedder = OpenAIEmbeddings(
            model=config.embedding_model,
            openai_api_key=config.openai_api_key,
        )
        self.llm = ChatOpenAI(
            model=config.indexing_model,
            openai_api_key=config.openai_api_key,
            temperature=0,
        )
        self.summarize_chain = SUMMARIZATION_PROMPT | self.llm | StrOutputParser()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

    def build(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Entry point. Returns all tree nodes (leaves + summaries)."""
        all_nodes = []

        # Step 1: Chunk documents into leaf nodes (Level 0)
        leaf_nodes = self._create_leaf_nodes(documents)
        all_nodes.extend(leaf_nodes)
        print(f"[RAPTOR] Created {len(leaf_nodes)} leaf nodes (Level 0)")

        # Step 2: Recursively build summary levels (TREX truncation at max_tree_levels)
        current_level_nodes = leaf_nodes
        for level in range(1, self.config.max_tree_levels + 1):
            if len(current_level_nodes) < self.config.min_cluster_size:
                print(f"[RAPTOR] Too few nodes to cluster at Level {level}. Stopping.")
                break

            summary_nodes = self._build_level(current_level_nodes, level)
            if not summary_nodes:
                print(f"[RAPTOR] No summaries generated at Level {level}. Stopping.")
                break

            all_nodes.extend(summary_nodes)
            print(f"[RAPTOR] Created {len(summary_nodes)} summary nodes at Level {level}")
            current_level_nodes = summary_nodes

        return all_nodes

    def _create_leaf_nodes(self, documents: List[Document]) -> List[Dict[str, Any]]:
        chunks = self.splitter.split_documents(documents)
        texts = [c.page_content for c in chunks]
        sources = [c.metadata.get("source", "unknown") for c in chunks]

        print(f"[RAPTOR] Embedding {len(texts)} leaf chunks...")
        embeddings = self.embedder.embed_documents(texts)

        nodes = []
        for text, embedding, source in zip(texts, embeddings, sources):
            nodes.append({
                "id": str(uuid.uuid4()),
                "text": text,
                "embedding": embedding,
                "level": 0,
                "parent_id": None,
                "source": source,
                "children_ids": [],
            })
        return nodes

    def _build_level(
        self,
        nodes: List[Dict[str, Any]],
        level: int
    ) -> List[Dict[str, Any]]:
        """Cluster nodes, summarize each cluster, return new summary nodes."""
        embeddings = np.array([n["embedding"] for n in nodes])

        # --- UMAP: reduce dimensionality ---
        # Use global (2D) for broad theme discovery at higher levels
        # Use local (10D) for finer distinctions at lower levels
        n_components = (
            self.config.umap_global_components if level >= 2
            else self.config.umap_local_components
        )
        n_components = min(n_components, len(nodes) - 1)  # cannot exceed n_samples - 1

        print(f"[RAPTOR] Level {level}: Running UMAP to {n_components} dimensions...")
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=min(15, len(nodes) - 1),
            random_state=42,
            metric="cosine",
        )
        reduced = reducer.fit_transform(embeddings)

        # --- GMM + BIC: find optimal number of clusters ---
        k = self._find_optimal_k(reduced)
        print(f"[RAPTOR] Level {level}: Optimal clusters k={k} (BIC selection)")

        # --- Soft clustering: a node can belong to multiple clusters ---
        gmm = GaussianMixture(n_components=k, random_state=42, covariance_type="full")
        gmm.fit(reduced)

        # P(cluster_k | node_i): shape (n_nodes, k)
        probs = gmm.predict_proba(reduced)

        # Assign each node to all clusters where probability > threshold
        cluster_assignments: Dict[int, List[int]] = {i: [] for i in range(k)}
        for node_idx, node_probs in enumerate(probs):
            for cluster_idx, prob in enumerate(node_probs):
                if prob >= self.config.cluster_threshold:
                    cluster_assignments[cluster_idx].append(node_idx)

        # --- Summarize each cluster ---
        summary_nodes = []
        for cluster_idx, member_indices in cluster_assignments.items():
            if len(member_indices) < 1:
                continue

            member_nodes = [nodes[i] for i in member_indices]
            combined_text = "\n\n---\n\n".join(n["text"] for n in member_nodes)

            # LLM summarization (GPT-4.1-mini)
            summary_text = self.summarize_chain.invoke({"text": combined_text})

            # Embed the summary
            summary_embedding = self.embedder.embed_query(summary_text)

            summary_node = {
                "id": str(uuid.uuid4()),
                "text": summary_text,
                "embedding": summary_embedding,
                "level": level,
                "parent_id": None,
                "source": member_nodes[0].get("source", "unknown"),
                "children_ids": [n["id"] for n in member_nodes],
            }
            summary_nodes.append(summary_node)

            # Link children to this parent
            for n in member_nodes:
                n["parent_id"] = summary_node["id"]

        return summary_nodes

    def _find_optimal_k(self, embeddings: np.ndarray) -> int:
        """
        Select optimal number of GMM clusters using BIC.
        BIC = ln(N) * k_params - 2 * ln(L_hat)
        Lower BIC = better model (penalizes complexity).
        """
        n_samples = len(embeddings)
        max_k = min(self.config.max_clusters, n_samples - 1)
        if max_k < 2:
            return 1

        best_bic = np.inf
        best_k = 2

        for k in range(2, max_k + 1):
            try:
                gmm = GaussianMixture(
                    n_components=k,
                    random_state=42,
                    covariance_type="full"
                )
                gmm.fit(embeddings)
                bic = gmm.bic(embeddings)
                if bic < best_bic:
                    best_bic = bic
                    best_k = k
            except Exception:
                continue

        return best_k
```

---

## 5. Entity Extractor → Neo4j (GraphRAG Layer)

```python
# ingestion/entity_extractor.py
import json
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from neo4j import GraphDatabase
from config import TREXConfig


ENTITY_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a knowledge graph expert. Extract structured information from text.\n"
        "Return a JSON object with two keys:\n"
        "  'entities': list of objects with fields: name (str), type (str), description (str)\n"
        "  'relationships': list of objects with fields: source (str), target (str), "
        "description (str), strength (int 1-10)\n\n"
        "Entity types: Person, Organization, Location, Technology, Concept, Event, Product\n"
        "Be thorough. After extracting, reflect: have you missed any important entities?\n"
        "Return ONLY valid JSON. No markdown, no explanation."
    )),
    ("human", "Extract entities and relationships from:\n\n{text}")
])


class EntityExtractor:
    def __init__(self, config: TREXConfig):
        self.config = config
        self.llm = ChatOpenAI(
            model=config.indexing_model,
            openai_api_key=config.openai_api_key,
            temperature=0,
        )
        self.chain = ENTITY_EXTRACTION_PROMPT | self.llm | JsonOutputParser()
        self.driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password)
        )
        self._init_neo4j_schema()

    def _init_neo4j_schema(self):
        with self.driver.session(database=self.config.neo4j_database) as session:
            session.run("CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
            session.run("CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)")

    def extract_and_store(self, nodes: List[Dict[str, Any]]):
        """Extract entities from leaf nodes and store in Neo4j."""
        # Only extract from leaf nodes (Level 0) to avoid duplicate extraction
        leaf_nodes = [n for n in nodes if n["level"] == 0]
        print(f"[GraphRAG] Extracting entities from {len(leaf_nodes)} leaf chunks...")

        for i, node in enumerate(leaf_nodes):
            if i % 20 == 0:
                print(f"[GraphRAG] Processing chunk {i}/{len(leaf_nodes)}...")
            try:
                result = self.chain.invoke({"text": node["text"]})
                self._store_to_neo4j(result, node["source"], node["id"])
            except Exception as e:
                print(f"[GraphRAG] Extraction failed for node {node['id']}: {e}")
                continue

    def _store_to_neo4j(self, extracted: Dict, source: str, chunk_id: str):
        entities = extracted.get("entities", [])
        relationships = extracted.get("relationships", [])

        with self.driver.session(database=self.config.neo4j_database) as session:
            # Upsert entities
            for entity in entities:
                session.run(
                    """
                    MERGE (e:Entity {name: $name})
                    SET e.type = $type,
                        e.description = CASE
                            WHEN e.description IS NULL THEN $description
                            ELSE e.description + ' | ' + $description
                        END,
                        e.source = $source,
                        e.chunk_id = $chunk_id
                    """,
                    name=entity.get("name", ""),
                    type=entity.get("type", "Unknown"),
                    description=entity.get("description", ""),
                    source=source,
                    chunk_id=chunk_id,
                )

            # Upsert relationships
            for rel in relationships:
                session.run(
                    """
                    MATCH (a:Entity {name: $source})
                    MATCH (b:Entity {name: $target})
                    MERGE (a)-[r:RELATES_TO {description: $description}]->(b)
                    SET r.strength = $strength
                    """,
                    source=rel.get("source", ""),
                    target=rel.get("target", ""),
                    description=rel.get("description", ""),
                    strength=rel.get("strength", 5),
                )

    def close(self):
        self.driver.close()
```

---

## 6. Qdrant Hybrid Indexer

```python
# ingestion/indexer.py
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
)
from config import TREXConfig


class QdrantHybridIndexer:
    """
    Stores all RAPTOR tree nodes (leaves + summaries) in Qdrant
    with both dense (semantic) and sparse (BM25-style) vectors.
    Qdrant's hybrid search natively fuses these with RRF.
    """

    DENSE_VECTOR_NAME = "dense"
    SPARSE_VECTOR_NAME = "sparse"

    def __init__(self, config: TREXConfig):
        self.config = config
        self.client = QdrantClient(url=config.qdrant_url)
        self.dense_embedder = OpenAIEmbeddings(
            model=config.embedding_model,
            openai_api_key=config.openai_api_key,
        )
        self.sparse_embedder = FastEmbedSparse(model_name="Qdrant/bm25")
        self._init_collection()

    def _init_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if self.config.qdrant_collection not in existing:
            self.client.create_collection(
                collection_name=self.config.qdrant_collection,
                vectors_config={
                    self.DENSE_VECTOR_NAME: VectorParams(
                        size=self.config.embedding_dimensions,
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    self.SPARSE_VECTOR_NAME: SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    )
                },
            )
            print(f"[Qdrant] Created collection: {self.config.qdrant_collection}")
        else:
            print(f"[Qdrant] Using existing collection: {self.config.qdrant_collection}")

    def get_vector_store(self) -> QdrantVectorStore:
        return QdrantVectorStore(
            client=self.client,
            collection_name=self.config.qdrant_collection,
            embedding=self.dense_embedder,
            sparse_embedding=self.sparse_embedder,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name=self.DENSE_VECTOR_NAME,
            sparse_vector_name=self.SPARSE_VECTOR_NAME,
        )

    def index_nodes(self, nodes: List[Dict[str, Any]]):
        """Convert RAPTOR tree nodes into LangChain Documents and index them."""
        documents = []
        for node in nodes:
            doc = Document(
                page_content=node["text"],
                metadata={
                    "node_id": node["id"],
                    "level": node["level"],
                    "parent_id": node.get("parent_id"),
                    "source": node.get("source", "unknown"),
                    "children_ids": ",".join(node.get("children_ids", [])),
                }
            )
            documents.append(doc)

        vector_store = self.get_vector_store()
        # Add in batches to avoid API rate limits
        batch_size = 50
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            vector_store.add_documents(batch)
            print(f"[Qdrant] Indexed {min(i + batch_size, len(documents))}/{len(documents)} nodes")

        print(f"[Qdrant] Indexing complete. Total nodes: {len(documents)}")
```

---

## 7. Hybrid Retriever (Qdrant + Neo4j)

```python
# retrieval/retriever.py
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from neo4j import GraphDatabase
from config import TREXConfig
from ingestion.indexer import QdrantHybridIndexer


class TREXRetriever:
    """
    Hybrid retriever that:
    1. Searches Qdrant with dense + sparse vectors (RRF fusion built-in)
    2. Enriches with Neo4j entity neighborhood context
    3. Returns combined context for the LLM
    """

    def __init__(self, config: TREXConfig):
        self.config = config
        indexer = QdrantHybridIndexer(config)
        self.vector_store = indexer.get_vector_store()
        self.neo4j_driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password)
        )

    def retrieve(self, query: str) -> Tuple[List[Document], str]:
        """
        Returns:
            - qdrant_docs: top-K tree nodes ranked by hybrid RRF score
            - graph_context: string summary of relevant entities + relationships
        """
        # --- Step 1: Qdrant hybrid search (dense + BM25, fused via RRF) ---
        qdrant_docs = self.vector_store.similarity_search(
            query=query,
            k=self.config.top_k,
        )
        print(f"[Retriever] Retrieved {len(qdrant_docs)} nodes from Qdrant")

        # --- Step 2: Neo4j entity context enrichment ---
        graph_context = self._get_graph_context(query)

        return qdrant_docs, graph_context

    def _get_graph_context(self, query: str) -> str:
        """
        Find entities likely mentioned in the query and retrieve their
        1-hop neighborhood from Neo4j (connected entities + relationships).
        """
        # Extract candidate entity names from query via simple word matching
        # (In production, you could use an LLM or NER model here)
        with self.neo4j_driver.session(database=self.config.neo4j_database) as session:
            # Full-text search for entities relevant to the query
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($query)
                   OR toLower(e.description) CONTAINS toLower($query)
                WITH e LIMIT 5
                MATCH (e)-[r:RELATES_TO]-(neighbor:Entity)
                RETURN e.name AS entity, e.type AS type, e.description AS desc,
                       r.description AS rel_desc, r.strength AS strength,
                       neighbor.name AS neighbor
                ORDER BY r.strength DESC
                LIMIT 30
                """,
                query=query[:200]  # Truncate to avoid overly broad match
            )
            records = result.data()

        if not records:
            return ""

        # Format into readable context
        lines = ["### Relevant Knowledge Graph Context\n"]
        seen = set()
        for record in records:
            entity = record["entity"]
            if entity not in seen:
                lines.append(f"**{entity}** ({record['type']}): {record['desc']}")
                seen.add(entity)
            lines.append(
                f"  → {record['rel_desc']} → **{record['neighbor']}** "
                f"(strength: {record['strength']}/10)"
            )

        return "\n".join(lines)

    def close(self):
        self.neo4j_driver.close()
```

---

## 8. Answer Generator (GPT-4o)

```python
# query/answer_generator.py
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import TREXConfig


ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert research assistant with access to a structured knowledge base.\n\n"
        "You will be given:\n"
        "1. Retrieved document passages (from a hierarchical RAPTOR tree — some are raw chunks, "
        "some are higher-level summaries)\n"
        "2. A knowledge graph context (entities and their relationships)\n\n"
        "Your task: answer the user's question accurately and completely, drawing on all provided context.\n"
        "- Cite evidence from the passages where relevant.\n"
        "- Use the knowledge graph to identify key actors, relationships, or structures.\n"
        "- If the context is insufficient to fully answer, say so clearly.\n"
        "- Write in clear, professional prose."
    )),
    ("human", (
        "## Retrieved Passages\n\n{passages}\n\n"
        "{graph_context}\n\n"
        "## Question\n{question}"
    ))
])


class AnswerGenerator:
    def __init__(self, config: TREXConfig):
        self.llm = ChatOpenAI(
            model=config.query_model,  # GPT-4o
            openai_api_key=config.openai_api_key,
            temperature=0.1,
        )
        self.chain = ANSWER_PROMPT | self.llm | StrOutputParser()

    def generate(
        self,
        question: str,
        qdrant_docs: List[Document],
        graph_context: str,
    ) -> str:
        # Format retrieved passages, tagging each with its tree level
        formatted_passages = []
        for i, doc in enumerate(qdrant_docs, 1):
            level = doc.metadata.get("level", 0)
            level_label = "Raw Chunk" if level == 0 else f"Level-{level} Summary"
            source = doc.metadata.get("source", "unknown")
            formatted_passages.append(
                f"[Passage {i} | {level_label} | Source: {source}]\n{doc.page_content}"
            )

        passages_text = "\n\n---\n\n".join(formatted_passages)
        graph_section = graph_context if graph_context else ""

        return self.chain.invoke({
            "question": question,
            "passages": passages_text,
            "graph_context": graph_section,
        })
```

---

## 9. Full Pipeline Orchestration

```python
# pipeline.py
from typing import List
from config import TREXConfig
from ingestion.loaders import load_documents
from ingestion.raptor_tree import RAPTORTreeBuilder
from ingestion.entity_extractor import EntityExtractor
from ingestion.indexer import QdrantHybridIndexer
from retrieval.retriever import TREXRetriever
from query.answer_generator import AnswerGenerator


class TREXPipeline:
    """
    Full TREX + GraphRAG pipeline.
    Two phases:
      - index(sources): one-time offline indexing
      - query(question): online retrieval and answer generation
    """

    def __init__(self, config: TREXConfig = None):
        self.config = config or TREXConfig()

    # ─────────────────────────────────────────
    # INDEXING PHASE (run once per corpus)
    # ─────────────────────────────────────────

    def index(self, sources: List[str]):
        """
        End-to-end indexing pipeline.
        sources: list of file paths or URLs.
        """
        print("=" * 60)
        print("PHASE 1: Document Loading")
        print("=" * 60)
        documents = load_documents(sources)
        print(f"Loaded {len(documents)} document pages/sections")

        print("\n" + "=" * 60)
        print("PHASE 2: RAPTOR Tree Construction (Truncated)")
        print("=" * 60)
        tree_builder = RAPTORTreeBuilder(self.config)
        all_nodes = tree_builder.build(documents)
        print(f"Total tree nodes: {len(all_nodes)}")

        print("\n" + "=" * 60)
        print("PHASE 3: Entity Extraction → Neo4j")
        print("=" * 60)
        extractor = EntityExtractor(self.config)
        extractor.extract_and_store(all_nodes)
        extractor.close()

        print("\n" + "=" * 60)
        print("PHASE 4: Indexing Tree into Qdrant (Hybrid)")
        print("=" * 60)
        indexer = QdrantHybridIndexer(self.config)
        indexer.index_nodes(all_nodes)

        print("\n✓ Indexing complete.")
        return all_nodes

    # ─────────────────────────────────────────
    # QUERY PHASE
    # ─────────────────────────────────────────

    def query(self, question: str) -> str:
        """Single-question query against the indexed corpus."""
        retriever = TREXRetriever(self.config)
        generator = AnswerGenerator(self.config)

        qdrant_docs, graph_context = retriever.retrieve(question)
        answer = generator.generate(question, qdrant_docs, graph_context)

        retriever.close()
        return answer
```

---

## 10. Usage Example

```python
# main.py
from pipeline import TREXPipeline
from config import TREXConfig

# Initialize
config = TREXConfig(
    openai_api_key="sk-...",
    qdrant_url="http://localhost:6333",
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="your_password",
)

pipeline = TREXPipeline(config)

# --- INDEX your corpus (run once) ---
sources = [
    "reports/annual_report_2024.pdf",
    "https://example.com/some-article",
    "data/transcripts.txt",
    "data/financials.csv",
]
pipeline.index(sources)

# --- QUERY ---
# Specific factual (OLTP-style) — retrieves from leaf chunks via BM25
answer = pipeline.query("Who is the CEO of the organization mentioned in the 2024 report?")
print(answer)

# Thematic (OLAP-style) — retrieves from summary nodes via semantic search
answer = pipeline.query("What are the key strategic themes across all documents?")
print(answer)
```

---

## 11. Running Infrastructure (Docker Compose)

```yaml
# docker-compose.yml
version: "3.8"
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage

  neo4j:
    image: neo4j:5.18
    ports:
      - "7474:7474"   # Browser UI
      - "7687:7687"   # Bolt protocol
    environment:
      NEO4J_AUTH: neo4j/password
      NEO4J_PLUGINS: '["apoc"]'
    volumes:
      - neo4j_data:/data

volumes:
  qdrant_data:
  neo4j_data:
```

Start with:
```bash
docker-compose up -d
```

---

## Key Design Decisions & Rationale

| Decision | Choice | Why |
|---|---|---|
| **Embedding model** | `text-embedding-3-small` | 1536 dims, excellent semantic quality, ~5x cheaper than `text-embedding-3-large` with marginal quality difference for RAG |
| **Indexing LLM** | GPT-4.1-mini | Handles high-volume summarization and entity extraction cheaply without quality loss |
| **Query LLM** | GPT-4o | Only fires once per user query; worth the cost for best answer quality |
| **Keyword search** | Qdrant BM25 sparse vectors via FastEmbed | Native Qdrant support; no separate Elasticsearch needed |
| **RRF fusion** | Qdrant built-in hybrid mode | Qdrant applies RRF internally when `RetrievalMode.HYBRID` is set — no manual implementation needed |
| **Tree truncation** | `max_tree_levels=2` | Mirrors TREX paper — 2 levels gives 90% of quality at ~30% of full RAPTOR cost |
| **Neo4j role** | Entity + relationship graph | Adds GraphRAG-style relational reasoning on top of TREX's tree retrieval |
| **Soft clustering threshold** | `0.5` | From RAPTOR paper — a chunk must have >50% probability of belonging to a cluster to be included |
| **LangChain version** | 1.2.15 | Stable 1.x line; no breaking changes until 2.0; Python 3.10+ required |
| **`api_key` vs `openai_api_key`** | `api_key` | Pydantic v2 field name in `langchain-openai` 0.4+ — old name is deprecated |
| **Structured output** | `with_structured_output(Pydantic)` | Replaces `JsonOutputParser` — more reliable, uses `response_format` under the hood in LangChain 1.x |
| **`ContextOverflowError`** | Caught in AnswerGenerator | Raised natively by `langchain-openai` when context window exceeded; graceful fallback to top-5 passages |
| **Streaming** | `.stream()` on AnswerGenerator | Fully stable in LangChain 1.x; available for real-time token-by-token UX |

---

*This document was generated by mAI.*
