from ragandcag.knowledge_base import KnowledgeBase
from ragandcag.database.vector.chroma_db import ChromaDB
from ragandcag.reranker import NoReranker
from examples.cls_pipeline import EMBED_DIM, HashEmbedder


class HashKnowledgeBaseEmbedding:
    """Adapter for the legacy KnowledgeBase API using the no-download HashEmbedder."""

    def __init__(self, dimension: int = EMBED_DIM):
        self.dimension = dimension
        self.encoder = HashEmbedder(dimensions=dimension)

    def embed_query(self, text: str) -> list[float]:
        return self.encoder.embed([text])[0]

    def embed_documents(self, texts, batch_size: int = 16, progress_callback=None):
        vectors = []
        total = len(texts)
        for start in range(0, total, batch_size):
            batch = texts[start:start + batch_size]
            vectors.extend(self.encoder.embed(batch))
            if progress_callback:
                progress_callback(min(start + batch_size, total), total)
        return vectors


def build_cls_kb(kb_id: str = "cls-hash-v1") -> KnowledgeBase:
    """
    Legacy inbox-daemon KnowledgeBase using the same no-download encoder as the app.

    ChromaDB is non-negotiable here: BasicVectorDB silently discards metadata_filter
    and breaks Prism lanes.
    """
    return KnowledgeBase(
        kb_id=kb_id,
        embedding_model=HashKnowledgeBaseEmbedding(),
        vector_db=ChromaDB(
            kb_id=kb_id,
            # persistence path is derived from kb_id; survives restarts
        ),
        auto_context_model=None,
        reranker=NoReranker(),       # v1 — add a reranker only if eval drops
    )
