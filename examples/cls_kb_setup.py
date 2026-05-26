from ragandcag.knowledge_base import KnowledgeBase
from ragandcag.database.vector.chroma_db import ChromaDB
from ragandcag.embedding import OllamaEmbedding
from ragandcag.llm import OllamaAPI
from ragandcag.reranker import NoReranker

def build_cls_kb(kb_id: str = "cls-v1") -> KnowledgeBase:
    """
    The canonical CLS KnowledgeBase. ChromaDB is non-negotiable here —
    BasicVectorDB silently discards metadata_filter and breaks Prism lanes.
    """
    return KnowledgeBase(
        kb_id=kb_id,
        embedding_model=OllamaEmbedding(
            model="nomic-embed-text",
            dimension=768,          # verified output dim
        ),
        vector_db=ChromaDB(
            kb_id=kb_id,
            # persistence path is derived from kb_id; survives restarts
        ),
        auto_context_model=OllamaAPI(model="llama3.2:1b"),
        reranker=NoReranker(),       # v1 — add a reranker only if eval drops
    )
