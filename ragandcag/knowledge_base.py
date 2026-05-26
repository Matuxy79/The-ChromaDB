import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter


ProgressCallback = Callable[[str, int, int], None]


class KnowledgeBase:
    def __init__(
        self,
        kb_id: str,
        embedding_model,
        vector_db,
        auto_context_model=None,
        reranker=None,
        chunk_size: int = 900,
        chunk_overlap: int = 120,
    ):
        self.kb_id = kb_id
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.auto_context_model = auto_context_model
        self.reranker = reranker
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def add_document(
        self,
        text: str,
        metadata: Dict[str, Any],
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "chunks": 0,
            "chunk_seconds": 0.0,
            "embed_seconds": 0.0,
            "store_seconds": 0.0,
        }

        chunk_start = time.perf_counter()
        if progress_callback:
            progress_callback("chunk", 0, 1)
        chunks = self._splitter.split_text(text)
        stats["chunk_seconds"] = time.perf_counter() - chunk_start
        stats["chunks"] = len(chunks)
        if progress_callback:
            progress_callback("chunk", len(chunks), max(len(chunks), 1))

        if not chunks:
            return stats

        embed_start = time.perf_counter()

        def _embed_progress(current: int, total: int) -> None:
            if progress_callback:
                progress_callback("embed", current, total)

        if progress_callback:
            progress_callback("embed", 0, len(chunks))
        vectors = self.embedding_model.embed_documents(
            chunks,
            progress_callback=_embed_progress,
        )
        stats["embed_seconds"] = time.perf_counter() - embed_start

        store_start = time.perf_counter()
        if progress_callback:
            progress_callback("store", 0, len(chunks))
        metadatas = [metadata.copy() for _ in chunks]
        ids = [f"{self.kb_id}_{uuid.uuid4()}" for _ in chunks]
        self.vector_db.add_vectors(vectors, metadatas, ids, documents=chunks)
        stats["store_seconds"] = time.perf_counter() - store_start
        if progress_callback:
            progress_callback("store", len(chunks), len(chunks))

        return stats

    def query(
        self,
        queries: List[str],
        metadata_filter: Optional[Dict] = None,
        rse_params: str = "balanced",
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Query & Retrieval pipeline:
        4. Vector Search
        5. RSE (Simulated)
        6. Reranking
        """
        if not queries:
            return []

        query_vector = self.embedding_model.embed_query(queries[0])
        results = self.vector_db.search(query_vector, top_k=top_k, metadata_filter=metadata_filter)

        formatted_results: List[Dict[str, Any]] = []
        if results and results.get("documents") and results.get("metadatas"):
            documents = results["documents"][0] or []
            metadatas = results["metadatas"][0] or []
            distances = (results.get("distances") or [[]])[0] or []
            for index, (text, meta) in enumerate(zip(documents, metadatas)):
                if not text:
                    continue
                distance = distances[index] if index < len(distances) else None
                formatted_results.append({
                    "text": text,
                    "metadata": meta,
                    "distance": distance,
                })

        if self.reranker:
            formatted_results = self.reranker.rerank(queries[0], formatted_results)

        return formatted_results

    def count_chunks(self) -> int:
        if hasattr(self.vector_db, "count"):
            return self.vector_db.count()
        return 0
