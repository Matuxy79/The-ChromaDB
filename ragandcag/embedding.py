from typing import Callable, List, Optional

import urllib.error
import urllib.request

from langchain_ollama import OllamaEmbeddings


OLLAMA_TAGS_URL = "http://127.0.0.1:11434/api/tags"


class OllamaEmbedding:
    def __init__(self, model: str = "nomic-embed-text", dimension: int = 768):
        self.model = model
        self.dimension = dimension
        self.client = OllamaEmbeddings(model=model)

    def embed_query(self, text: str) -> List[float]:
        return self.client.embed_query(text)

    def embed_documents(
        self,
        texts: List[str],
        batch_size: int = 16,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[List[float]]:
        if not texts:
            if progress_callback:
                progress_callback(0, 0)
            return []

        total = len(texts)
        vectors: List[List[float]] = []

        for start in range(0, total, batch_size):
            batch = texts[start:start + batch_size]
            vectors.extend(self.client.embed_documents(batch))
            if progress_callback:
                progress_callback(min(start + batch_size, total), total)

        return vectors

    def is_available(self, timeout: float = 1.0) -> bool:
        try:
            with urllib.request.urlopen(OLLAMA_TAGS_URL, timeout=timeout) as response:
                return response.status == 200
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            return False
