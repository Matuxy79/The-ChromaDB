from typing import List, Dict, Any

class NoReranker:
    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return documents
