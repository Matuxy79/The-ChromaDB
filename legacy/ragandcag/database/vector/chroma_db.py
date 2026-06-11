import chromadb
from typing import Dict, Any, List, Optional

class ChromaDB:
    def __init__(self, kb_id: str, path: Optional[str] = None):
        self.kb_id = kb_id
        if path is None:
            path = f"./{kb_id}_chroma_db"
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(name=kb_id)

    def format_metadata_filter(self, metadata_filter: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translates the RAG+CAG filter schema into Chroma native where= clause.
        Example: {"field": "colour_code", "operator": "equals", "value": "purple"}
        """
        field = metadata_filter.get("field")
        operator = metadata_filter.get("operator")
        value = metadata_filter.get("value")
        
        op_map = {
            "equals": "$eq",
            "not_equals": "$ne",
            "greater_than": "$gt",
            "less_than": "$lt",
            "in": "$in",
            "not_in": "$nin"
        }
        
        chroma_op = op_map.get(operator, "$eq")
        return {field: {chroma_op: value}}

    def search(self, query_vector: List[float], top_k: int, metadata_filter: Optional[Dict] = None):
        where = self.format_metadata_filter(metadata_filter) if metadata_filter else None
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        return results

    def count(self) -> int:
        return self.collection.count()

    def add_vectors(
        self,
        vectors: List[List[float]],
        metadatas: List[Dict],
        ids: List[str],
        documents: Optional[List[str]] = None
    ):
        self.collection.add(
            embeddings=vectors,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
