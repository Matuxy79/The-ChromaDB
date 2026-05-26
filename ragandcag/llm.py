from typing import Dict, Generator, List, Optional

import urllib.error
import urllib.request

from langchain_ollama import ChatOllama


OLLAMA_TAGS_URL = "http://127.0.0.1:11434/api/tags"


class OllamaAPI:
    def __init__(self, model: str = "llama3.2:1b"):
        self.model = model
        self.client = ChatOllama(model=model)

    def generate(self, messages: List[Dict[str, str]], system: Optional[str] = None) -> str:
        if system:
            messages = [{"role": "system", "content": system}] + messages
        response = self.client.invoke(messages)
        return response.content

    def stream(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
    ) -> Generator[str, None, None]:
        if system:
            messages = [{"role": "system", "content": system}] + messages
        for chunk in self.client.stream(messages):
            yield chunk.content

    def is_available(self, timeout: float = 1.0) -> bool:
        try:
            with urllib.request.urlopen(OLLAMA_TAGS_URL, timeout=timeout) as response:
                return response.status == 200
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            return False
