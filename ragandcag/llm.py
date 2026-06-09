import json
import os
from typing import Dict, Generator, List, Optional

import urllib.error
import urllib.request


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("CLS_OLLAMA_MODEL", "")


class OllamaAPI:
    def __init__(self, model: str | None = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL):
        if not model:
            raise RuntimeError(
                "OllamaAPI is an optional legacy adapter. Pass a model name explicitly; "
                "the v0.9 dLLM path uses CLS_DLLM_API_URL and never pulls local chat models."
            )
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise RuntimeError("OllamaAPI requires the optional langchain-ollama package.") from exc
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.client = ChatOllama(model=model, base_url=self.base_url)

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
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=timeout) as response:
                return response.status == 200
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            return False

    def list_models(self, timeout: float = 1.0) -> List[str]:
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError, json.JSONDecodeError):
            return []
        models = payload.get("models", [])
        names = []
        for item in models:
            name = item.get("name") or item.get("model")
            if name:
                names.append(name)
        return names
