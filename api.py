from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from cls_config import APP_VERSION, DEFAULT_DLLM_MODEL
from cls_service import answer_text, ask_manual, call_dllm_api, dllm_status, service_status

CLS_RAG_MODEL = "cls-rag-cag-v0.9"


app = FastAPI(
    title="CLS IVU RAG+CAG API",
    version=APP_VERSION,
    description="Shared RAG API plus dLLM API proxy for Streamlit and OpenAI-compatible frontends.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(8, ge=1, le=20)
    cache_enabled: bool = True
    min_similarity: float = Field(0.97, ge=0.0, le=1.0)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = CLS_RAG_MODEL
    messages: list[ChatMessage]
    stream: bool = False
    top_k: int = Field(8, ge=1, le=20)
    cache_enabled: bool = True
    min_similarity: float = Field(0.97, ge=0.0, le=1.0)


class DllmChatRequest(BaseModel):
    model: str = DEFAULT_DLLM_MODEL
    messages: list[ChatMessage]
    system: str | None = None


def _query_payload(request: QueryRequest) -> dict[str, Any]:
    result = ask_manual(
        request.query,
        top_k=request.top_k,
        cache_enabled=request.cache_enabled,
        min_similarity=request.min_similarity,
    )
    return {
        "query": request.query,
        "answer": result["answer"],
        "answer_text": answer_text(result["answer"]),
        "category": result["category"],
        "from_cache": result["from_cache"],
        "similarity": result["similarity"],
        "rows": result["rows"],
    }


def _last_user_message(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user" and message.content.strip():
            return message.content.strip()
    raise HTTPException(status_code=400, detail="At least one user message is required.")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"version": APP_VERSION, "status": "ok", **service_status()}


@app.get("/v1/models")
def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": CLS_RAG_MODEL,
                "object": "model",
                "created": 0,
                "owned_by": "cls",
            },
            {
                "id": DEFAULT_DLLM_MODEL,
                "object": "model",
                "created": 0,
                "owned_by": "dllm-api",
            },
        ],
    }


@app.post("/v1/query")
def query_manual(request: QueryRequest) -> dict[str, Any]:
    return _query_payload(request)


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming is not implemented for the RAG endpoint yet.")
    if request.model == DEFAULT_DLLM_MODEL:
        try:
            text = call_dllm_api(
                [message.model_dump() for message in request.messages],
                model=request.model,
            )
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"dLLM API model call failed: {exc}") from exc
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    payload = _query_payload(
        QueryRequest(
            query=_last_user_message(request.messages),
            top_k=request.top_k,
            cache_enabled=request.cache_enabled,
            min_similarity=request.min_similarity,
        )
    )
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": payload["answer_text"]},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "cls_rag": payload,
    }


@app.get("/v1/dllm/status")
def dllm_status_endpoint() -> dict[str, Any]:
    return dllm_status()


@app.post("/v1/dllm/chat")
def dllm_chat(request: DllmChatRequest) -> dict[str, str]:
    try:
        text = call_dllm_api(
            [message.model_dump() for message in request.messages],
            system=request.system,
            model=request.model,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"dLLM API model call failed: {exc}") from exc
    return {"model": request.model, "content": text}
