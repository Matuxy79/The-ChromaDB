"""FastAPI shared bridge — an *optional* HTTP layer in front of the RAG+CAG backend.

This API is only active when the environment variable ``CLS_USE_API=1`` is set.
By default the Streamlit UI (``app.py``) and the Chainlit Ask Lane (``chat_lane.py``)
import ``cls_service`` directly; no HTTP hop is needed for local single-machine use.

Three route groups are exposed:

    POST /v1/query
        Structured retrieval request.  Returns grounded evidence rows and the
        assembled extractive answer.  The schema mirrors OpenAI where sensible
        so existing tooling can point at this endpoint with minimal changes.

    POST /v1/chat/completions
        OpenAI-compatible chat endpoint.  Wraps the same retrieval backend so
        any OpenAI-compatible client (LangChain, LlamaIndex, curl, etc.) can
        drive the CLS knowledge base without code changes.

    POST /v1/dllm/chat
        Thin proxy to the configured generative carrier (OpenRouter by default,
        Ollama for offline use).  Only reachable when ``CLS_RETRIEVAL_ONLY=0``.

CORS is restricted to localhost origins; no external traffic is expected in the
current prototype deployment.

Start standalone:  ./scripts/launch_api.sh
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from cls_config import APP_VERSION, DEFAULT_DLLM_MODEL, KEYWORD_ONLY_RETRIEVAL, RETRIEVAL_ONLY
from cls_service import answer_text, ask_manual, call_dllm_api, dllm_status, service_status

CLS_RAG_MODEL = "cls-rag-cag-v1.0"


app = FastAPI(
    title="CLS RAG+CAG API",
    version=APP_VERSION,
    description="Shared RAG API plus inference carrier proxy for Streamlit, Chainlit, and OpenAI-compatible frontends.",
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
    top_k: int = Field(16, ge=1, le=24)
    cache_enabled: bool = True
    min_similarity: float = Field(0.80, ge=0.0, le=1.0)
    metadata_filter: dict | None = None
    debate_enabled: bool = False
    keyword_only: bool | None = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = CLS_RAG_MODEL
    messages: list[ChatMessage]
    stream: bool = False
    top_k: int = Field(16, ge=1, le=24)
    cache_enabled: bool = True
    min_similarity: float = Field(0.80, ge=0.0, le=1.0)
    metadata_filter: dict | None = None
    debate_enabled: bool = False
    keyword_only: bool | None = None


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
        metadata_filter=request.metadata_filter,
        debate_enabled=request.debate_enabled,
        keyword_only=request.keyword_only,
    )
    return {
        "query": request.query,
        "answer": result["answer"],
        "answer_text": answer_text(result["answer"]),
        "category": result["category"],
        "from_cache": result["from_cache"],
        "similarity": result["similarity"],
        "retrieval_mode": result.get("retrieval_mode", "semantic"),
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
    data = [
        {
            "id": CLS_RAG_MODEL,
            "object": "model",
            "created": 0,
            "owned_by": "cls",
        },
    ]
    if not RETRIEVAL_ONLY:
        data.append(
            {
                "id": DEFAULT_DLLM_MODEL,
                "object": "model",
                "created": 0,
                "owned_by": "dllm-api",
            }
        )
    return {
        "object": "list",
        "data": data,
        "retrieval_only": RETRIEVAL_ONLY,
        "keyword_only": KEYWORD_ONLY_RETRIEVAL,
    }


@app.post("/v1/query")
def query_manual(request: QueryRequest) -> dict[str, Any]:
    return _query_payload(request)


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming is not implemented for the RAG endpoint yet.")
    if request.model == DEFAULT_DLLM_MODEL:
        if RETRIEVAL_ONLY:
            raise HTTPException(
                status_code=403,
                detail="Retrieval-only mode is active; LLM chat completions are disabled.",
            )
        try:
            text = call_dllm_api(
                [message.model_dump() for message in request.messages],
                model=request.model,
            )
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Inference carrier model call failed: {exc}") from exc
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
            metadata_filter=request.metadata_filter,
            debate_enabled=request.debate_enabled,
            keyword_only=request.keyword_only,
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
    if RETRIEVAL_ONLY:
        raise HTTPException(
            status_code=403,
            detail="Retrieval-only mode is active; LLM chat is disabled.",
        )
    try:
        text = call_dllm_api(
            [message.model_dump() for message in request.messages],
            system=request.system,
            model=request.model,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Inference carrier model call failed: {exc}") from exc
    return {"model": request.model, "content": text}
