"""Production ASGI entry point — the module Railway (and any PaaS that expects a
``main`` module) imports when it runs ``uvicorn main:app``.

The actual FastAPI application lives in ``api.py``; this module re-exports it so
the deployment contract holds without duplicating any route or middleware code,
and adds a ``__main__`` fallback that binds to ``$PORT`` when executed directly
(``python main.py``), matching Railway's injected port.
"""

from __future__ import annotations

import os

from api import app  # noqa: F401  — the ASGI application the platform imports

__all__ = ["app"]


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")