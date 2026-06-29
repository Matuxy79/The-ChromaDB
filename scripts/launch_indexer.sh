#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_CMD="${PYTHON_CMD:-python3}"

cd "$ROOT_DIR" || exit 1

if [ ! -x "$VENV_DIR/bin/python" ]; then
    "$PYTHON_CMD" -m venv "$VENV_DIR" || exit 1
fi

if [ ! -x "$VENV_DIR/bin/streamlit" ]; then
    "$VENV_DIR/bin/python" -m pip install -r "$ROOT_DIR/requirements.txt" || exit 1
fi

mkdir -p "$ROOT_DIR/docs/inbox" "$ROOT_DIR/docs/processed" "$ROOT_DIR/docs/failed"

printf "[CLS RAG+CAG Indexer] Using HashEmbedder only; no local LLM or embedding model will be pulled.\n"

exec "$VENV_DIR/bin/python" "$ROOT_DIR/ingest_daemon.py" "$@"
