#!/usr/bin/env bash
set -u

APP_NAME="CLS RAG+CAG API"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_CMD="${PYTHON_CMD:-python3}"
PORT="${API_PORT:-8010}"
REQUIREMENTS_MARKER="$VENV_DIR/.requirements-installed"

cd "$ROOT_DIR" || exit 1

say() {
    printf "\n[%s] %s\n" "$APP_NAME" "$1"
}

die() {
    printf "\n[%s] ERROR: %s\n" "$APP_NAME" "$1" >&2
    exit 1
}

pick_python() {
    if command -v "$PYTHON_CMD" >/dev/null 2>&1; then
        return
    fi
    if command -v python >/dev/null 2>&1; then
        PYTHON_CMD="python"
        return
    fi
    die "Python was not found. Install Python 3, then run this launcher again."
}

ensure_venv() {
    if [ ! -x "$VENV_DIR/bin/python" ]; then
        say "Creating the local Python environment..."
        "$PYTHON_CMD" -m venv "$VENV_DIR" || die "Could not create .venv."
    fi
}

ensure_requirements() {
    if [ ! -x "$VENV_DIR/bin/uvicorn" ] || [ ! -f "$REQUIREMENTS_MARKER" ] || [ "$ROOT_DIR/requirements.txt" -nt "$REQUIREMENTS_MARKER" ]; then
        say "Installing or refreshing Python packages..."
        "$VENV_DIR/bin/python" -m pip install -r "$ROOT_DIR/requirements.txt" || die "Package installation failed."
        touch "$REQUIREMENTS_MARKER"
    fi
}

load_env() {
    # Match launch_cls.sh: auto-source local carrier config so FastAPI bridge mode
    # sees the same OpenRouter key/model as the Streamlit embedded path.
    if [ -f "$ROOT_DIR/cls.env" ]; then
        say "Loading config from cls.env"
        set -a
        # shellcheck disable=SC1091
        . "$ROOT_DIR/cls.env"
        set +a
    fi
}

say "Preparing API from $ROOT_DIR"
load_env
pick_python
ensure_venv
ensure_requirements

if [ -n "${CLS_DLLM_API_URL:-}" ]; then
    say "Inference carrier endpoint: $CLS_DLLM_API_URL"
else
    say "Using default inference carrier endpoint unless overridden by CLS_DLLM_API_URL."
fi

if [ "${LAUNCHER_DRY_RUN:-0}" = "1" ]; then
    say "Dry run complete. API would launch on http://127.0.0.1:$PORT"
    exit 0
fi

say "Launching API at http://127.0.0.1:$PORT"
say "OpenAPI docs: http://127.0.0.1:$PORT/docs"

exec "$VENV_DIR/bin/uvicorn" api:app \
    --host 127.0.0.1 \
    --port "$PORT"
