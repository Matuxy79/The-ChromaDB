#!/usr/bin/env bash
set -u

APP_NAME="CLS Ask Lane (Chainlit)"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_CMD="${PYTHON_CMD:-python3}"
PORT="${CHAINLIT_PORT:-8000}"
REQUIREMENTS_MARKER="$VENV_DIR/.requirements-installed"

cd "$ROOT_DIR" || exit 1

say()  { printf "\n[%s] %s\n" "$APP_NAME" "$1"; }
die()  { printf "\n[%s] ERROR: %s\n" "$APP_NAME" "$1" >&2; exit 1; }

ensure_venv() {
    if [ ! -x "$VENV_DIR/bin/python" ]; then
        say "Creating the local Python environment..."
        command -v "$PYTHON_CMD" >/dev/null 2>&1 || PYTHON_CMD="python"
        "$PYTHON_CMD" -m venv "$VENV_DIR" || die "Could not create .venv."
    fi
}

ensure_requirements() {
    if [ ! -x "$VENV_DIR/bin/chainlit" ] || [ ! -f "$REQUIREMENTS_MARKER" ] || [ "$ROOT_DIR/requirements.txt" -nt "$REQUIREMENTS_MARKER" ]; then
        say "Installing or refreshing Python packages..."
        "$VENV_DIR/bin/python" -m pip install -r "$ROOT_DIR/requirements.txt" || die "Package installation failed."
        touch "$REQUIREMENTS_MARKER"
    fi
}

load_env() {
    # Carrier + parrot config (CLS_PARROT_URL/MODEL, CLS_DLLM_*). Gitignored.
    if [ -f "$ROOT_DIR/cls.env" ]; then
        say "Loading config from cls.env"
        set -a
        # shellcheck disable=SC1091
        . "$ROOT_DIR/cls.env"
        set +a
    fi
}

say "Preparing launch from $ROOT_DIR"
load_env
ensure_venv
ensure_requirements
if [ "${CLS_RETRIEVAL_ONLY:-1}" = "0" ] || [ "${CLS_RETRIEVAL_ONLY:-1}" = "false" ]; then
    say "The Ask Lane parrot uses the local model at ${CLS_PARROT_URL:-http://localhost:11434/v1} (${CLS_PARROT_MODEL:-qwen2.5:0.5b}). Start Ollama first: ollama serve"
else
    say "Retrieval-only mode active. The Ask Lane parrot is disabled."
fi

URL="http://localhost:$PORT"

if [ "${LAUNCHER_DRY_RUN:-0}" = "1" ]; then
    say "Dry run complete. Chainlit Ask Lane would launch at $URL"
    exit 0
fi

say "Launching Chainlit Ask Lane at $URL"
command -v xdg-open >/dev/null 2>&1 && xdg-open "$URL" >/dev/null 2>&1 &

exec "$VENV_DIR/bin/chainlit" run "$ROOT_DIR/chat_lane.py" --headless --port "$PORT"
