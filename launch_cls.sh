#!/usr/bin/env bash
set -u

APP_NAME="CLS RAG+CAG Prototype"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_CMD="${PYTHON_CMD:-python3}"
PORT="${STREAMLIT_PORT:-8501}"
OLLAMA_LOG="$ROOT_DIR/ollama.log"
REQUIREMENTS_MARKER="$VENV_DIR/.requirements-installed"

cd "$ROOT_DIR" || exit 1

say() {
    printf "\n[%s] %s\n" "$APP_NAME" "$1"
}

die() {
    printf "\n[%s] ERROR: %s\n" "$APP_NAME" "$1" >&2
    printf "\nPress Enter to close this window."
    read -r _ || true
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
    if [ ! -x "$VENV_DIR/bin/streamlit" ] || [ ! -f "$REQUIREMENTS_MARKER" ] || [ "$ROOT_DIR/requirements.txt" -nt "$REQUIREMENTS_MARKER" ]; then
        say "Installing or refreshing Python packages..."
        "$VENV_DIR/bin/python" -m pip install -r "$ROOT_DIR/requirements.txt" || die "Package installation failed."
        touch "$REQUIREMENTS_MARKER"
    fi
}

ensure_ollama_server() {
    if ! command -v ollama >/dev/null 2>&1; then
        die "Ollama was not found. Install it from https://ollama.com, then run this launcher again."
    fi

    if ollama list >/dev/null 2>&1; then
        return
    fi

    say "Starting Ollama in the background..."
    nohup ollama serve > "$OLLAMA_LOG" 2>&1 &

    for _ in $(seq 1 20); do
        if ollama list >/dev/null 2>&1; then
            return
        fi
        sleep 1
    done

    die "Ollama did not start. Check $OLLAMA_LOG for details."
}

model_is_installed() {
    local model="$1"
    ollama list | awk 'NR > 1 {print $1}' | grep -Eq "^${model}(:|$)"
}

ensure_model() {
    local model="$1"

    if model_is_installed "$model"; then
        return
    fi

    say "Missing Ollama model: $model"
    printf "Download it now? [Y/n] "
    read -r answer || answer="n"

    case "$answer" in
        ""|y|Y|yes|YES)
            ollama pull "$model" || die "Could not download $model."
            ;;
        *)
            die "The app needs $model before ingestion/query will work."
            ;;
    esac
}

pick_port() {
    PORT="$("$VENV_DIR/bin/python" - "$PORT" <<'PY'
import socket
import sys

start = int(sys.argv[1])
for port in range(start, start + 50):
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", port))
    except OSError:
        continue
    finally:
        sock.close()
    print(port)
    break
else:
    raise SystemExit("No open Streamlit port found.")
PY
)" || die "Could not find an open local port."
}

open_browser() {
    local url="$1"

    if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$url" >/dev/null 2>&1 &
    fi
}

say "Preparing launch from $ROOT_DIR"
pick_python
ensure_venv
ensure_requirements

if [ "${SKIP_OLLAMA_CHECK:-0}" != "1" ]; then
    ensure_ollama_server
    ensure_model "nomic-embed-text"
else
    say "Skipping Ollama checks because SKIP_OLLAMA_CHECK=1."
fi

pick_port

if [ "${LAUNCHER_DRY_RUN:-0}" = "1" ]; then
    say "Dry run complete. Streamlit would launch on http://localhost:$PORT"
    exit 0
fi

URL="http://localhost:$PORT"
say "Launching Streamlit at $URL"
open_browser "$URL"

exec "$VENV_DIR/bin/streamlit" run "$ROOT_DIR/app.py" \
    --server.port "$PORT" \
    --server.headless true \
    --browser.gatherUsageStats false
