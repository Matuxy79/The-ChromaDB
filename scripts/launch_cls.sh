#!/usr/bin/env bash
set -u

APP_NAME="CLS RAG+CAG Prototype"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_CMD="${PYTHON_CMD:-python3}"
PORT="${STREAMLIT_PORT:-8501}"
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

load_env() {
    # Auto-source local carrier config (API URL/key/model) so synthesis works
    # without exporting vars by hand. Gitignored.
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
pick_python
ensure_venv
ensure_requirements
say "No local model checks are run. Carrier synthesis/cleanup use CLS_DLLM_API_URL when configured."

pick_port

if [ "${LAUNCHER_DRY_RUN:-0}" = "1" ]; then
    say "Dry run complete. Streamlit would launch on http://localhost:$PORT"
    exit 0
fi

URL="http://localhost:$PORT"
say "Launching Streamlit at $URL"
if [ "${CLS_USE_API:-0}" = "1" ]; then
    say "API bridge enabled via CLS_API_URL=${CLS_API_URL:-http://127.0.0.1:8010}"
fi
open_browser "$URL"

exec "$VENV_DIR/bin/streamlit" run "$ROOT_DIR/app.py" \
    --server.port "$PORT" \
    --server.headless true \
    --browser.gatherUsageStats false
