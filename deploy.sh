#!/usr/bin/env bash
# deploy.sh
# Handles first install and subsequent updates independently.
# Safe to run repeatedly.
#
# Usage (one line on the Jetson):
#   bash deploy.sh
#
# To auto-update every 5 minutes add to crontab  (crontab -e):
#   */5 * * * * /home/user/agentic_control/deploy.sh >> /home/user/deploy.log 2>&1

set -euo pipefail

REPO="https://github.com/gadgetlabs/agentic_control.git"
DIR="$HOME/agentic_control"
BRANCH="main"
MARKER="$DIR/.installed"   # created after a successful install; not committed to git

# ── Helpers ───────────────────────────────────────────────────────────────
install_ollama() {
    if command -v ollama &>/dev/null; then
        echo "[deploy] ollama already installed ($(ollama --version))"
    else
        echo "[deploy] installing ollama ..."
        curl -fsSL https://ollama.com/install.sh | sh
    fi
}

pull_ollama_model() {
    local model
    model=$(grep -E "^OLLAMA_MODEL=" "$DIR/.env" 2>/dev/null | cut -d= -f2 || echo "qwen2.5:3b")
    echo "[deploy] pulling ollama model '$model' (skipped if already present) ..."
    ollama pull "$model"
}

download_piper_voice() {
    local voice="en_GB-jenny_dioco-medium"
    local base="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/jenny_dioco/medium"
    local voices_dir="$DIR/voices"

    mkdir -p "$voices_dir"

    if [ -f "$voices_dir/${voice}.onnx" ]; then
        echo "[deploy] piper voice already present"
        return
    fi

    echo "[deploy] downloading piper voice '$voice' (~60 MB) ..."
    curl -fsSL -o "$voices_dir/${voice}.onnx"      "$base/${voice}.onnx"
    curl -fsSL -o "$voices_dir/${voice}.onnx.json" "$base/${voice}.onnx.json"
    echo "[deploy] piper voice downloaded"
}

install_python_deps() {
    pip install -r "$DIR/requirements.txt"
    pip install git+https://github.com/gadgetlabs/simple-wake-word
}

# ── Clone if the directory doesn't exist yet ─────────────────────────────
if [ ! -d "$DIR/.git" ]; then
    echo "[deploy] cloning $REPO ..."
    git clone --branch "$BRANCH" "$REPO" "$DIR"
fi

cd "$DIR"

# ── Install if not yet done (handles both fresh clone and manual clone) ───
# The marker file is created here after a successful install and is
# listed in .gitignore so it is never committed to the repo.
if [ ! -f "$MARKER" ]; then
    echo "[deploy] running first-time install ..."

    install_ollama
    pull_ollama_model
    download_piper_voice
    install_python_deps

    touch "$MARKER"
    echo "[deploy] install complete. copy .env.example to .env, then: python $DIR/main.py"
    exit 0
fi

# ── Check for new commits ─────────────────────────────────────────────────
git fetch origin "$BRANCH" --quiet

LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse "origin/$BRANCH")

if [ "$LOCAL" = "$REMOTE" ]; then
    echo "[deploy] up to date at $(git rev-parse --short HEAD)"
    exit 0
fi

echo "[deploy] $(git rev-parse --short HEAD) → $(git rev-parse --short origin/$BRANCH)"

# ── Pull and reinstall ────────────────────────────────────────────────────
git pull origin "$BRANCH" --quiet

pull_ollama_model      # no-op if model unchanged
download_piper_voice   # no-op if voice already present
pip install -r requirements.txt --quiet

echo "[deploy] done. restart main.py to apply."

# ── Optional: restart systemd service if you set one up ──────────────────
# sudo systemctl restart chaos-brain
