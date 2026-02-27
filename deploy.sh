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
install_system_deps() {
    echo "[deploy] installing system dependencies ..."
    sudo apt-get update -qq
    sudo apt-get install -y \
        portaudio19-dev \
        python3-dev \
        libasound2-dev
}

install_ollama() {
    if command -v ollama &>/dev/null; then
        echo "[deploy] ollama already installed ($(ollama --version))"
    else
        echo "[deploy] installing ollama ..."
        curl -fsSL https://ollama.com/install.sh | sh
        hash -r   # flush bash's command cache so ollama is found immediately
    fi
}

ensure_ollama_running() {
    if ollama list &>/dev/null; then
        echo "[deploy] ollama server is running"
        return
    fi
    echo "[deploy] starting ollama service ..."
    sudo systemctl enable --now ollama
    # Wait up to 15s for the server to accept connections
    for i in $(seq 1 15); do
        sleep 1
        if ollama list &>/dev/null; then
            echo "[deploy] ollama server ready"
            return
        fi
    done
    echo "[deploy] warning: ollama server did not start in time, continuing anyway"
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

clone_wake_word() {
    local wake_dir="$DIR/simple-wake-word"
    if [ -d "$wake_dir" ]; then
        echo "[deploy] simple-wake-word already cloned"
        return
    fi
    echo "[deploy] cloning simple-wake-word ..."
    git clone https://github.com/gadgetlabs/simple-wake-word "$wake_dir"
}

download_nltk_data() {
    echo "[deploy] downloading NLTK corpora for g2p-en ..."
    python3 -c "
import nltk
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('cmudict', quiet=True)
"
}

install_python_deps() {
    # These must be force-installed to shadow outdated system apt versions
    # that break with modern numpy / transformers
    pip install scipy Pillow --ignore-installed
    pip install -r "$DIR/requirements.txt"
    clone_wake_word
    download_nltk_data  # idempotent - skips corpora already present
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

    install_system_deps
    install_ollama
    ensure_ollama_running
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

ensure_ollama_running
pull_ollama_model      # no-op if model unchanged
download_piper_voice   # no-op if voice already present
pip install -r requirements.txt --quiet
download_nltk_data     # no-op if corpora already present

echo "[deploy] done. restart main.py to apply."

# ── Optional: restart systemd service if you set one up ──────────────────
# sudo systemctl restart chaos-brain
