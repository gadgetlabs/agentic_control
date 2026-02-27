#!/usr/bin/env bash
# deploy.sh
# Checks GitHub for new commits. If found, pulls and reinstalls.
# Safe to run repeatedly - does nothing when already up to date.
#
# Usage (one line on the Jetson):
#   bash deploy.sh
#
# To auto-update every 5 minutes add to crontab  (crontab -e):
#   */5 * * * * /home/user/chaos_brain/deploy.sh >> /home/user/deploy.log 2>&1

set -euo pipefail

REPO="https://github.com/gadgetlabs/agentic_control.git"
DIR="$HOME/agentic_control"
BRANCH="main"

# ── Helpers ───────────────────────────────────────────────────────────────
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

install_ollama() {
    if command -v ollama &>/dev/null; then
        echo "[deploy] ollama already installed ($(ollama --version))"
    else
        echo "[deploy] installing ollama ..."
        curl -fsSL https://ollama.com/install.sh | sh
    fi
}

pull_ollama_model() {
    # Read the model name from .env if it exists, else fall back to default
    local model
    model=$(grep -E "^OLLAMA_MODEL=" "$DIR/.env" 2>/dev/null | cut -d= -f2 || echo "qwen2.5:3b")
    echo "[deploy] pulling ollama model '$model' (skipped if already present) ..."
    ollama pull "$model"
}

# ── First run: clone and install everything ───────────────────────────────
if [ ! -d "$DIR/.git" ]; then
    echo "[deploy] cloning $REPO ..."
    git clone --branch "$BRANCH" "$REPO" "$DIR"
    cd "$DIR"

    install_ollama
    pull_ollama_model
    download_piper_voice

    pip install -r requirements.txt
    pip install git+https://github.com/gadgetlabs/simple-wake-word

    echo "[deploy] done. copy .env.example to .env, then: python $DIR/main.py"
    exit 0
fi

# ── Check for new commits without touching local files ────────────────────
cd "$DIR"
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

# Re-pull the model in case OLLAMA_MODEL changed in this update
pull_ollama_model
download_piper_voice

pip install -r requirements.txt --quiet

echo "[deploy] done. restart main.py to apply."

# ── Optional: restart systemd service if you set one up ──────────────────
# sudo systemctl restart chaos-brain
