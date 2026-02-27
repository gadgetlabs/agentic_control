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

REPO="https://github.com/YOUR_USERNAME/chaos-brain.git"
DIR="$HOME/chaos_brain"
BRANCH="main"

# ── First run: clone and install ──────────────────────────────────────────
if [ ! -d "$DIR/.git" ]; then
    echo "[deploy] cloning $REPO ..."
    git clone --branch "$BRANCH" "$REPO" "$DIR"
    cd "$DIR"
    pip install -r requirements.txt
    pip install git+https://github.com/gadgetlabs/simple-wake-word
    echo "[deploy] installed. copy .env.example to .env and run: python $DIR/main.py"
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
pip install -r requirements.txt --quiet

echo "[deploy] done. restart main.py to apply."

# ── Optional: restart systemd service if you set one up ──────────────────
# sudo systemctl restart chaos-brain
