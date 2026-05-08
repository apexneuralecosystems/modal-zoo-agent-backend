#!/usr/bin/env bash
# One-shot setup for the Mac Mini. Run once after cloning the agent code onto the Mac.
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -d .venv ]]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

if [[ ! -f config.json ]]; then
    cp config.example.json config.json
    echo "Created config.json — edit it with secret_token + branch_id before launching."
fi

mkdir -p logs models_cache scripts_cache

echo "Setup complete. Edit config.json then run: source .venv/bin/activate && python main.py"
echo "To install as a launchd service:"
echo "  cp com.apexneural.visionagent.plist ~/Library/LaunchAgents/"
echo "  launchctl load ~/Library/LaunchAgents/com.apexneural.visionagent.plist"
