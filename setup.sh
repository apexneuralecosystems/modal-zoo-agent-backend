#!/usr/bin/env bash
# One-time setup for a fresh Mac. Run ONCE after downloading + unzipping a chosen
# agent version onto the Mac. It:
#   1. installs Python dependencies into a shared .venv
#   2. creates config.json (you edit it: secret_token + branch_id)
#   3. builds the self-update folder layout:
#        <here>/versions/<version>/   <- the code (a copy of this bundle)
#        <here>/current_version       <- text: <version>
#        <here>/last_good             <- text: <version>
#        <here>/run.sh                <- launchd entrypoint (reads current_version)
#   4. installs the launchd service pointing at run.sh (KeepAlive)
#
# After this, future versions are applied by clicking Update in the dashboard —
# the Mac downloads the new zip into versions/<new>/, flips current_version, and
# restarts on the new code. You never touch this Mac again.
set -euo pipefail

AGENT_HOME="$(cd "$(dirname "$0")" && pwd)"
cd "$AGENT_HOME"

if [[ ! -f VERSION ]]; then
    echo "ERROR: no VERSION file next to setup.sh — is this a proper agent bundle?" >&2
    exit 1
fi
VERSION="$(tr -d '[:space:]' < VERSION)"
echo "Installing agent version $VERSION into $AGENT_HOME"

# ── 1. Python env + dependencies (shared across versions) ──────────────────
if [[ ! -d .venv ]]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# ── 2. config.json (shared; survives updates) ──────────────────────────────
if [[ ! -f config.json ]]; then
    cp config.example.json config.json
    echo "Created config.json — you must edit it with secret_token + branch_id."
fi

# ── 3. shared runtime dirs (survive updates) ───────────────────────────────
mkdir -p logs models_cache scripts_cache

# ── 4. build versions/<VERSION>/ from this bundle ──────────────────────────
# Copy the code into its version folder. Skip shared state + already-built dirs
# so we capture only the code (matches what a self-update zip unpacks).
mkdir -p "versions/$VERSION"
for item in *; do
    case "$item" in
        versions|.venv|venv|logs|models_cache|scripts_cache|config.json|current_version|last_good)
            continue ;;
        *)
            cp -R "$item" "versions/$VERSION/" ;;
    esac
done

# run.sh must sit at AGENT_HOME (launchd execs it); copy it up from the bundle.
cp "versions/$VERSION/run.sh" "$AGENT_HOME/run.sh"
chmod +x "$AGENT_HOME/run.sh"

# ── 5. version pointers ────────────────────────────────────────────────────
echo "$VERSION" > current_version
echo "$VERSION" > last_good

# ── 6. install the launchd service (points at run.sh, not main.py) ─────────
PLIST_DEST="$HOME/Library/LaunchAgents/com.apexneural.visionagent.plist"
mkdir -p "$HOME/Library/LaunchAgents"
cat > "$PLIST_DEST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.apexneural.visionagent</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/sh</string>
        <string>$AGENT_HOME/run.sh</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$AGENT_HOME</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$AGENT_HOME/logs/launchd-stdout.log</string>
    <key>StandardErrorPath</key>
    <string>$AGENT_HOME/logs/launchd-stderr.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PYTHONUNBUFFERED</key>
        <string>1</string>
    </dict>
</dict>
</plist>
PLIST

echo
echo "Setup complete for version $VERSION."
echo "Next steps:"
echo "  1. Edit config.json  (set secret_token + branch_id)"
echo "  2. Start the agent:"
echo "       launchctl unload \"$PLIST_DEST\" 2>/dev/null || true"
echo "       launchctl load \"$PLIST_DEST\""
echo
echo "From the next version on, just click Update in the dashboard — no Mac access needed."
