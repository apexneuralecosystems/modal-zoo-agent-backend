# One-time bootstrap: versioned layout + self-update

This is the **last manual (AnyDesk) step per Mac**. After it, every future agent
update is self-served from the cloud — no more visiting machines.

## What changes

Today the agent runs from one fixed folder. We convert it to:

```
<AGENT_HOME>/
  versions/1.0.0/        # the current code (main.py, *.py, VERSION, …) moves here
  current_version        # text file containing: 1.0.0
  last_good              # text file containing: 1.0.0
  run.sh                 # launchd execs this; it reads current_version
  config.json            # stays at AGENT_HOME (shared, survives updates)
  logs/  models_cache/  scripts_cache/   # shared, survive updates
```

launchd points at `run.sh` (not `main.py`). `run.sh` reads `current_version` and
launches `versions/<that>/main.py`. To update, the agent unpacks a new
`versions/<new>/`, rewrites `current_version`, and exits — launchd re-execs
`run.sh` on the new code. The startup watchdog reverts `current_version` to
`last_good` if the new version doesn't heartbeat healthy within 5 minutes.

## Steps (run on the Mac, in a shell)

Assume the agent currently lives in `~/vision-agent` with `main.py` etc. directly
inside it. Set `AGENT_HOME` accordingly.

```sh
AGENT_HOME=~/vision-agent
cd "$AGENT_HOME"

# 1. Stop the running agent so files aren't in use.
launchctl unload ~/Library/LaunchAgents/com.apexneural.visionagent.plist 2>/dev/null || true

# 2. Move the code into versions/1.0.0/ (keep shared state at AGENT_HOME).
mkdir -p versions/1.0.0
# Move every code file (*.py + VERSION) but NOT the shared state dirs/files.
for f in *.py VERSION requirements.txt; do
  [ -e "$f" ] && mv "$f" versions/1.0.0/
done
# Bring run.sh up to AGENT_HOME (it ships inside the code bundle).
cp versions/1.0.0/run.sh ./run.sh 2>/dev/null || true
chmod +x ./run.sh

# 3. If config.json / logs / caches were INSIDE the code folder, move them up.
for f in config.json logs models_cache scripts_cache; do
  [ -e "versions/1.0.0/$f" ] && mv "versions/1.0.0/$f" ./
done

# 4. Seed the version pointers.
echo 1.0.0 > current_version
echo 1.0.0 > last_good
```

### Update the launchd plist

Edit `~/Library/LaunchAgents/com.apexneural.visionagent.plist` so
`ProgramArguments` execs `run.sh` (keep `KeepAlive` = true):

```xml
<key>ProgramArguments</key>
<array>
  <string>/bin/sh</string>
  <string>/Users/<user>/vision-agent/run.sh</string>
</array>
<key>KeepAlive</key>
<true/>
<key>WorkingDirectory</key>
<string>/Users/<user>/vision-agent</string>
```

Then reload:

```sh
launchctl load ~/Library/LaunchAgents/com.apexneural.visionagent.plist
```

## Verify

- `tail -f "$AGENT_HOME/logs/agent.log"` — the agent boots and logs
  `Vision AI Mac Agent starting … ver=1.0.0`.
- In the cloud super-admin Mac page, the Mac heartbeats and shows **agent v1.0.0**.
- `cat "$AGENT_HOME/last_good"` stays `1.0.0` (watchdog promoted it as healthy).

## First real self-update (smoke test)

1. Build a zip of the new agent code with a bumped `VERSION` (e.g. `1.0.1`) at the
   zip root, upload it via the cloud presigned URL, and register it
   (`POST /platform/agent-releases`).
2. On the Mac's detail page, click **Update to v1.0.1**.
3. Watch `logs/agent.log`: `upgrade … applied -> 1.0.1 (exiting for restart)`,
   then a fresh boot at `ver=1.0.1`, then `version 1.0.1 confirmed healthy`.
4. `cat current_version` → `1.0.1`; `cat last_good` → `1.0.1`.

To prove rollback: register a deliberately-broken `1.0.2` (e.g. a `main.py` that
exits immediately), click Update, and confirm that within ~5 min `current_version`
reverts to `1.0.1` and the cloud still shows v1.0.1.
