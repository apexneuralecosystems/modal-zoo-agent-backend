#!/bin/sh
# launchd entrypoint for the self-updating Mac agent.
#
# launchd (KeepAlive:true) execs THIS script, not main.py directly. We read the
# `current_version` pointer file and exec that version's main.py. When the agent
# applies an update it rewrites current_version and exits; launchd re-execs this
# script, which then launches the new version. Rollback works the same way: the
# watchdog rewrites current_version back to last_good and exits.
#
# Layout (see agent_paths.py):
#   <AGENT_HOME>/run.sh                (this file)
#   <AGENT_HOME>/current_version       (text: e.g. "1.1.0")
#   <AGENT_HOME>/versions/<v>/main.py  (the code)
set -eu

AGENT_HOME="$(cd "$(dirname "$0")" && pwd)"
VERSION="$(cat "$AGENT_HOME/current_version")"
CODE_DIR="$AGENT_HOME/versions/$VERSION"

# Pick a Python: prefer the shared venv created by setup.sh, else system python3.
if [ -x "$AGENT_HOME/.venv/bin/python3" ]; then
  PY="$AGENT_HOME/.venv/bin/python3"
elif [ -x "$AGENT_HOME/venv/bin/python3" ]; then
  PY="$AGENT_HOME/venv/bin/python3"
else
  PY="$(command -v python3 || echo /usr/bin/python3)"
fi

exec "$PY" "$CODE_DIR/main.py"
