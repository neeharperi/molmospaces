#!/usr/bin/env bash
# Respawn wrapper for a policy server. Usage: serve_supervised.sh <name> <command...>
#
# Campaign 1 lost 8 hours to a TiPToP server that died mid-run while its client retried
# forever. That had two halves. The client half is fixed in tiptop_policy.py
# (RECONNECT_MAX_ATTEMPTS=15, ~6 minutes, then the cell fails loudly). This is the server
# half: without it, a dead server stays dead and every remaining cell in that lane fails.
#
# Together they turn that failure into roughly a one-minute blip -- the server respawns in
# 15s and the client's bounded reconnect is still well inside its budget when it comes back.
#
# The log is appended to, never truncated, so the restart history survives and "how many
# times did this crash?" is answerable after the fact.
set -uo pipefail
NAME="${1:?usage: serve_supervised.sh <name> <command...>}"; shift
DATE_TAG="${DATE_TAG:-manual}"
mkdir -p runs/_servers
LOG="runs/_servers/${NAME}_${DATE_TAG}.log"
while true; do
    echo "=== [$(date -Is)] starting $NAME: $* ===" >> "$LOG"
    "$@" >> "$LOG" 2>&1 &
    echo $! > "runs/_servers/${NAME}.pid"
    wait $!
    rc=$?
    echo "=== [$(date -Is)] $NAME exited rc=$rc; restarting in 15s ===" >> "$LOG"
    sleep 15
done
