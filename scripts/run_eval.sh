#!/usr/bin/env bash
# Wrapper for canonical eval runs. Logs cost-time-stamp for the COSTS.md ledger.
set -euo pipefail

MODEL="${1:-mock}"
ENV="${2:-mock-libero}"
TASKS="${3:-}"
ROLLOUTS="${4:-20}"

TASKS_ARG=()
if [[ -n "$TASKS" ]]; then
    TASKS_ARG=(--tasks "$TASKS")
fi

START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "[$START] Starting eval: model=$MODEL env=$ENV rollouts=$ROLLOUTS"
echo "[$START]   (Remember to log spend in docs/COSTS.md if a paid GPU is in use.)"
echo

vla-bench eval \
    --model "$MODEL" \
    --env "$ENV" \
    --rollouts "$ROLLOUTS" \
    "${TASKS_ARG[@]}"

END=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo
echo "[$END] Completed."
