#!/usr/bin/env bash
# Regenerate tests/fixtures/*.json from tools/scenarios/*.json using the real
# vLLM scheduler (vllm-project/vllm @ adc3e03517, CPU-only .venv).
set -euo pipefail

VLLM_ROOT="${VLLM_ROOT:?set VLLM_ROOT to your vllm checkout}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/../tests/fixtures"
mkdir -p "$OUT"

for scenario in "$HERE"/scenarios/*.json; do
  name="$(basename "$scenario" .json)"
  if [[ $# -gt 0 ]] && [[ " $* " != *" $name "* ]]; then continue; fi
  (
    cd "$VLLM_ROOT"
    PYTHONHASHSEED=0 PYTHONPATH=. "$VLLM_ROOT/.venv/bin/python" \
      "$HERE/trace_vllm.py" "$scenario" "$OUT/$name.json" 2>/dev/null
  )
done
