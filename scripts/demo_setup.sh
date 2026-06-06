#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

export PYTHONPATH="${PYTHONPATH:-src}"

mkdir -p data/incidents data/qdrant data/approvals data/vision_outputs models

if [[ ! -f "data/incidents/ai4i_incident_corpus.jsonl" ]]; then
  "${PYTHON_BIN}" -m industrial_ai.incidents.generate --source-failure-rows "${SOURCE_FAILURE_ROWS:-100}"
fi

"${PYTHON_BIN}" -m industrial_ai.incidents.memory index

echo "Demo artifacts prepared:"
echo "- data/incidents/ai4i_incident_corpus.jsonl"
echo "- data/qdrant/"
echo "- data/approvals/"
echo "- models/"
