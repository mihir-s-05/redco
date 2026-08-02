#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="/tmp/rlm/src${PYTHONPATH:+:$PYTHONPATH}"
exec /tmp/vf-rlm/env/bin/python -m rlm.cli "$@"
