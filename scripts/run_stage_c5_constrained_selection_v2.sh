#!/usr/bin/env bash
set -euo pipefail

export REDCO_STAGE_C5_CAMPAIGN_VERSION=v2
export REDCO_STAGE_C5_RUN_SEED=7203006

exec bash "${REDCO_REPO_ROOT:-/workspace/redco}/scripts/run_stage_c5_constrained_selection_v1.sh"
