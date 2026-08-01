#!/usr/bin/env bash
set -euo pipefail

if test "${REDCO_REPLAY_NETNS_ACTIVE:-0}" != "1"; then
  export REDCO_REPLAY_NETNS_ACTIVE=1
  exec unshare --user --map-root-user --net bash -lc \
    'ip link set lo up; exec bash "$1" "$2"' _ "$0" "${1:?output required}"
fi

test "$(uname -s)" = "Linux"
repo_root="/mnt/c/Users/mihir/Documents/redco"
output="$(realpath -m -- "${1:?output required}")"
uv_bin="/home/mihir/.local/uv-latest/uv"
project="$repo_root/.redco/vendor/rlm-56218f3"
replay_tmp="$(mktemp -d /home/mihir/.cache/redco/rlm-replay.XXXXXX)"
replay_tmp="$(readlink -f -- "$replay_tmp")"
case "$replay_tmp" in
  /home/mihir/.cache/redco/rlm-replay.*) ;;
  *) echo "unsafe replay temp path" >&2; exit 1 ;;
esac
cleanup() {
  rm -rf -- "$replay_tmp"
}
trap cleanup EXIT

export UV_PROJECT_ENVIRONMENT="/home/mihir/.venvs/redco-rlm-replay"
export UV_CACHE_DIR="/home/mihir/.cache/uv-redco"
export PYTHONPATH="$repo_root/src:$repo_root"

cd "$project"
"$uv_bin" run --frozen --no-dev python \
  "$repo_root/scripts/run_stage_d_rlm_replay_fixture.py" \
  --cassette "$repo_root/tests/fixtures/stage_d_rlm_replay_cassette_v1.json" \
  --workspace "$replay_tmp/workspace" \
  --output "$output" \
  --require-network-blocked
