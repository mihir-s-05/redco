#!/usr/bin/env bash
set -euo pipefail

workspace="${1:-/workspace}"
archive="$workspace/redco-stage-d0-qasper-src.tar.gz"
archive_sha256="356af61f704be507ce52212fcce9a1468fa82c92b91d90943d3fca7d5701cec9"
prime_commit="3b22dd951cad1036d1fe8dd0a0bfc40807a9b360"
verifiers_commit="b13ba60da63cea91389e7575766b7270d0d11fc5"
uv_bin="/home/ubuntu/.local/uv-latest/uv"

test "$(sha256sum "$archive" | cut -d ' ' -f 1)" = "$archive_sha256"
test ! -e "$workspace/redco"
mkdir "$workspace/redco"
tar -xzf "$archive" -C "$workspace/redco"

if ! test -x "$uv_bin"; then
  curl -LsSf https://astral.sh/uv/0.12.0/install.sh |
    env UV_INSTALL_DIR=/home/ubuntu/.local/uv-latest sh
fi
"$uv_bin" --version

git clone --filter=blob:none --no-checkout \
  https://github.com/PrimeIntellect-ai/prime-rl.git \
  "$workspace/redco/external/prime-rl"
git -C "$workspace/redco/external/prime-rl" checkout "$prime_commit"
git -C "$workspace/redco/external/prime-rl" submodule update \
  --init --depth 1 deps/pydantic-config deps/renderers deps/verifiers

test "$(
  git -C "$workspace/redco/external/prime-rl" rev-parse HEAD
)" = "$prime_commit"
test "$(
  git -C "$workspace/redco/external/prime-rl/deps/verifiers" rev-parse HEAD
)" = "$verifiers_commit"
printf 'REMOTE_SOURCE_READY\n'
