#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
prime_source="$repo_root/external/prime-rl"
prime_commit="3b22dd951cad1036d1fe8dd0a0bfc40807a9b360"
temp_root="$(mktemp -d)"
prime_worktree="$temp_root/prime-rl"
if test -x "$HOME/.local/uv-latest/uv"; then
  uv_bin="$HOME/.local/uv-latest/uv"
else
  uv_bin="$(command -v uv)"
fi

cleanup() {
  git -C "$prime_source" worktree remove --force "$prime_worktree" \
    >/dev/null 2>&1 || true
  rm -rf -- "$temp_root"
}
trap cleanup EXIT

git -C "$prime_source" worktree add --detach "$prime_worktree" "$prime_commit"
git -C "$prime_worktree" config url.https://github.com/.insteadOf git@github.com:
git -C "$prime_worktree" submodule update --init \
  deps/pydantic-config \
  deps/renderers \
  deps/verifiers
git -C "$prime_worktree" apply "$repo_root/patches/prime-rl-redco-noop.patch"
cd "$prime_worktree"

export PYTHONPATH="$prime_worktree/src:$prime_worktree"
"$uv_bin" run \
  --no-project \
  --python 3.12 \
  --index https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match \
  --with pytest \
  --with 'torch==2.9.1+cpu' \
  --with msgspec \
  --with beartype \
  --with jaxtyping \
  --with wandb \
  --with-editable ./deps/pydantic-config \
  --with-editable ./deps/renderers \
  --with-editable ./deps/verifiers \
  --with-editable ./packages/prime-rl-configs \
  python -m pytest -q \
    tests/unit/orchestrator/test_advantage.py \
    tests/unit/orchestrator/test_redco_noop_integration.py
