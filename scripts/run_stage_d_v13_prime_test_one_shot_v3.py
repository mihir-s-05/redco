"""Fixed no-argument public entry point for the committed v3 one-shot."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from redco.analysis.stage_d_v13_prime_test_one_shot_contract_v2 import canonical_json  # noqa: E402
from redco.analysis.stage_d_v13_prime_test_one_shot_runtime_v3 import (  # noqa: E402
    run_prime_test_one_shot_v3,
)


def main() -> int:
    try:
        result = run_prime_test_one_shot_v3()
    except (OSError, subprocess.SubprocessError, RuntimeError, ValueError, TimeoutError):
        print(
            canonical_json(
                {"schema_version": 2, "state": "failed_terminal", "live_result": False}
            ).decode("utf-8")
        )
        return 20
    print(canonical_json(result.value()).decode("utf-8"))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
