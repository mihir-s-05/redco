"""Fixed no-argument runner for the authorization-child Prime test one-shot."""

from __future__ import annotations

import subprocess

from redco.analysis.stage_d_v13_prime_test_one_shot_contract_v2 import canonical_json
from redco.analysis.stage_d_v13_prime_test_one_shot_lifecycle_v2 import (
    run_prime_test_one_shot_v2,
)


def main() -> int:
    try:
        result = run_prime_test_one_shot_v2()
    except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as error:
        print(
            canonical_json(
                {
                    "schema_version": 2,
                    "state": "failed_terminal",
                    "failure": type(error).__name__,
                    "live_result": False,
                }
            ).decode()
        )
        return 20
    print(canonical_json(result.value()).decode())
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
