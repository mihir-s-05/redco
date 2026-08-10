"""Verify and summarize the compact terminal Stage-C4 selection-v4 bundle."""

from functools import partial

from redco.analysis import stage_c4_selection_bundle_verification as bundle

verify = partial(bundle.verify_factorized, spec=bundle.V4_SPEC)
main = partial(bundle.run_verification_cli, description=__doc__, verifier=verify)

if __name__ == "__main__":
    main()
