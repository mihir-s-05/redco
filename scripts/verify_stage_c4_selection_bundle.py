"""Verify and summarize a compact terminal Stage-C4 selection bundle."""

from functools import partial

from redco.analysis import stage_c4_selection_bundle_verification as bundle

verify = bundle.verify_v2_bundle
main = partial(bundle.run_verification_cli, description=__doc__, verifier=verify)

if __name__ == "__main__":
    main()
