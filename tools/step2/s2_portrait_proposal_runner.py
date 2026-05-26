"""Compatibility wrapper for the generic proposal runner.

The implementation moved to `tools.proposals.proposal_runner`. This wrapper is
kept so older local commands and tests that import the S2-specific path do not
break during the transition.
"""

from __future__ import annotations

from tools.proposals.proposal_runner import main, run_proposal_runner


if __name__ == "__main__":
    raise SystemExit(main())
