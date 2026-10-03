#!/usr/bin/env python3
"""Fail-closed tombstone for the retired count-only PDF batch workflow."""
import sys


def main():
    print("Deprecated: the historical count-only PDF batch/resume workflow is disabled. "
          "Use the CCTS pipeline. Archived PDF scripts/results are reference-only; "
          "no batches may be reused or executed through this entry point. "
          "Future reactivation requires reviewed ordered PDF/checkpoint/output identities.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
