"""``python -m aaa.promotion``: contract identity and the successor self-test.

    python -m aaa.promotion contract-id
    python -m aaa.promotion selftest     # exit 1 unless every fixture behaves as declared

The self-test adjudicates synthetic fixtures only. It observes no candidate
evidence and makes no promotion decision.
"""

from __future__ import annotations

import argparse
import sys

from . import CONTRACT_ID, FORBIDDEN_CONTRACTS, ContractError, adjudicate, fixtures, require_admissible


def selftest() -> list[str]:
    failures: list[str] = []
    cases = [
        ("multi-series groups, 15% better", [5, 4, 8], 0.85, {"PROMOTE"}),
        ("multi-series + one single-series group, 15% better", [5, 4, 1], 0.85, {"PROMOTE"}),
        ("multi-series + one single-series group, no effect", [5, 4, 1], 1.0, {"INCONCLUSIVE"}),
        ("multi-series + one single-series group, 40% worse", [5, 4, 1], 1.4, {"REJECT"}),
    ]
    for label, series, effect, allowed in cases:
        contract = fixtures.contract(series)
        result = adjudicate(contract, fixtures.primitives(contract, effect=effect))
        primary, independent = result.get("primary", {}), result.get("independent", {})
        print(
            f"{label}: verdict {result['verdict']}; scope {contract.scope}; "
            f"primary {primary.get('geometric_ratio'):.6f} [{primary.get('lower'):.4f}, {primary.get('upper'):.4f}] "
            f"independent {independent.get('geometric_ratio'):.6f} "
            f"[{independent.get('lower'):.4f}, {independent.get('upper'):.4f}]"
        )
        if result["verdict"] not in allowed:
            failures.append(f"{label}: {result['verdict']} not in {sorted(allowed)}")
    contract = fixtures.contract([5, 4, 1])
    records = [r for r in fixtures.primitives(contract) if r["group"] != "g2"]
    omitted = adjudicate(contract, records)
    print(f"declared group omitted: verdict {omitted['verdict']} ({omitted.get('reason')})")
    if omitted["verdict"] != "INVALID_EVIDENCE":
        failures.append("an omitted declared group did not fail closed")
    for forbidden in FORBIDDEN_CONTRACTS:
        try:
            require_admissible(forbidden)
        except ContractError:
            print(f"{forbidden}: forbidden for promotion")
        else:
            failures.append(f"{forbidden} was admitted")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m aaa.promotion", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("contract-id", help="print the current promotion contract identity")
    sub.add_parser("selftest", help="adjudicate the synthetic fixtures and check every expected verdict")
    args = parser.parse_args(argv)
    if args.command == "contract-id":
        print(CONTRACT_ID)
        return 0
    failures = selftest()
    for failure in failures:
        print(f"FAILED: {failure}", file=sys.stderr)
    print("selftest: " + ("FAILED" if failures else "PASS"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
