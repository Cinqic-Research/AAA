# Promotion contract `aaa.promotion.crossed.v1`

The versioned decision contract that repairs `AAA-180` **prospectively**. The
code is in [`aaa/promotion/`](../aaa/promotion/). The frozen `aaa.1k.v2` K5 path
is unchanged, still reproduces, and is listed in `FORBIDDEN_CONTRACTS`: it may
not adjudicate a promotion again.

## What was wrong in `aaa.1k.v2`

1. The frozen `run_confirmation` collected Monash evidence but called
   `decide` without `monash_primitives`, so K5 would have been decided on 15
   of its 19 declared groups.
2. With Monash supplied, the primary `stats.geometric_relative` returned
   `INSUFFICIENT_EVIDENCE` for the whole geometric mean. Monash `saugeen` has
   one series, and the crossed bootstrap refuses fewer than two streams. The
   independent `recompute` still measured an interval, because a
   multinomial over one series always weighs it 1. The two K5
   implementations therefore disagreed: `INCONCLUSIVE` against `FAIL` for all
   six arms with Monash cells.
3. `recompute` added the Monash groups only when the challenger had Monash
   primitives, so a challenger without them would silently be adjudicated on
   fewer groups (`AAA-182`).

No v2 decision changed: the phase had no challenger (`NO_CHALLENGER`). The
defect is part of the historical record and stays in the frozen source.
`python tools/check_promotion_successor.py` reproduces it next to the
successor's result.

## The successor

**Estimand.** For each declared group, the ratio of means
`mean(challenger)/mean(reference)` over its complete declared
`initialization x series` grid, combined across groups by a geometric mean.
Groups are the declared suite and are never resampled.

**Group designs, declared before observation.**

- `crossed` (two or more series): each bootstrap draw resamples the shared
  initializations once and that group's series. The interval describes the
  population of series.
- `conditional_on_single_series` (exactly one series): the sole series is
  held fixed and only the shared initializations are resampled. The group's
  contribution is **conditional on the observed series** and says nothing about
  other series from the same source. A result that includes such a group has
  scope `conditional_on_observed_series` and names the group; a population
  claim requires every group to be `crossed`.

The contract refuses a crossed group with one series and a conditional group
with more than one. No series dimension is ever fabricated. This rule was
chosen because it keeps the estimand unchanged, uses every declared group,
and adds no unobserved variance. Excluding the group would change the
estimand; treating one series as a population would fabricate one. It was
fixed from those properties, not from any verdict. Fewer than two
initializations leave nothing to resample, and every criterion is then
`INSUFFICIENT_EVIDENCE`.

**Criteria and verdicts.** `noninferior`: PASS if upper <= threshold, FAIL if
lower > threshold, else INCONCLUSIVE. `superior`: PASS if upper < threshold,
FAIL if lower >= threshold, else INCONCLUSIVE. A criterion without a measured
interval is `INSUFFICIENT_EVIDENCE`, never PASS. The verdict is
intersection-union: any FAIL gives `REJECT`; every criterion PASS gives
`PROMOTE`; otherwise `INSUFFICIENT_EVIDENCE` or `INCONCLUSIVE`.

**Two structurally independent implementations.** The primary
([`primary.py`](../aaa/promotion/primary.py)) validates into pre-shaped grids
and resamples by index, in chunks. The recomputation
([`independent.py`](../aaa/promotion/independent.py)) validates by comparing
identity sets, assembles grids by sorting, resamples with multinomial count
weights (an explicit `[1]` weight for a conditional group), draws from a
separately salted stream, and re-implements the status and verdict rules from
their text. It imports only the contract's declared values and error type,
and a test enforces that.

**Fail-closed adjudication** ([`adjudicate.py`](../aaa/promotion/adjudicate.py)).
A verdict exists only if both implementations agree on:

- the point value, to a relative 1e-12;
- the interval status;
- both bounds, within 0.05 interval widths;
- every criterion;
- the verdict.

The 0.05 tolerance is more than seven Monte Carlo standard errors of the
difference of two independent 2.5% percentile estimates at the default 20,000
draws (derivation in the contract module). Any disagreement is `DISAGREEMENT`.
Missing, extra, duplicated, malformed, non-finite, non-positive or
identity-mismatched primitives are `INVALID_EVIDENCE`. Neither verdict ever
promotes.

## Evidence that it behaves

- `python -m aaa.promotion selftest`: synthetic fixtures with multiple and
  single series in every verdict region; an omitted group is `INVALID_EVIDENCE`;
  the frozen v2 contract is refused.
- `tests/test_promotion.py`: agreement of point value, interval status,
  bounds, criteria and verdict on multi-series and single-series fixtures. An
  omitted required group, including one omitted for one arm only, cannot
  promote a challenger that would otherwise pass. Extra groups, mismatched
  initialization or series identities, missing and duplicate cells,
  non-finite, zero, negative, `None`, string and boolean values, and malformed
  records are all refused by *each* implementation. Injected point, bound,
  status, criterion and verdict disagreements all yield `DISAGREEMENT`. A
  check confirms that the single-series interval equals an explicit
  initialization-only bootstrap.
- `python tools/check_promotion_successor.py`: on the retained v2 K5 groups
  (19 groups, Saugeen conditional) both implementations agree for all six arms,
  descriptively. Those arms were never challengers, so this is not a promotion
  decision.

## Using it

A future protocol declares its groups (with designs), initializations, series,
criteria, thresholds, seed and draws in a `Contract` **before** any held-out
observation, commits it with its source identity, and passes primitives
`{arm, group, init, series, value}` to `adjudicate`. `aaa.python.v0` names this
contract as its future promotion path but declares no candidate or criteria.
