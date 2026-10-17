# ADR 0005: Exact expected-F0.5 decision instead of a fixed threshold

**Status:** Accepted

## Context

The metric is macro F0.5 per Source-1 record, with a special convention: if a record has no true match, an empty prediction scores 1 and any prediction scores 0. About 5.6% of records are such singletons. About 70% of singletons have a target with the exact same name, so a name match alone is a trap.

A fixed probability threshold ignores two things:
- **The rest of the list.** Whether adding a third candidate helps depends on how likely the first two are.
- **The value of an empty list.**

## Decision

For each record, sort candidates by calibrated probability. Then pick the prefix, including the empty one, with the highest **exact expected F0.5** (Ye et al., ICML 2012):

- The distribution of true matches inside and outside the chosen set is a Poisson-binomial, computed in O(n²) per record.
- An estimated rate of true matches that blocking missed enters as a small extra term (`missed`, 0.031 in the final system).
- The empty set's value is the probability that the record has no match at all.

Before deciding, two adjustments make the probabilities consistent:
1. **Single-owner renormalisation.** Each target belongs to exactly one record in the ground truth, so a target's probabilities across competing records are rescaled.
2. **Test-time odds shifts**, both validated on held-out data:
   - ×3 for blank-address and website targets, which are almost always genuine copies;
   - ×0.5 for same-street pairs whose house number differs, because test has 83–85% more digit-substitution decoys than training.

## Alternatives considered

- **Global threshold (tuned).** Dev 0.9230, against 0.9244 for the exact decision on the same model.
- **Ratio-of-expectations approximation.** Slightly wrong for short lists. Our unit tests compare the exact rule with brute-force enumeration over all 2ⁿ outcomes.
- **Per-country thresholds for France.** Tested as leaderboard probes: looser cost −0.002; tighter changed nothing. The decision was already calibrated.

## Consequences

- Singletons are handled naturally: a lone candidate is only predicted if its probability exceeds about 0.5 (exactly 0.5 when no missed matches are assumed).
- The decision is only as good as the calibration. This is why stage 1 uses isotonic calibration and why every model is validated under test-like decoy density.
- Implemented in `src/decide.py`; tested in `tests/test_decide_dta.py`.
