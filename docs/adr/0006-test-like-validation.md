# ADR 0006: Grouped cross-fitting, a sealed fold and decoy injection

**Status:** Accepted

## Context

- **Leakage risk.** Stacked models trained on in-sample stage-1 scores look excellent and fail on test.
- **Train/test shift.** Six independent label-free indicators (blank-address rate, website names, acronyms, name and address key hits) agree that 41–48% of test targets are ownerless, against 26% in training. A model validated on the training distribution would be too trusting on test.
- **Few uploads.** The leaderboard allowed 5 uploads per day, so it could not be our validation set.

## Decision

1. **Grouped folds.** Five folds over Source-1 records. Each group is one record plus all its targets, which is exact because every target has one owner. Folds are stratified by country, match count and presence of Indian-script targets.
2. **Cross-fitting.** Models trained on some folds score the others, so every stacked feature is out of fold.
3. **A sealed fold.** One fold is evaluated only at stage gates, never tuned on.
4. **Decoy injection.** A share of training records is hidden from the query set while their targets stay in the pool. Their copies become ownerless decoys, which reproduces the test's decoy density. This setting is used for training and for choosing decision parameters.
5. **Leaderboard as confirmation only.** An upload was spent only on a change that had already gained on held-out data beyond its confidence interval.

## Alternatives considered

- **Random pair-level splits.** Leak information through shared records.
- **Tuning on the public leaderboard.** Too few uploads, and it rewards noise.
- **Instance reweighting towards test.** Domain-adaptation literature (DADER, SIGMOD 2022) finds feature-level invariance better than reweighting. We preferred features that behave the same across regimes.

## Consequences

- Held-out gains transferred to the leaderboard almost one for one. For example, v5 → v7 gained +0.0236 on the sealed fold and +0.024 on the leaderboard.
- Stress tests quantified decoy sensitivity: the stage-2 model loses 0.0075 between 26% and 48% decoys, while staying far above the stage-1-only model.
- Implemented in `src/splits.py` and `src/stress.py`.
