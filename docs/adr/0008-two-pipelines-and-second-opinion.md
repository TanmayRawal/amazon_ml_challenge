# ADR 0008: Two independent pipelines, a logit average and a second-opinion correction

**Status:** Accepted

## Context

By the last day both pipelines had plateaued near 0.989 on the sealed fold. They had been built independently, with different blocking (11 channels and a second hop, against 8 passes with TF-IDF search), different candidate depths (60 against 30) and different features. Their errors were therefore partly uncorrelated: 38% of the true pairs pipeline A never retrieved were in pipeline B's top 30.

## Decision

1. **Logit average for US and India.** Average the two pipelines' probabilities in log-odds space with weight 0.5 each. A pair proposed by only one pipeline keeps that pipeline's probability. The candidate file is the union of both candidate sets.
2. **Second opinion.** A small LightGBM (400 rounds, 31 leaves) corrects each averaged log-odds. It was trained out of fold on the decoy-injected training pool. Inputs:
   - both pipelines' log-odds, and which pipelines proposed the pair;
   - list context;
   - the transformer scores;
   - 23 structural conflict signals, such as house-number and street conflicts.

   Its main job is to put pairs seen by one pipeline on the same scale as pairs seen by both.
3. **France takes a separate path:** a guarded ensemble that cannot re-introduce known decoy classes ([ADR 0009](0009-label-free-validation-for-france.md)).

## Alternatives considered

- **Choosing the better pipeline.** Leaves +0.0013 (held out) on the table.
- **A learned stacker instead of the average.** +0.00036 over the average; superseded by the second opinion, which also uses list context.
- **Tuning the average weight.** Any weight between 0.3 and 0.7 scores within ±0.00005, so 0.5 was kept.

## Consequences

- Held-out: 0.9890 (A) and 0.9884 (B) → 0.9903 (average) → 0.9918 with the retrieval fix and second opinion (test-like setting).
- Leaderboard: 0.985328 (A alone) → 0.987776 (average with the first France rules) → 0.989165 (second opinion and corrections).
- The full system needs both pipelines' runs, roughly doubling compute. The one-command driver (`run_all.sh`) runs both in order.
- One side effect was caught by a label-free check: the second opinion over-accepted US digit-substitution pairs (test acceptance 0.0646 per record against a true rate of 0.0612). Those 1,602 pairs are removed by an explicit list.
