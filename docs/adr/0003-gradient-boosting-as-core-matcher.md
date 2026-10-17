# ADR 0003: Gradient-boosted trees as the core matcher

**Status:** Accepted

## Context

The matcher must score more than 100M candidate pairs on consumer laptops (6–12 GB GPU, 16–32 GB RAM) within a three-day window. The rules allow only MIT or Apache-2.0 models with at most 8B parameters. We have 7.6M labelled positive pairs and many hard negatives from blocking.

## Decision

LightGBM (MIT) on engineered pair features is the core matcher in both pipelines:
- **Name:** token overlap, edit distances, Jaro-Winkler and fuzzy ratios, concatenated-name similarity, acronym and website coverage, legal-form relation, cross-script skeleton similarity, and the frequency of words that changed.
- **Address:** rarity-weighted overlap, street similarity, house-number relation (equal, near, truncated, substituted, signed offset), blank and website flags.

Stage-1 probabilities are calibrated with isotonic regression on out-of-fold scores, because the decision rule ([ADR 0005](0005-exact-expected-f05-decision.md)) needs calibrated probabilities.

Transformers are added only where they pay off ([ADR 0007](0007-cross-encoders-on-uncertain-band.md)).

## Alternatives considered

- **Logistic regression.** Our first baseline: sealed 0.8648 on the same candidates, against 0.9285 for the first LightGBM matcher.
- **Generative LLM matchers** (prompted or fine-tuned). Rejected on throughput (100M+ pairs), on licence (the strongest open matchers are Llama-based) and on evidence that fine-tuned LLM matchers transfer poorly across domains. France is a cross-domain setting.
- **A cross-encoder on every pair.** Too slow on our hardware, and unnecessary for the clear-cut majority of pairs.

## Consequences

- Training a matcher takes minutes, so we could iterate many times per day.
- Feature importance is inspectable, which made error analysis fast. For example, it showed that decoy features took 7 of the top 10 gains when they were introduced.
- Gains come from features and data design, not model size. Most of our improvements were new features or better training data.
