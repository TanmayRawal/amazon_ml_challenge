# ADR 0007: Small cross-encoders on the uncertain band only

**Status:** Accepted

## Context

Hand-built features struggle with a few patterns: names in nine Indian scripts against their Latin spellings, paraphrased names, and subtle word swaps. Transformers read these better. But a cross-encoder over all 122.6M candidate pairs is far beyond a laptop GPU, and most pairs are already decided with near-certain probability.

## Decision

- **Pipeline A:** three cross-encoders fine-tuned on our training pairs (two paraphrase-multilingual-MiniLM-L12-v2, Apache-2.0, 118M; one multilingual-e5-base, MIT, 278M). They re-score only the **uncertain band** of the stacked probability. A small LightGBM "link" model combines the transformer scores with the stack probability; pairs outside the band keep their stack probability.
- **Pipeline B:** two MiniLM "pair judges", trained with different seeds on alternating data halves, are scored out of fold and enter the stacked stage as features.
- Every model is fine-tuned only on the provided training pairs. Pretrained weights are the only outside input, which the rules allow.

## Alternatives considered

- **Larger models (xlm-roberta-base, bge-reranker-v2-m3).** xlm-roberta-base was trained and tested: blended into the final probabilities, it lowered the held-out score at every weight. The five existing transformers already captured that signal.
- **A bi-encoder retrieval channel.** Recovered few matches ([ADR 0001](0001-blocking-then-learned-ranking.md)).
- **Off-the-shelf multilingual embeddings for Indian scripts.** Retrieval research ("Lost in Transliteration", 2025) shows sharp degradation across the native/Latin script gap. Rule-based transliteration plus consonant skeletons was cheaper and lifted Indic-target recall from 0.830 to 0.871, and later to 0.909.

## Consequences

- The transformer link added about +0.003 on the leaderboard (v18c 0.98216 → v20 0.985328, together with pipeline A's other changes).
- GPU time dominates the end-to-end run (15–20 hours in total).
- All models stay far below the 8B limit, under MIT or Apache-2.0 licences.
