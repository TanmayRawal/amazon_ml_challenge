# ADR 0001: Multi-channel blocking plus a learned ranker

**Status:** Accepted

## Context

There are 1.73M test Source-1 records and about 10M target records. Scoring every pair within a country is impossible: it would be on the order of 10¹² pairs. Whatever the candidate generator misses can never be matched, so blocking sets the ceiling on recall.

Our audit of the 7.6M true training pairs showed that no single key is enough:
- 8% (US) to 29% (India) of true pairs share no name word at all: acronyms, website names, invented trade names, Indian scripts.
- 6–7% of US pairs have house numbers that disagree, because of typos and truncation.
- About 4–5% of targets have a blank address.
- Common words make huge blocks: "center" appears in 293k US names. Postal codes are effectively absent, so the usual postcode key does not exist.

## Decision

1. **Many narrow keys, each aimed at a measured failure mode.** Exact normalised name, rare name words and word pairs, house number plus street words, rare address words, typo keys, website 4-grams and concatenated-name prefixes, Indic consonant skeletons, house numbers with one digit dropped, and rare name word × address word. Pipeline B adds a character-trigram TF-IDF search for heavily misspelled names. Keys have document-frequency caps, so common words never create mega-blocks.
2. **A learned ranker instead of a hand-weighted cut.** The union of all channels is about 400 candidates per record. A cross-fitted LightGBM ranker, trained on blocking-graph features (which channels hit, block sizes, per-channel ranks) and cheap similarities, keeps the best 60 (pipeline A) or 30 (pipeline B).
3. **A second hop.** Records that share an exact address with one of a record's confident matches are added, if their names agree.

## Alternatives considered

- **Single key or few keys, plain truncation.** Recall 0.859 at 38 candidates per record, oracle F0.5 0.932. Too low.
- **Dense embedding retrieval only.** A multilingual e5 channel recovered few extra matches, and generic French names ("Bordeaux Club") embed almost identically. Lexical address evidence cannot be replaced.
- **Larger K (60 → 100).** Recall stops growing after about 40 candidates; at most +0.00005 possible.

## Consequences

- Sealed-fold recall rose from 0.859 (hand keys) to 0.952 (first learned ranker), then to 0.989 at the best 30 in pipeline B. A perfect matcher on those candidates would score 0.9965.
- Most remaining blocking misses are invented names at truncated addresses. They look identical to decoys at the same address, so retrieving them would mainly add false matches.
- The candidate set is large (122.6M test pairs in total). Feature building is therefore chunked per country.
