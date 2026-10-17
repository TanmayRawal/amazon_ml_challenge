# ADR 0004: A stacked second stage that reads each record's whole candidate list

**Status:** Accepted

## Context

The data generator creates decoys by copying a real business and changing one detail: swapping the legal form (Pvt ↔ Public, LLC ↔ Inc, SARL ↔ SAS), replacing one house-number digit, or swapping one name word for another common word. Seen in isolation, such a decoy is often more similar to the Source-1 record than a noisy true copy is.

What gives a decoy away is its relation to the other candidates. A copy whose house number disagrees with every other confident candidate of the same record is suspicious. A copy whose number agrees with them is probably genuine, even if it disagrees with the Source-1 record itself. Error analysis showed that 95% of our false matches pointed at ownerless decoys, and that 98% of them had probability above 0.5: confident mistakes.

## Decision

A second LightGBM stage takes the out-of-fold stage-1 probabilities and adds features computed over each record's whole candidate list:
- **Consensus:** probability-weighted agreement of the candidate's house number, core name and legal form with the record's other candidates.
- **Competition:** the strongest other Source-1 record claiming the same target, and the gap to it.
- **Decoy relations:** legal-form swap / added / dropped, unmatched house numbers, smallest numeric offset.
- **Vocabulary:** how common the changed name words are. A typo creates a rare word; a decoy swaps in a common one.

Stage-1 probabilities used as stage-2 inputs are always out of fold ([ADR 0006](0006-test-like-validation.md)), so stage 2 never sees leaked labels.

## Alternatives considered

- **Pairwise features only.** Cannot express "disagrees with its siblings".
- **A third stacking level.** Its validation loss equalled stage 2's, and it stopped after about 110 trees: no gain.
- **Graph clustering after matching.** Our ground truth is a strict star (each target has exactly one owner), which renormalisation already handles ([ADR 0005](0005-exact-expected-f05-decision.md)).

## Consequences

- Introducing the decoy features moved the sealed fold from 0.9431 to 0.9666 in one step. Validation log loss halved (0.019 → 0.009), and false-match pairs fell from 29,271 to 7,333.
- The consensus features make the model somewhat more sensitive to decoy density, which motivated decoy injection in training ([ADR 0006](0006-test-like-validation.md)).
