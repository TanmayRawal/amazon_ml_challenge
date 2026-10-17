# ADR 0009: France — rules validated without labels

**Status:** Accepted

## Context

France is 15% of the test records and has no training labels. Early estimates put French accuracy at about 0.88–0.90, against 0.97+ for the US and India, which made it the largest single source of loss. French data differs in ways our US/India models never saw:
- **Repeated names.** French names come from a small vocabulary; "Bordeaux Club SARL" appears 205 times.
- **Accent-deleting websites.** Website names delete accented letters ("École" becomes `cole`).
- **Acronyms.** Acronym names are about 20 times more common than in the US.
- **Shared addresses.** 40% of French records share a street address with another record.
- **Decoy offsets.** House-number decoys shift the number upwards by fixed offsets.

## Decision

We treat France as an analysis problem. For each generator operation, we ask whether it creates decoys or copies, and we encode the answer as a rule. A rule is adopted only if it passes **all** applicable label-free checks:

1. **Same pattern in labelled countries.** The rule's US/India analogue must be highly accurate in training (for example acronyms 99.8–99.9%, accent-deleting domains 99.99%, invented brand names 98.3–98.6%).
2. **Class volumes.** For each edit type, accepted pairs per record on test must not exceed the true rate per record in training. The generator is the same, so an excess signals false matches.
3. **Blank-address test.** Blank-address records are almost always owned, so a genuine copy pattern appears among them and a decoy pattern does not.
4. **Agreement with confident decisions.** A rule must agree with the model's own confident French decisions (for example 99.87% for acronyms, 100% for formatting-based tie resolution).
5. **Leaderboard probes, sparingly.** A few uploads only confirmed the direction of a change.

Final French rules, among others:
- **Removals:** decoy offsets, legal-form swaps on identical names, one-word common-word swaps.
- **Additions:** accent-deleting domains (+1,413), acronyms (+2,153), legal-form and formatting tie resolution (+2,839), invented brand names (+2,125).
- **Guarded ensemble:** +1,330 / −173 pairs.

## Alternatives considered

- **Moving France's decision threshold.** Probed on the leaderboard: looser −0.002, tighter no change. France's gap is discrimination, not calibration.
- **Plain two-pipeline average in France.** It would have dropped about 6,500 true list-word copies, so it was replaced by the guarded version.
- **Synthetic French training data.** Promising, but not feasible in the time window; listed as future work.

## Consequences

- France rules contributed most of the final +0.0017 (0.987776 → 0.989462).
- France risk is symmetric between the public and private leaderboards: a wrong rule would be equally wrong on both, so it cannot produce a public-only gain.
- The rules are specific to this data generator. They are not general-purpose French business-matching rules.
