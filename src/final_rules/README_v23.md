# v23: v22c (LB 0.989359) plus a guarded France ensemble and small validated fixes

**Checks**
- Official validator (`--check-ids`): PASS, no warnings.
- `matching_results.tsv`: SHA-256 starts `93233675b19e36c9`.
- Cross-country pairs: 0.
- Targets claimed by 2 S1s: 6, all exact-probability ties between duplicate S1s.
- Groups over the per-source cap (5 in S2, 6 in S3): 0.
- Pairs per S1: US 3.387, India 3.374, France 3.321.
- Total vs the 0.989165 base: +5,443 pairs, −306 pairs. `added.csv` / `removed.csv` hold the entity-ID lists; the parquet files per rule are in `changes/`.

| Change | Pairs | Evidence |
|---|---|---|
| v22c rules (acronyms, formatting ties, stop-word ties, French decoys) | +3,706 / −89 | LB +0.000194 (v22c) |
| **Guarded France ensemble** (V20 ⊕ v18, w = 0.7 on the V20 logit) | +1,330 / −173 | See below |
| US/India domain rule (India limited to targets with a street address) | +87 | Train precision 98.6% on targets the model left unclaimed (US 100%) |
| Recovered US/India pairs (R1: never retrieved, name ≥90, same numbers and street, unique S1) | +49 | Train 94–96% |
| French ties at posterior 0.85–0.9, model p ≥ 0.2 | +271 | Tie posterior; 22/22 agreement with claimed ties in the same band |
| Double-claimed targets: keep the higher-p claimant | −10 | Every target has at most one owner |
| Over-cap groups: lowest-p extras removed | −34 | The generator never exceeds 5 S2 / 6 S3 copies per S1 (train) |

## Guarded France ensemble: how it was built

France had never received the second pipeline's opinion.
1. French p is the logit average of the V20 stack and our v18 stack.
2. I rebuilt the France rule chain approximately (vocab swaps, suffix ×5, audit rules 1/3/4, list-word floor, `dta_own`). With V20 p alone, it matches the uploaded French rows to within about 2%.
3. Only the *difference* between the chain run on the ensemble p and on V20 p is applied to the uploaded rows.

**Guards:**
- **Additions** only in true-copy classes at the same address: exact names, list-word copies, dropped words, typos, web names, and copies at a second house number that keep their legal form.
- **Blank additions** only for single-owner targets that add a legal form, or for ties the formatting posterior confirms.
- **Removals** only for added-word pairs where the street disagrees, and for exact names whose house number changed and legal form was added.
- **Protected:** list-word copies, suffix and acronym copies, domain pairs and tie pairs are never removed.
- **Rejected from the raw ensemble:** about 6,500 removals of true list-word copies (v18 lacks the France suffix rule) and about 3,000 same-address alias additions of unknown precision.

**Expected:** about +0.00005 over v22c, so roughly 0.98940 (range 0.98935–0.98950). If the score is below 0.989359, re-upload v22c as the last submission.
