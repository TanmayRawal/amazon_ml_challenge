# v24 = v23 + French aliases + singleton guard

**Checks:**
- Official validator (`--check-ids`): PASS.
- `matching_results.tsv`: SHA-256 starts `519fb0dd02944e83`.
- Change vs the 0.989165 base: +7,568 pairs, −573 pairs. Everything in v23 is included.

| Change on top of v23 | Pairs | Evidence |
|---|---|---|
| French aliases: a single-word made-up brand name at exactly one S1's full address (`ag_alias_add.parquet`) | +2,125 | Same rule on train: 98.3% true (US), 98.6% (India). France leaves 29% of these targets unclaimed, against ~1% in the US. Worst case ~90% |
| Singleton guard: US/India S1s whose only predicted pair has corrected p < 0.6 are left empty (`guard06_remove_v23.parquet`) | −267 | Fold 4: US +0.00004, India +0.00012 (z 2.8) |

**Tested tonight and rejected:**

| Idea | Result on fold 4 or train |
|---|---|
| Cross-encoders (MiniLM, xlm-roberta-base), fixed and learned blends | Negative or noise |
| French blank typo and suffix copies | Train precision 1–42% |
| US/India non-tie blank records | ≥90% precision only at ~150 targets |
| `missed` / owner-renormalization power | Noise |

**Expected:** about 0.9895. If it scores below the best earlier upload, the fallback is v23, or v22c if v23 was never uploaded.
