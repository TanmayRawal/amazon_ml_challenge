# v22c: final version and all changes since your v21e3b

**File to upload:** `submission_v22c/matching_results.tsv`
- SHA-256 starts `4fc5c50bcd9a3600`.
- 1,732,544 rows.
- The official validator (`--check-ids`) passes with `submission_v22c/candidate_pairs.tsv`, with no warnings.

**Expected LB:** about 0.98932. The base file scored 0.989165.

| Version | What it adds | LB |
|---|---|---|
| your v21e3b | V21 + v18 ensemble, second opinion, France variant B | (not uploaded alone) |
| v21e3bx_fr2 | v21e3b − 1,602 US pairs + 2,500 French pairs | **0.989165** (uploaded) |
| **v22c** | v21e3bx_fr2 + 3,706 pairs − 89 pairs | ≈ 0.98932, not yet uploaded |

**No new model.** v22 is rule-based post-processing on the 0.989165 file. Every rule was checked against train labels (US/India) or with label-free consistency checks (France). US/India models, probabilities and the second opinion are unchanged from your v21e3b.

## 1. What changed from v21e3b to v21e3bx_fr2 (LB 0.989165)

Details and scripts are in `scripts/v21e3bx_fr2_build/`.

- **−1,602 US pairs:** exact name, same street, house number digit-substituted, added by the second opinion. This was the only class where test acceptance per S1 exceeded train truth (0.0646 vs 0.0612).
- **+1,413 French pairs, rule D:** domain names that delete accented letters (École → `cole`), same address, unique S1. Train analog: 99.99% true.
- **+1,087 French pairs, rule A:** blank-address ties resolved by legal-form posterior ≥0.9, with p ≥0.3.

## 2. What v22c adds on top

All change lists are in `changes/`:
- `added.csv` / `removed.csv`: entity IDs, with the rule that produced each pair.
- the same pairs as `.parquet` in test-index form (s1, t, file order).

| # | Change | Pairs | Evidence | Est. LB |
|---|---|---|---|---|
| 1 | **Acronym targets** (`acro_add`) | +2,225 (France 2,153, India 57, US 15) | US/India train: 99.75% / 99.92% true. France: the rule matches the model's own choice on 99.87% of 15,849 claimed French acronym targets. 762 of the pairs were never scored by either pipeline | +0.0001 |
| 2 | **French ties by formatting** (`fr_tie_add`) | +899 | Owner posterior from inherited formatting. Agrees with 100% of claimed ties at posterior >0.75 | +0.00003 |
| 3 | **French ties by stop words** (`fr_tie2_add`) | +582 | Stop-word sequence de/du/des/la/le/les/en/au/aux: owner agreement 0.969 vs same-name null 0.808. Agrees with 3,154/3,154 claimed ties | +0.00002 |
| 4 | **French decoys** (`fr_fp_remove`) | −89 | See below | +0.00001 |

### Acronym targets
France has 25,392 targets named with 2–5 letters ("PC", "TC", "PMC"). France claims 78% of them, against 94–96% in US/India, and on train acronyms appear almost only on true copies. One cause is that "PC" and "SA" are stripped as US legal forms, so the name becomes empty.

The rule gives an acronym target to the unique same-country S1 that meets all of these:
- its initials spell the acronym, tried four ways: all words, without stop words, without legal forms, and without both;
- it has the same house number as the target;
- the street matches: every target street word equals, or is a typo of, an S1 street word, and at least one specific street word is shared.

### French ties by formatting
On 17,000 French blank copies whose owner is certain (only one S1 has that name), copies inherit the owner's formatting:
- word order 98.7%;
- "&" 99.1%;
- parentheses 98.2%;
- accents 87.4%;
- plus a legal-form transition table.

A naive-Bayes posterior over the same-name claimants picks the owner. A pair is added when the posterior is ≥0.9 and the model p is ≥0.2.

### French decoys
- Exact names at house-number offsets +13 and +21 with the legal form added. These have the same signature as the decoy offsets +1…+11 that are already removed.
- Typo'd names at +1 and +2.
- Estimated 65–90% false.

## 3. Rebuild v22c from the 0.989165 file (pure Python, no other inputs)

```bash
python apply_changes.py base_v21e3bx_fr2/matching_results.tsv changes/added.csv changes/removed.csv v22c_rebuilt.tsv
```

It prints `removed 89, added 3706`. The result has the same ID set on every row as `submission_v22c/matching_results.tsv`, which I checked on all 1,732,544 rows.

The rule scripts that produced the change lists are in `scripts/`:

| Folder | Contents |
|---|---|
| `acro_rule.py`, `acro_rule2.py` | acronym rule and train validation |
| `fr_tie_fmt.py`, `fr_tie_nb.py` | formatting calibration and naive Bayes |
| `fr_tie2/` | stop-word attribute |
| `frfp/` | offset analysis |
| `build_combo.py` | merges everything and builds the candidate file |

Their input paths point to a work directory (`work/`) holding the parsed test parquet files, `w20/final_pairs.parquet` and `w20/test/scores_c.parquet` (pipeline A's stack probabilities). They are included as documentation. `../finalize.py` is what you need to rebuild the file.

## 4. Tested today and rejected (don't redo)

| Idea | Result |
|---|---|
| Ranker K 60 → 100 | Recall is flat after K = 40; oracle gain ≤ +0.00005 |
| US/India blank ties | 53% of tie groups have byte-identical names. Nothing gets owner picks above 65% precision, and a pick needs ~75–80%. Tested: 57-feature owner model; copy-count prior (the generator caps copies at 5 in S2 and 6 in S3); per-entity noise profile; S1 address attributes; entity-ID and file-row structure (all random) |
| US/India unretrieved pairs | Best rule adds 49 pairs; unretrieved targets have made-up names that look like decoys |
| US/India fine-class false positives | Excess ≤ 900 pairs per class, which is noise |
| French vocabulary swaps | 99% of rejected swaps are vocabulary swaps (Club ↔ Amicale), i.e. decoys |
| French abbreviations | Already 96–99% accepted |
| French number typos | Legal form kept only 40% of the time, so mixed with decoys |
| Blank copies with an appended suffix word | 11–37% precision on train |
| Your France "tight" probe | Not used. None of the cells it cuts is over-accepted: same-address alias names are 0.045 per S1 against train truth of 0.056 (US) and 0.089 (India); acronyms are confirmed at 99.87%; suffix-copy q ≈ 0.8, above the 0.77 break-even for removal |
| Your country feature in the correction | Noise, as you found |

**What limits the score:** about 71,000 blank-address records (US 23.5k, India 31.5k, France 16k) are shared by same-name S1s, and no field in the data separates them. Resolving all of them correctly would be worth about +0.0025.

## 5. Upload plan

- Upload v22c.
- If it scores above 0.989165, it is final: put `submission_v22c/*.tsv` into the final package `output/`.
- If it scores lower and an upload remains, re-upload `base_v21e3bx_fr2/matching_results.tsv`.
