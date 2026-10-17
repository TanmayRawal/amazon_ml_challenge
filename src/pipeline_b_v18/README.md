# Business Entity Resolution — pipeline

Links every test Source‑1 business to its matching Source‑2/3 records (zero, one or
many), scored by macro F0.5 per Source‑1 entity. Everything runs offline on the
provided data only: no external lookups and no external data.

## Environment

- Python 3.12.13; exact package versions in `requirements.txt` (the pipeline itself
  needs polars, pyarrow, numpy, scikit-learn, rapidfuzz, lightgbm and indic-transliteration).
- Tested on Windows 11, 16 GB RAM, 24 logical CPUs. No GPU is needed for the current stages.

```bash
uv venv --python 3.12 venv
uv pip install --python venv/Scripts/python.exe -r requirements.txt
```

## Paths

- `BER_DATA_DIR`: the `dataset/` folder containing `train/` and `test/`. Its parent must
  contain `utils/validate_submission.py`, as in `student_resource/`.
- `BER_WORK_DIR`: a scratch folder for parquet caches, models and submissions (about 15 GB).
  Keep it outside cloud-synced folders.

Both can also be given as `--data-dir` / `--work-dir` flags.

## Reproduce end to end

Run from this folder:

```bash
python src/run.py prepare --split all      # parse TSVs (quoting disabled), normalize, cache
python src/run.py folds                    # 5 entity-disjoint stratified folds; fold 4 sealed
python src/run.py block --split train      # Plan A candidates + recall report
python src/run.py features --split train
python src/run.py train-a                  # fit on folds 0-3, tune the decision rule, report fold 4
python src/run.py block --split test
python src/run.py features --split test
python src/run.py submit-a --version v1    # writes and validates the two output files
```

`submit-a` writes `matching_results.tsv`, `candidate_pairs.tsv` and `manifest.json` to
`$BER_WORK_DIR/submissions/<version>/`. It then re-reads both files and checks:

- one row per test S1, in file order;
- exact headers;
- every ID exists in the test S2/S3 files and has an `S2-`/`S3-` prefix;
- no duplicate IDs within a list;
- matches ⊆ candidates.

Finally it runs `utils/validate_submission.py --check-ids`. Any failure raises.

Runtime on the machine above: prepare about 3 min; block, features and train about 10 min each.

## Pipeline (Plan A baseline)

| Stage | Module | What it does |
|---|---|---|
| Load | `data_io.py` | Tab-separated, quote processing off, row counts checked against raw newlines |
| Normalize | `normalize.py` | Accent folding (Latin only; Indic signs kept), junk and template removal, pipe websites → domain field, leet fixes, legal forms kept as a separate field, PO boxes, placeholders and zero padding removed from addresses |
| Folds | `splits.py` | S1 entity groups (single-owner targets), stratified by country × match count × Indic target |
| Blocking | `blocking.py` | Per country: exact name, rare name token, (house number, street prefix), concatenated-name prefix; union ranked and cut to 50 per S1 |
| Features | `features.py` | 28 pair features: token/set overlaps, rapidfuzz ratios, name rarity, pass flags, per-S1 rank |
| Matcher | `model_a.py` | Logistic regression (linear baseline); threshold, target exclusivity and relative cut tuned for macro F0.5 on dev folds |
| Metric | `score.py` | Exact macro F0.5 including singletons; candidate recall ceiling, oracle F0.5, reduction ratio |
| Output | `make_submission.py` | Writing, integrity checks, official validator, manifest |

## Stage 2: learned candidate ranker (variant `b`)

```bash
python src/run.py rank-train                     # sample union pairs from folds 0-3, fit 2 cross-fitted rankers
python src/run.py rank --split train             # ranked candidates + recall@K report (dev vs sealed)
python src/run.py features --split train --variant b
python src/run.py train-a --variant b
python src/run.py rank --split test
python src/run.py features --split test --variant b
python src/run.py submit-a --variant b --version v2
```

`ranker.py` (supervised meta-blocking):

1. **Union.** Every pass is kept whole, within block-size caps: exact name (≤ 1000 targets per block), rare name word (DF ≤ 1000), (house number, address-word prefix) for 3 numbers × 10 words, rare address word (DF ≤ 300, a new pass), and name/website prefix.
2. **Scoring.** A LightGBM model scores each pair using only blocking-graph features: which passes fired, summed and maximum IDF of shared keys, block sizes, per-S1 ranks, union size, and token counts per record.
3. **Cut.** The top 60 per S1 are stored; the matcher uses the first `rk_cand_k` (40 up to v7, 50 from v8).
4. **Cross-fitting.** Model A is trained on folds {0,1} and scores folds {2,3}; model B the reverse. The sealed fold and test get the mean of both.

Tests: `python -m pytest -q tests`.

## Submitted versions

Validation macro F0.5 is on sealed fold 4: train entities never used for fitting or tuning.

| Version | Commands | Sealed-fold macro F0.5 |
|---|---|---|
| v1 | Plan A: `prepare`, `folds`, `block`, `features`, `train-a`, `submit-a --version v1` | 0.8203 |
| v2 | Stage 2: v1 steps up to `folds`, then `rank-train`, `rank --split train/test`, `features --variant b` (train/test), `train-a --variant b`, `submit-a --variant b --version v2` | 0.8648 |
| v3 | Stage 3: v2 ranking steps, then `features --variant c` (train/test), `train-b`, `submit-b --version v3` | 0.9285 |
| v4 | v3 steps, then `score-b --split test`, `stack-features --split train`, `stack-features --split test`, `train-c`, `submit-c --version v4` | 0.9348 |
| v5 | Stage 5: `translit --split all`, then the v4 steps from `rank-train` onwards (fresh `score-b` cache), `submit-c --version v5` | 0.9431 |
| v6 | Stage 6: v5 up to `score-b --split test`, then `stack-features --split train/test`, `train-c`, `submit-c --version v6` | 0.9666 |
| v7 | v6 steps, then `stack-features --level 3` (train/test), `train-c --level 3`, `submit-c --level 3 --version v7`; decision shift 0.6 | 0.9667 (0.9671 without the shift) |
| v8 | Ranker similarity features, `rk_cand_k` 50: `rank-train` → `rank` (train/test) → `features --variant c` (train/test) → `train-b` → `score-b --split test` → `stack-features` (train/test) → `train-c` → `submit-c --version v8` (level 2, shift 0.6) | 0.9724 (0.9731 without the shift) |
| v9 | v8 plus stage-2 name-vocabulary features: `stack-features` (train/test) → `train-c` → `submit-c --version v9` | 0.9759 (0.9763 without the shift) |
| v10 | v9 plus stage-2 per-source list features (`stack.source_features`): same steps, `submit-c --version v10` | 0.9761 (0.9766 without the shift) |

## Stage 3 and v4 (matchers B and C)

- **`matcher_b.py`** (v3) trains two cross-fitted LightGBM models on 40 pair features (feature variant `c`; 44 from v5) and calibrates them with isotonic regression on out-of-fold scores.
- **`decide.py`** chooses each S1's predicted set by exact expected F0.5 (Ye et al., ICML 2012), including the empty-list option, after renormalizing each target's probability across the S1s that claim it (every ground-truth target has one owner).
- **`stack.py`** (v4) is a second, cross-fitted stage. It learns from stage-1 probabilities plus features that need other pairs' scores: competition for the same target, the S1's candidate-list context, and similarity to the S1's most confident other candidate.
- **`stress.py`** (`python src/run.py stress`) runs decoy injection for v3. It hides 20% and 30% of train S1s so their records become ownerless decoys, as in test. Held-out macro F0.5 moved from 0.9282 to 0.9267 and 0.9255.

## Stage 5 and v5: names written in Indian scripts

In India, 13–23% of true target names are written in an Indian script (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada or Malayalam) while the S1 name is in Latin letters. In v3, India S1s with such targets accounted for 28% of the total loss.

```bash
python src/run.py translit --split all     # name skeletons -> s1_sk.parquet, tgt_sk.parquet
```

Then rerun the v4 steps from `rank-train` onwards (with `score-b --split test` on a fresh cache), ending with `submit-c --version v5`.

- **Skeletons** (`normalize.name_skeleton`):
  - Indian-script runs are transliterated to Latin with `indic_transliteration` (IAST).
  - Each word is then reduced to a consonant skeleton. The first letter stays; later vowels and the aspiration `h` are dropped.
  - Letters that transliteration confuses are merged (g/c/q→k, d→t, b/v/w/f→p, z/j→s), repeats collapse, and legal words are dropped.
  - "Ram Marketing" and "राम मार्केटिंग" both become `rm mrktnk`.
- **Blocking** (`ranker.py`): a skeleton pass for Indian-script and mixed-script targets. Its keys are the exact full skeleton (blocks of up to 1,000 targets) and unordered pairs of skeleton words (up to 500). The ranker gets five skeleton features.
- **Matcher features** (`features.py`): skeleton-pass hits and IDF, plus token-set ratio and Jaro-Winkler similarity of the skeletons.
- **Effect on the sealed fold:** at 40 candidates per S1, candidate recall for Indian-script targets rose from 0.830 to 0.871 (India overall from 0.942 to 0.948).

## Stage 6 and v6: decoy detection in stage 2

An error analysis of v5 on the dev folds found that 95% of false matches point at records that belong to no S1. These decoys follow a recipe: they copy an S1's name, swap its main legal form, and shift its house or unit number by a small amount. Examples of the swap: Private ↔ Public Limited, Private Limited → LLP, LLC ↔ Inc/Corp/Ltd. For pairs with p > 0.2, these swaps were true matches 0% of the time. Genuine copies differ: their noise keeps the number, and the S1's true records agree with each other.

`stack.decoy_features` adds 14 stage-2 features:

- **Legal-form relation:** swapped, added, extra, lost or dropped forms ("group" and "co" ignored).
- **House-number relation:** numbers found on only one side, and the log numeric offset between them.
- **Consensus within the S1's candidate list:** p1-weighted support for the target's first number, core name and legal form from the S1's other candidates; whether its number is the list's p1-weighted mode; the share of list mass on the S1's own number.

Only stage 2 is retrained. On the sealed fold, false-match pairs fell from 29,271 to 7,333 and singleton accuracy rose from 0.876 to 0.961.

## v7: stage 3 and a decoy-robust decision

- **Stage 3** (`--level 3`) reruns the stacked model on stage-2 scores. List and consensus features are then weighted by probabilities in which decoys already score low. The gain is small: sealed 0.9671 without the shift, against 0.9666 for v6.
- **Decision shift** (`config.decision_shift = 0.6`, `decide.prior_shift`) multiplies the final odds before the owner normalization and the expected-F0.5 selection. Test is inferred to hold ~41–48% ownerless records, against 26% in train.
  - `python src/run.py stress --variant c --pool dev` hides 0–40% of dev-fold S1s, so their records become ownerless, and rebuilds both stages for the remaining S1s.
  - Averaged over the 20% and 30% levels (≈41–48% decoys), r = 0.6 scores 0.9582 against 0.9570 for r = 1. At 0% it costs 0.0007.
  - The sealed fold gives the same picture (`--pool sealed`).

## v8: similarity-aware ranker

In v7, 2.1% of dev true pairs were in the blocking union but ranked outside the top 40. For example, "A-400610 Cnsultancy Limited" at an identical address ranked 277th of 277. The ranker only saw which keys fired, not how similar the two records are.

`ranker._sim_features` adds four features to every union pair:
- token Jaccard of core names;
- token Jaccard of address words;
- address containment;
- same first house number.

Tokens are pre-hashed to integer sets per record and compared in 2M-pair batches, which adds ~8 s per 12M-pair chunk. The matcher now takes 50 candidates per S1.

On the sealed fold:
- ranker validation log loss fell from 0.0103/0.0111 to 0.0072/0.0062;
- recall@40 rose from 0.9546 to 0.9689 (Indic targets 0.871 → 0.909);
- the oracle on the candidates rose from 0.9828 to 0.9891;
- final macro F0.5 rose from 0.9667 to 0.9724.

## v9: which name words changed

Genuine noise and decoys both change one word of a name, but they draw on different words:
- **Genuine noise** mostly adds or substitutes a few generic words ("center", "services", "partners") or garbles a word into a one-off token ("advisors" → "advieors").
- **Decoys** swap in another business-name word ("holdings", "exports", "ventures"; "Lab" → "Fab").

`stack.vocab_features` adds five stage-2 features:
- how many target words are missing from the S1 name, and how many S1 words are missing from the target;
- log document frequency (over the split's target names, per country) of the rarest and of the commonest new word, and of the rarest missing word.

On the sealed fold, stage-2 log loss fell from 0.0077 to 0.0064 and false-match pairs from 5,454 to 3,460. Macro F0.5 rose from 0.9724 to 0.9759.
