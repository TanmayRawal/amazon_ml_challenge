# Reproduction guide

The repository regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the provided train and test TSVs.

- Everything runs offline on the provided data: no external lookups, APIs or data.
- Pretrained transformers (Apache-2.0 / MIT, ≤ 278M parameters) are downloaded once and fine-tuned only on the provided training pairs.
- The methodology is described in [`methodology.md`](methodology.md) (full report) and [`summary.md`](summary.md) (1–2 pages).

## Layout

| Path | Contents |
|---|---|
| `src/*.py`, `tests/`, `reproduce.sh`, `scripts_pipeline_a/` | **Pipeline A (V21).** Blocking over 11 channels plus a second hop, LightGBM ranker (top 60), stacked LightGBM matchers, 3 cross-encoders, and a LightGBM link model. `docs/pipeline_a.md` documents every stage. |
| `src/pipeline_b_v18/` | **Pipeline B (v18).** 8 blocking passes including character-trigram TF-IDF, LightGBM ranker (top 30), stage-1 and stacked stage-2 LightGBM, and 2 MiniLM pair judges. It has its own `src/`, `tests/`, `README.md` and `requirements.txt`, and PowerShell run chains (`v18_chain.ps1`). |
| `src/france/` | France decision rules: `build_variant.py` (vocabulary swaps, suffix copies) and `d01_audit.py` (legal-swap, street and decoy-offset rules, list-word floor). |
| `src/france/fr2_rules/` | France accent-deleting domain rule and legal-form tie rule. |
| `src/ensemble/` | `ensemble_v21e.py`: V21 ⊕ v18 logit average, second-opinion correction and decision. Also here: `ens_second.py` with its model `ens_second_with_p18_p20.txt`; `ens_exp_v21.py`, which trains and evaluates the correction; and the US digit-substitution correction (`build_e3x.py`, `build_e3bx_fr2.py`, `us_removed.csv`, `fr2_added.csv`). |
| `src/final_rules/` | The v22–v24 rule overlays. `scripts/` holds each rule's code and its train or label-free validation; `changes/` holds the exact pair lists (test index and entity IDs); `README_v22.md`, `README_v23.md` and `README_v24.md` give the evidence for every rule. |
| `requirements.txt` | Pinned environment for pipeline A, the ensemble and the rules (Python 3.12). |
| `requirements_transformer.txt` | Pinned environment for the cross-encoders (PyTorch + transformers, CUDA). |

## Reproduction in one command

```bash
BER_DATA_DIR=<student_resource/dataset> PY=<python with requirements.txt> PY_TORCH=<python with requirements_transformer.txt> \
VALIDATOR=<student_resource/utils/validate_submission.py> bash run_all.sh
```

`run_all.sh` runs every stage below in order and writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
- No path needs editing. Work folders default to `./work_a` and `./work_b`, the output to `./output`, and all can be overridden with `WORK_A`, `WORK_B` and `OUT_DIR`.
- Each finished stage is marked, so rerunning the same command resumes after an interruption.
- Expect roughly 15–20 hours on one 8–12 GB GPU with 32 GB RAM; the transformer stages dominate.

Two inputs ship with the code:
- **`src/france/france_variant_B_rows.tsv`**: the French rows produced by pipeline A's France audit tools (variant B: decoy removals, suffix floors, list-word floor), which the ensemble takes as its France input.
- **The final change lists** in `src/ensemble/` and `src/final_rules/changes/`, applied by `src/final_rules/finalize.py`.

Starting from the ensemble output, `finalize.py` reproduces the submitted matches exactly. We checked every row: all 1,732,544 hold the same set of matched IDs as the submitted `matching_results.tsv`, and the official validator passes. In some rows the IDs may be listed in a different order, which does not affect the score: F0.5 is computed from the set of IDs in each row.

The analysis scripts in `src/final_rules/scripts/` and `src/france/fr2_rules/` show how each change list was derived and validated. They are evidence for the rules, not part of the one-command run.

## Reproduction, step by step

Set `DATA=<student_resource/dataset>` and give each pipeline its own empty work directory, `<work_A>` and `<work_B>`.

1. **Pipeline A:**
   ```bash
   PY=<python> PY_TORCH=<python with requirements_transformer.txt> BER_DATA_DIR=$DATA BER_WORK_DIR=<work_A> VERSION=v21 bash reproduce.sh
   ```
   `reproduce.sh` runs data preparation, folds, test-like density matching, ranker, matchers, cross-encoders and the link model. The version-specific additions are in `scripts_pipeline_a/run_v14.sh … run_v21.sh`.

   Output: `<work_A>/test/{scores_c,xenc_band,xenc_scores,xenc_scores2,xenc_scores3,s1,tgt}.parquet` and `models/`.

2. **Pipeline B**, from `src/pipeline_b_v18/`, all commands as `python src/run.py …`:
   - `prepare --split all`
   - `translit --split all`
   - `folds`
   - `indic-dict`
   - `fuzzy --split train`, then `fuzzy --split test`
   - `rank-train`
   - `rank --split train`, then `rank --split test`
   - `features --split train --variant c`, then `features --split test --variant c`
   - `train-b`
   - `score-b --split test`
   - `pj-train`
   - `pj-score --split all`
   - `pj-train --pj-tag 2`
   - `pj-score --split all --pj-tag 2`
   - `stack-features --split train --pj-sets ",2" --drop-noisy-hidden`
   - `train-c --pj-sets ",2" --drop-noisy-hidden`
   - `stack-features --split test --pj-sets ",2"`

   `v18_chain.ps1` runs the same sequence. Output: `<work_B>/test/scores_c.parquet`.

3. **France rows**, in `src/france/`:
   - `build_variant.py` (novocab + suffix);
   - `d01_audit.py` (variant B: rules 1, 3 and 4, plus the list-word floor).

4. **Ensemble and second opinion**, in `src/ensemble/`. The output is v21e3b:
   ```bash
   python ensemble_v21e.py --code ../ --work <work_A> --v18-scores <work_B>/test/scores_c.parquet \
          --data-dir $DATA --france <France variant B file> --second ens_second_with_p18_p20.txt --version v21e3b
   ```

5. **US correction and France rules D/A:** `src/ensemble/build_e3bx_fr2.py`, which gives v21e3bx_fr2 (LB 0.989165). It does two things:
   - removes the 1,602 US digit-substitution pairs in `us_removed.csv`;
   - adds the 2,500 French pairs from `src/france/fr2_rules/`.

   `cand_fr2.py` appends the new pairs to the candidate file.

6. **Final rule overlays → v24:** `src/final_rules/scripts/build_combo.py`, run with v21e3bx_fr2 as the base.
   - **Additions**, in order:
     1. acronyms (`acro_add`);
     2. French formatting ties (`fr_tie_add`);
     3. French stop-word ties (`fr_tie2_add`);
     4. US/India domain rule (`q_dom_usin_add_tight`);
     5. recovered US/India pairs (`q_r1_add`);
     6. French ties at posterior 0.85–0.9 (`q_fr_tie3_add`);
     7. guarded France ensemble (`ens_add_w0.7_cap`);
     8. French aliases (`ag_alias_add`).
   - **Removals:**
     1. French decoys (`fr_fp_remove`);
     2. ensemble removals (`ens_rem_w0.7`);
     3. double claims (`q_dup_remove`);
     4. over-cap pairs (`q_cap_remove`);
     5. singleton guard (`guard06_remove_v23`).
   - An addition is skipped if its target is already claimed by another S1.
   - The script also appends every added pair to the candidate file.
   - `changes/added.csv` and `changes/removed.csv` list the net effect by entity ID.

7. **Validate:**
   ```bash
   python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir $DATA/test --check-ids
   ```
   The result is PASS with no warnings.

**Paths:** the analysis and rule scripts in `src/final_rules/scripts/`, `src/france/` and `src/ensemble/` set their work and data directories in constants at the top of each file (`W`, `ROOT`, `OUT`, `SUB`). Point these at your own directories before running.

## Final output (not stored in this repository)

| File | Details |
|---|---|
| `output/matching_results.tsv` | v24; SHA-256 starts `519fb0dd02944e83`. 1,732,544 rows in test order; 5,842,273 pairs; no cross-country pairs; no S1 over the generator caps (5 S2 / 6 S3) |
| `output/candidate_pairs.tsv` | The union of both pipelines' candidates plus rule additions; every match is a candidate |
