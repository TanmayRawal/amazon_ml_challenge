# ADR 0010: Final rules shipped as explicit, auditable change lists

**Status:** Accepted

## Context

The final submission is the ensemble output plus about a dozen validated rule overlays ([ADR 0009](0009-label-free-validation-for-france.md)), built over the last day from many analysis scripts. Several of those scripts depend on intermediate files from the research process. Re-running all of them in a reproduction would be fragile, and a reviewer would struggle to see what each rule actually changed.

## Decision

- Each rule's net effect is stored as an **explicit list of (Source-1 ID, target ID) pairs**:
  - `src/ensemble/us_removed.csv` and `fr2_added.csv`;
  - `src/final_rules/changes/added.csv` and `removed.csv`, plus one parquet file per rule.
- `src/final_rules/finalize.py` applies them to the ensemble output in the original build order: removals, then additions. An addition is skipped if another record already owns the target, so the single-owner property is preserved. Every added pair is appended to the candidate file.
- The scripts that derived each list, and their validation, ship alongside as evidence in `src/final_rules/scripts/` and `src/france/fr2_rules/`. Each version's README (`README_v22.md` to `README_v24.md`) gives the numbers.

## Alternatives considered

- **Re-deriving every rule in the reproduction run.** More "pure", but it would pull dozens of research scripts and intermediate files into the critical path.
- **Folding rules into model code as flags.** Hides what changed, and makes per-rule rollback hard.

## Consequences

- Reproduction is deterministic. Starting from the ensemble output, `finalize.py` produces the same set of matched IDs in every one of the 1,732,544 rows as the submitted file, and the official validator passes. Only the order of IDs within a row may differ, which does not affect the score.
- Every rule can be audited, counted or reverted independently.
- The change lists are tied to this test set. On new data the rules must be re-derived with the evidence scripts.
