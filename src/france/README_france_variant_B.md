# v21e_fr_audit1234: v21e_fr_audit134 plus rule 2 (floor on appended list-word copies)

This is the full v21e file, with only French rows changed. US/India rows are byte-identical to `v21e_bundle/submission_v21e/matching_results.tsv`. The row count is 1,732,545 and the S1 order is the same. The official validator passes with `--check-ids`.

**Rules.** Rules 1, 3 and 4 are the same as in `v21e_fr_audit134` (p set to 0). Rule 2 then floors p at 0.9 before the v20 French decision is re-run. It applies to two groups:
- **List-word copies (33,172 pairs).** One type word is dropped and a list word is appended. The list is fils, associes, services, service, developpement, france, freres, and Cie/Groupe. All of these must hold:
  - the list word comes after the legal form, or at the end after "&"/"et" when there is no legal form;
  - the house number is equal;
  - the legal forms agree once Cie/Groupe are ignored;
  - the street is not in explicit disagreement.
- **Appended "Développement" (1,608 pairs).** Same address and legal form, and the only extra word is "Développement", placed after the legal form or at the end.

Together this is 34,780 eligible pairs. 6,060 of them were not in v21e.

**Result vs v21e France.**
- 2,566 pairs removed: rules 1/3/4 as in audit134, plus 165 re-decision side effects.
- 5,828 pairs added, 5,743 of them rule-2 pairs.
- 7,894 French rows changed.
- 3.292 matches per French S1 (v21e: 3.279).

**Why the list-word copies look true:**
- US train: the same operator is 99.8% true, and US test has no suffix-swap decoys (0.159 per S1 on both train and test).
- The blank-address test puts appended list words at the level of known-true noisy classes.
- The pipeline A audit estimates 0.85–0.99, with a floor rule of its own.
- The v21e France gain is consistent.

The model's p for these pairs depends only on which suffix word is used.

**Expected effect** (rule-2 truth rates 0.95 / 0.85 / 0.75):

| | France F0.5 | LB |
|---|---|---|
| Central | +0.0017 | +0.00025 |
| Range | +0.0007 to +0.0025 | +0.00011 to +0.00038 |

The added copies break even at about 77% true. Rule 2 alone is worth +0.0005 France at the central estimate and −0.0001 in the pessimistic case.
