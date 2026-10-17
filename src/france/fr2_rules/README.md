# France addition rules D and A (FRANCE-2)

These scripts add 2,500 French pairs to the final file. The added pairs are listed in `../../ensemble/fr2_added.csv`, together with the rule that added each one.

## Rule D: domain-name targets (1,413 pairs)

The French website-name generator **deletes** accented letters instead of transliterating them (École → `cole`, Médical → `mdical`). It also builds domains from initials plus words (`nclub` = Nantes Club).

A pair is added when all of these hold:
- The domain is exactly a concatenation of the S1's name words and initials, in any order. At least one piece is a full word of three or more letters.
- The target has the same house number as the S1, or no number.
- Every unmatched street word of the target fuzzy-matches (ratio ≥ 70) an unmatched S1 street word.
- Exactly one S1 qualifies.

On US/India train, the same rule is 99.99% true.

## Rule A: tied blank-address targets (1,087 pairs)

A blank-address target whose name is shared by several S1s goes to the S1 whose legal form best explains the target's legal form. The legal-form probabilities are learned from single-S1 blank groups per country.

A pair is added when all of these hold:
- The posterior is ≥ 0.9.
- The model p is ≥ 0.3.
- Word order does not contradict the choice.
- No other S1 claims the target.

On train the rule is 90–97% true. Break-even under F0.5 is about 75%.

## Order

1. **Rule A:** `p1.py` → `p1b.py` → `p1c.py` → `p1f.py` → `p1g.py` (writes `p1g_france.parquet`).
2. **Rule D:** `dom.py` → `dom2.py` → `dom3.py` (writes `dom3_fr_add.parquet`).
3. **Build:** `build.py` writes the rule pairs over v21e3x.

Paths are set at the top of each script:
- `W`: pipeline B's work directory;
- `OUT`: the output folder.

`../../ensemble/build_e3bx_fr2.py` merges these pairs into the final file.
