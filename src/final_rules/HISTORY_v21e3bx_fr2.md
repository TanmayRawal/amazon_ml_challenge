# Step v21e3bx_fr2 (LB 0.989165): changes made on top of v21e3b

Public LB 0.989165, up from 0.987776 for v21e. This was an intermediate step; the final submission is v24 (LB 0.989462), see `README_v24.md`.

| | |
|---|---|
| File | `submissions/v21e3bx_fr2/matching_results.tsv` (SHA-256 starts `3526231d1b9dfa80`) |
| Candidates | `submissions/v21e3x_fr2/candidate_pairs.tsv` |
| In the final package | both files are copied to `final_package/final/output/` |

## 1. What the final file is

**final = your v21e3b − 1,602 US pairs + 2,500 French pairs**

Your v21e3b is the V21 + v18 ensemble, the second-opinion correction, and France variant B. I made three changes on top of it:

| # | Change | Country | Pairs | Rows changed | Expected LB |
|---|---|---|---|---|---|
| 1 | Remove the second opinion's digit-substituted US pairs | US | −1,602 | 1,500 | +0.00003 to +0.0001 |
| 2 | Rule D: accent-deleting domain names | France | +1,413 | 2,438 French rows (changes 2 and 3 together) | +0.00007 |
| 3 | Rule A: tied blank-address targets resolved by legal form | France | +1,087 | (see change 2) | +0.00003 |

Pairs: v21e3b 5,834,380 → final 5,835,278.

We only uploaded the combined file, so the leaderboard can't split the +0.0014 over v21e between your changes and mine. On held-out estimates, most of it is your second opinion (+0.0007) and France variant B (+0.00025).

**Everything else is unchanged**, byte for byte:
- all India rows;
- US rows other than the 1,500;
- French rows other than the 2,438;
- the S1 order.

## 2. Change 1: US digit-substitution removal

**The problem.** I compared, for each edit class (name relation × house-number relation), accepted pairs per S1 on test with true pairs per S1 on train. The test generator is the same for US/India, so any excess over train truth must be false positives. One class was above truth, and the second opinion made it worse:

| US, exact name, same street, house number digit-substituted (315 → 325) | pairs per S1 |
|---|---|
| train truth | 0.0612 |
| v21e | 0.0627 |
| v21e3 (after the second opinion) | **0.0646** |

**Why these are decoys:**
- Test holds 83–85% more digit-substitution candidates per S1 than train. Truncations (9914 → 914) are not enriched.
- The decoy generator changes a house-number digit. The copy generator truncates.

**The fix.** I removed exactly the 1,602 pairs of that class that the second opinion added, and nothing else.
- Every other class of v21e3 is at or below train truth, so no other removal is justified.
- Files: `us_removed.csv` (the pairs); `build_e3x.py` and `e3_us_numsub_added.parquet` (how they were found).

## 3. Change 2: France rule D, accent-deleting domains (+1,413 pairs)

**What we found.** The French website-name generator deletes accented letters instead of transliterating them:
- École → `cole`
- Mérignac Club → `mrignacclub`
- Fédération des Cercle → `fdrationdescercle`

It also mixes initials with words: `nclub` = Nantes Club, `epclub` = Exupéry Pacte Club. Our name features don't see these as matches. 833 of these pairs had never been scored by either pipeline, and many others were scored far too low.

**The rule.** Add the pair when all of these hold:
- The domain is exactly a concatenation of the S1's name words and initials, in any order.
- At least one piece is a full word of three or more letters.
- The target has the same house number, or no number.
- Every leftover street word fuzzy-matches (ratio ≥ 70) an S1 street word. This blocks the "same number, other street" decoy.
- Exactly one S1 qualifies.

**Evidence:**
- On US/India train the same rule is 99.99% true overall, and 53 of 54 are true where the model scored the target low or left it unclaimed.
- Unclaimed domain targets with a unique claimant: France 1,413, against 17–37 in US and 105–118 in India. So this is a France-only gap.
- After the rule, France's domain claim share rises from 0.939 to 0.964, still below the US's 0.971.
- Example: "Generation Societe (France) SARL, 24 Avenue des Gemmeurs" ↔ `generationsocietefrance`, "Av Des Gemmeusr".

## 4. Change 3: France rule A, tied blank targets by legal form (+1,087 pairs)

**The problem.** 44% of French blank-address targets have a name shared by two or more French S1s, against 26% in the US. France leaves 30% of even the clearest of these ties unclaimed, against 1–3% in US/India. Their p sits at 0.5–0.77, so the France policy rejects them even though the model is fairly confident.

**The rule.** Give the blank target to the S1 whose legal form best explains the target's, using P(target form | S1 form) learned from single-S1 blank groups. For example, an S1 without a legal form beats a same-name S1 with a different one, because legal swaps are decoys. All of these must hold:
- the posterior is ≥ 0.9;
- the model p is ≥ 0.3;
- word order does not contradict the choice;
- no other S1 claims the target.

**Evidence:** on train, the rule is 97% true overall and 90–93% where the model had left the target unclaimed. Break-even under F0.5 is about 75%.

## 5. Candidate file

The official validator checks that every predicted pair is a candidate. 646 of the 2,500 French pairs were not in v21e3's candidate file, so `cand_fr2.py` appends them to the right rows. The validator (`--check-ids`) passes with no warnings.

## 6. Tested today and rejected

| Idea | Result |
|---|---|
| Ranker K 60 → 100 | Recall flattens after K = 40; the oracle gain at K = 100 is ≤ +0.00005. About 60% of V21's unretrieved misses are found by neither pipeline at any K. Not worth a rerun. |
| US/India blank-tie tie-breaker (15 signals, 57-feature owner model) | 53% of tie groups have claimants with identical raw names. The owner model picks the true owner 37% of the time, against 35% for p. Best rule +0.00003. Not shipped. |
| Legal form or raw formatting alone as a France tie-breaker | 28–50% true where the model abstained; below break-even |
| Row order / entity-id numbers as a tie signal | no signal, and no leak |
| My learned combiner (instead of the fixed 0.5/0.5 average) | +0.00036 on fold 4, superseded by your second opinion |
| France from the two-pipeline ensemble | would drop 7.1k list-word copies and add decoys; rejected |

## 7. Rebuild

```bash
python build_e3bx_fr2.py
```

It reads v21e3b, v21e3 and v21e3x, plus `../v21e3x_fr2/changes.csv`, and writes this folder. The France rule scripts are in `src/france/fr2_rules/` (README inside). The full run order is in the top-level `README.md`.
