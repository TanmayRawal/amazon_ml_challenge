# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Antidex  
**Team Members:** Tanmay Rawal, Shiviansh Yadav, Mudit Todi, Yash Jain  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

**The task.** For every Source-1 (S1) business record, we had to find all the records in Source 2 and Source 3 that describe the same real business. That can be none, one or many. Each S1 is scored with F0.5, which weighs precision twice as much as recall, and the leaderboard averages that score over all S1 records. Two things made this harder than a textbook matching task:
- the test set introduces a third country, France, that has no training labels at all;
- the test files contain far more look-alike "decoy" records than the training files.

**Our approach.** We built two complete entity-resolution pipelines independently and then combined them:

- **Pipeline A** finds candidates through 11 blocking channels and a "second hop" through confident matches. A LightGBM ranker keeps the best 60 candidates per S1, and two stacked LightGBM matchers score each pair. A final link stage re-scores the uncertain pairs with three fine-tuned multilingual cross-encoders.
- **Pipeline B** finds candidates through 8 blocking passes, including character-trigram TF-IDF search. A ranker keeps the best 30, and a calibrated matcher is followed by a stacked matcher that reads the whole candidate list. Two fine-tuned MiniLM "pair judges" add a transformer's view as features.

We average the two pipelines' probabilities, then apply a small learned "second opinion" model that corrects the average using the context of each S1's candidate list. Each final list is then chosen to maximise the **expected F0.5 exactly**, so an empty list is chosen whenever that is the better bet.

For France, where no labels exist, we combined the models with a set of carefully checked rules. They target patterns that our US/India-trained models systematically missed or over-accepted.

**Results:**

| Measure | Score |
|---|---|
| Held-out macro F0.5 on a sealed fold (US + India, test-like conditions) | **0.9918** |
| Public leaderboard, final submission (v24) | **0.989462** |

Over the three days, the public score rose step by step from 0.9812 to 0.989462. Every step was validated before we spent an upload on it.

**Compliance.** Everything was trained only on the provided training data. We used no external lookups, APIs, geocoders or gazetteers. Our models are LightGBM (MIT) and transformers of at most 278M parameters under Apache-2.0 or MIT licences, far below the 8B limit.

---

## 2. Methodology

### 2.1 Problem Analysis

Before building any model, we spent the first hours auditing all seven data files. We read them with quoting disabled, because some names contain `"` characters, and asserted row counts and ID formats. The audit shaped every later decision, so we summarise what we learned.

**The ground truth has a clean structure:**
- **Single owner.** Each of the 7,638,365 true pairs links a target to exactly one S1. No target ever belongs to two businesses, and no match crosses countries.
- **Match counts.** About 5.6% of S1s have no match at all (singletons). The rest average 3.46 matches, with a maximum of 11. No S1 ever has more than 5 copies from Source 2 or 6 from Source 3.
- **Duplicates.** Target records that are exact duplicates of each other always belong to the same S1, in all 43,910 such groups.

**True copies and decoys look different.** The most useful insight came from comparing true copies of a business with records that match nothing. The data generator makes two distinct kinds of change:

| True copies carry *noise* | Decoys make *one meaningful change* |
|---|---|
| typos and leet (`L0TUS`), dropped or reordered words, added accents | the legal form is swapped (Pvt ↔ Public, LLC ↔ Inc, SARL ↔ SAS) |
| a legal form added, dropped or moved into brackets | one house-number digit is replaced (315 → 325); in France the number is shifted upwards by a fixed offset |
| website names (`wilfordhancock.com`) and acronyms (`PMC`) | the same number on a different street |
| blank addresses, and house numbers truncated or zero-padded (9914 → 914, 05131) | one name word replaced by another common word ("Judo Club" → "Judo Maison") |
| names written in Indian scripts (nine scripts, 13–23% of India's true pairs) | |
| invented trade names ("Fluxonyx") at the owner's address | |

Put simply: true copies are messy but faithful, while decoys are clean but wrong in one specific way. Our features and rules are built around telling these two apart.

**The test set has more decoys than training.**
- Test has about 5.8 target records per S1, against 4.7 in training. So roughly 41–48% of test records belong to no S1, against 26% in training. Six independent indicators agree on this estimate.
- A model trained naively on the training files would therefore be too trusting on test.
- One useful exception: blank-address and website-name records appear on test at exactly the rate of *owned* records in training. On test, these are therefore almost always genuine copies.

**France is different again.**
- French names come from a small vocabulary; "Bordeaux Club SARL" appears 205 times.
- House numbers are short, and 40% of French S1s share a street address with another S1.
- S1 addresses end with the region, while targets often end with the department.
- French website names *delete* accented letters instead of converting them (École becomes `cole`).
- Acronym names are about 20 times more common than in the US.

**Where the remaining loss sits.** Once our models were strong, the largest remaining error came from blank-address copies whose name is shared by several S1s. We call these "ties": there are about 71,000 such records in test, costing about 0.005 of F0.5 per country. Under F0.5, adding a pair only helps if it is correct at least about 75–80% of the time. When two S1s have identical names and the copy has no address, nothing in the data tells them apart, and guessing costs more than it gains.

### 2.2 Solution Strategy

**Approach type:** a hybrid system. Blocking and learned ranking produce candidates. Stacked gradient-boosted matchers and fine-tuned cross-encoders score them, in two independent pipelines. A learned ensemble correction and an exact expected-F0.5 decision produce the final lists.

These ideas made the biggest difference:

1. **Training that looks like test.** We hid a share of the training S1s, so their copies became ownerless decoys and the training data matched the test's decoy density. Every model is cross-fitted: models trained on folds {0, 1} score folds {2, 3}, and the reverse. Every stacked feature is therefore out of fold, and a fifth fold stayed sealed for honest final estimates.
2. **Reading the whole candidate list, not just the pair.** A decoy is often only recognisable against its siblings. For example, a copy whose house number disagrees with every other candidate of the same S1 is suspicious. Our stacked stage sees each S1's full list: consensus on house number, name and legal form, competition from other S1s for the same record, and how common the changed words are.
3. **Two different pipelines, then a correction.**
   - The two pipelines use different blocking and different features, so they make different mistakes. Averaging them gained about +0.0013 on held-out data over the better single pipeline.
   - A small second-opinion model then corrects the average in list context, gaining another +0.0009.
4. **Deciding each list exactly.** Instead of a fixed threshold, we compute for each S1 the subset of candidates, including the empty set, with the highest *expected* F0.5 (Ye et al., ICML 2012). Singletons are handled naturally: the empty list wins whenever it is the better bet.
5. **Validating France without labels.** We could not measure French accuracy directly, so we used four indirect checks:
   - **Class volumes:** for each type of edit, we compared how many pairs we accept per S1 on test with how many are truly correct per S1 in training. The generator is the same, so an excess signals false matches.
   - **The blank-address test:** blank records are almost always owned, so a true-copy pattern appears among them and a decoy pattern does not.
   - **Agreement with confident decisions:** a rule must agree with the model's own confident French decisions.
   - **Leaderboard probes:** a few confirmed the direction of each French change.

---

## 3. Candidate Generation (Blocking)

Blocking decides the ceiling on recall: a true match that is never proposed can never be found. We therefore invested heavily here and let each pipeline search differently.

All blocking runs within a country, and country is treated as an open set of labels, so France is processed like any other country. Both pipelines first normalise the records:
- accents folded, leet corrected, junk prefixes and bracketed legal forms removed;
- the legal form kept as a separate field;
- street abbreviations mapped (Rd/Road, R/Rue, Av/Avenue, …) and house numbers parsed;
- Indian-script names transliterated and reduced to consonant "skeletons", so that a Hindi or Tamil spelling can meet its English original.

**Pipeline A (V21).** Pipeline A builds candidates from 11 channels, each added to recover a specific kind of missed match:
- exact normalised name, rare name words, and pairs of name words;
- house number plus street words, pairs of address words, and rare address words;
- typo keys, to catch misspelled words;
- 4-grams of website labels, and concatenated-name prefixes, for merged names and domains;
- Indic consonant skeletons;
- house numbers with one digit dropped.

A LightGBM ranker, trained on blocking-graph and similarity features, keeps the **best 60** per S1. A "second hop" then adds records that share an exact address with one of the S1's confident matches, provided the names agree. In the final version this name check also compares transliterated names, which recovered 19.5% of the India matches the previous version missed.

**Pipeline B (v18).** Pipeline B uses 8 passes:
- exact name, rare name tokens, and house number with an address prefix;
- rare address tokens, and concatenated-name prefixes;
- Indic skeletons;
- rare name word × address word;
- a **character-trigram TF-IDF search** (top 50 by cosine similarity, computed on the GPU), which catches heavily misspelled names that no exact key can find.

A cross-fitted ranker keeps the **best 30** per S1. On the sealed fold, that captures 98.9% of true pairs, and a perfect matcher on those candidates would score 0.9965. Going beyond 40 candidates adds less than 0.0002 recall.

**The final candidate set.** `candidate_pairs.tsv` is the union of both pipelines' candidates, about 122.6 million test pairs, which is exactly the set our models score. The few pairs added by our final rules are appended to their S1's candidate list, so every predicted match is a candidate.

**Why two pipelines helped recall.** The two candidate sets complement each other: 38% of the true pairs that pipeline A never retrieved are among pipeline B's top 30. The pairs neither pipeline finds are mostly invented names on truncated addresses, and those look identical to the decoys at the same address.

---

## 4. Matching Model

### 4.1 Features

We describe each candidate pair from four angles:

- **Name:** token overlap, edit distances, Jaro-Winkler and fuzzy ratios, similarity of the concatenated names, acronym and website coverage. Also how the legal form changed (equal, added, dropped, swapped), how rare the name and its shared words are in that country, cross-script skeleton similarity, and which words changed with how common the new words are. That last feature tells a typo, which creates a rare word, from a decoy, which swaps in a common one.
- **Address:** token overlap weighted by rarity, street-name similarity, and how the house numbers relate (equal, near, truncated, substituted, suffix conflict, signed offset), plus flags for blank addresses and website names.
- **List and competition context** (stacked stage): the pair's rank and score gap within the S1's list, the strongest other S1 claiming the same record, how complete the list is per source, and whether the pair agrees with the S1's other confident candidates on house number, name and legal form.
- **Transformer scores:** in pipeline A, three cross-encoders fine-tuned on our training pairs (two paraphrase-multilingual-MiniLM-L12-v2, Apache-2.0, 118M, and one multilingual-e5-base, MIT, 278M) re-score the uncertain pairs through a small LightGBM "link" model. In pipeline B, two MiniLM pair judges, trained with different seeds on alternating data halves, enter the stacked stage as features.

### 4.2 Model architecture

| Stage | Pipeline A | Pipeline B |
|---|---|---|
| Candidate ranker | LightGBM, best 60 + second hop | LightGBM, best 30 |
| Stage 1 | LightGBM matcher, isotonic calibration | LightGBM matcher, isotonic calibration |
| Stage 2 | stacked LightGBM with list context and decoy features | stacked LightGBM with list context, decoy and vocabulary features, and pair-judge scores |
| Transformers | 3 cross-encoders + LightGBM link over uncertain pairs | 2 pair judges used as features |

**Combining the pipelines.** For US and India we average the two pipelines' probabilities in log-odds space (weight 0.5 each). A pair that only one pipeline proposed keeps that pipeline's probability. The result is stable: any weight between 0.3 and 0.7 gives the same score within ±0.00005.

**The second opinion.** A small LightGBM (400 rounds, 31 leaves) then corrects each averaged probability, starting from its log-odds. It was trained out of fold on the test-like training pool. It looks at:
- both pipelines' log-odds, and which pipelines proposed the pair;
- the S1's list context;
- the three transformer scores;
- 23 structural signals such as number and street conflicts.

Its main job is to reconcile pairs that only one pipeline saw with pairs both saw. It added +0.0009 (India) and +0.0007 (US) on the sealed fold.

### 4.3 Decision (threshold selection)

We never use a fixed threshold. For each S1:

1. We start from the combined probability.
2. We multiply the odds by 3 for blank-address and website targets. On test these are almost always genuine copies (Section 2.1); on held-out data this gave +0.0002.
3. We apply **single-owner renormalisation.** Each record can belong to only one S1, so its probabilities across competing S1s are rescaled to be consistent.
4. We apply the second-opinion correction.
5. We halve the odds of same-street pairs whose house number differs. Test contains 83–85% more digit-substitution look-alikes per S1 than training, while truncations are not over-represented.
6. We choose the subset with the **highest exact expected F0.5**, including the empty set.

### 4.4 France

With no labels, we treated France as a reasoning problem. Which of the generator's operations create decoys, and which create copies? Where do our US/India-trained models misjudge French patterns?

**Pairs we remove (decoy operations):**
- a one-word swap between two common French name words;
- a legal-form swap on an identical name;
- the same house number on a clearly different street;
- an identical name at the upward decoy offsets (+1…+5, +7, +9, +11, and +13 and +21 when the legal form was also added).

**Pairs we keep or boost (copy noise):** names with an appended list word such as "& Fils", "Associés", "Services" or "Frères", when the house number matches and the legal forms agree. The same pattern is 99.8% correct in the US training data.

**Pairs we add**, where the models were blind to a French-specific pattern:

| Rule | What it recognises | Pairs | Evidence |
|---|---|---|---|
| Accent-deleting domains | a website name built from the S1's words and initials with accents deleted (`mrignacclub` = Mérignac Club), at a matching address, with exactly one S1 qualifying | +1,413 | 99.99% correct on the same rule in training; 833 of these pairs had never been proposed by either pipeline |
| Acronyms | a 2–5 letter name spelling the initials of the only S1 at the same house number and street | +2,153 (+72 in US/India) | 99.8–99.9% correct in training; agrees with 99.87% of the model's own confident French acronym matches; France had matched only 78% of its acronyms, against 94–96% elsewhere |
| Ties by legal form | a blank-address record shared by same-name S1s, given to the S1 whose legal form best explains it (posterior ≥ 0.9) | +1,087 | 90–97% correct in training |
| Ties by inherited formatting | copies keep their owner's word order (98.7%), connecting words such as "de/du/des" (97%), "&" (99%) and parentheses (98%). We learned this from 17,000 French blank copies whose owner is certain, and use it to pick the owner among same-name S1s | +1,752 | agrees with 100% of the model's own confident tie decisions |
| Invented brand names | a single-word made-up name at the full address of exactly one S1 | +2,125 | 98.3–98.6% correct in training; France had left 29% of these unmatched, against about 1% in the US |
| Guarded ensemble | pipeline B's opinion added to France, but only in classes where it cannot re-introduce decoys | +1,330 / −173 | never removes list-word, acronym or domain copies |

**Final corrections (all countries):**

| Correction | Pairs | Reason |
|---|---|---|
| US digit substitutions added by the second opinion, removed | −1,602 | the only class where our test acceptance exceeded the true rate (0.0646 vs 0.0612 per S1) |
| US/India domain rule and exact-address pairs no pipeline had proposed | +136 | 94–100% correct in training |
| At most 5 Source-2 and 6 Source-3 copies per S1 | −34 | the generator never exceeds these limits |
| Records claimed by two S1s, resolved | −10 | every record has one owner |
| **Singleton guard:** an S1 whose only predicted pair has probability < 0.6 is left empty | −267 | improved the held-out score in both countries |

---

## 5. Results & Error Analysis

**Held-out scores.** Macro F0.5 on US + India, over the 1.43 million training S1s scored by both pipelines. "Test-like" means we removed the ownerless blank records that our own S1-hiding created.

| System | Sealed fold | Development folds | Sealed fold, test-like |
|---|---|---|---|
| Pipeline A alone | 0.9890 | 0.9886 | – |
| Pipeline B alone | 0.9884 | 0.9884 | – |
| Two-pipeline average | 0.9903 | 0.9902 | 0.9909 |
| + retrieval fix + second opinion | – | – | **0.9918** |

**How the public score grew.** Each row is an upload, with the change that produced it.

| Version | Main change | Public LB |
|---|---|---|
| v5 | stacked stage 2, Indic skeletons | 0.926 |
| v7 | decoy-aware list features | 0.950 |
| v10 | ranker similarity features | 0.961 |
| v13 | name-word × address-word blocking, decoy features in stage 1 | 0.9812 |
| v18c | TF-IDF search, pair judges, test-like training | 0.98216 |
| v20 | pipeline A with three cross-encoders | 0.985328 |
| v21e | two-pipeline average + first France rules | 0.987776 |
| v21e3bx_fr2 | second opinion, US correction, domain and legal-tie rules | 0.989165 |
| v22c | acronyms, formatting-based tie resolution, extra decoy offsets | 0.989359 |
| **v24 (final)** | guarded France ensemble, brand names, domain rule, singleton guard | **0.989462** |

**What we tried that did not work.** These results were just as informative; each was measured on held-out data or checked before any upload.

| Experiment | What we found |
|---|---|
| More candidates (60 → 100) | Recall stops growing after 40 candidates; at most +0.00005 possible |
| Dense embedding retrieval (multilingual e5) | Recovered few extra matches |
| A learned combiner instead of a simple average | +0.00036, then superseded by the second opinion |
| New cross-encoders (MiniLM, xlm-roberta-base) blended into the final probabilities | Lowered the held-out score at every weight: the existing five transformers already capture that signal |
| Moving France's decision threshold | Looser: −0.002 on the leaderboard; tighter: no change. The threshold was already right |
| Averaging the pipelines in France without safeguards | Would have dropped about 6,500 true list-word copies; replaced by the guarded version |
| Breaking US/India ties (57-feature owner model, copy counts, ID and file order) | At most 65% correct, below the ~77% needed; IDs and file order are random |
| Adding blank copies with typos or appended words | 1–42% correct in training |

**Typical errors that remain.**
- **Wrong merges:**
  - decoys that keep the name and street but change one house-number digit;
  - common French names at the same number on another street;
  - one-word vocabulary swaps.
- **Missed matches (about 80% of the remaining loss):**
  - blank-address copies of names shared by several S1s, about 0.005 per country on the sealed fold and more in France;
  - copies no pipeline proposed, about 1% of true pairs, mostly invented names on truncated addresses;
  - copies where both the name and the address changed.

**Checks on the final file.**
- Empty lists: 5.8% in each country, close to the 5.6% of S1s that truly have no match.
- Matches per S1: 3.33–3.39, against 3.46 in training.
- No cross-country pairs, no duplicates, and no S1 over the per-source copy limit.
- We scanned all 5.84 million predicted pairs. The only pairs without obvious name or address overlap are correct ones: names written in another script, and hyphenated website-style names.

---

## 6. Conclusion

- **What mattered most.** Our biggest gains came from understanding the data rather than from bigger models. We made training look like test, so the models learned to distrust decoys. We let the models read each S1's whole candidate list. And we chose each list by exact expected F0.5.
- **Two pipelines beat one.** Building two pipelines independently and combining them added +0.0024 on the leaderboard. A learned correction, a targeted US fix and the France rules added a further +0.0017 (0.987776 → 0.989462).
- **France without labels.** For France we worked like analysts. We identified which of the generator's operations create decoys and which create copies. We checked each rule on the same pattern in the labelled countries and required it to agree with the model's own confident decisions. Acronyms, accent-deleting website names, invented brand names and inherited formatting were the patterns our models had missed.
- **What is left.** The remaining loss is dominated by blank-address records shared by S1s with identical names. No field in the data identifies their owner, and under F0.5 guessing costs more than it earns.
- **Next steps, with more time:** a dense retrieval channel for never-proposed matches, a larger cross-encoder stacked into the correction model, a full-data refit, and training on synthetic French-style noise.

---

## Appendix

### A. Code Artefacts

Everything is in `code/business_entity_resolution/`. Every source file and function has a docstring describing its purpose, inputs and outputs.

| Folder / file | Contents |
|---|---|
| `src/*.py`, `tests/`, `reproduce.sh`, `scripts_pipeline_a/` | Pipeline A (V21); stage-by-stage notes in `docs/pipeline_a.md` |
| `src/pipeline_b_v18/` | Pipeline B (v18), with its own source, tests, README, requirements and run script |
| `src/ensemble/` | two-pipeline average, second-opinion model and its trained file, US correction |
| `src/france/` | French decision rules, domain rule and legal-form tie rule |
| `src/final_rules/` | final rule overlays, the scripts that validate them, the exact change lists, and one README per version explaining the evidence |
| `run_all.sh`, `README.md` | one-command reproduction from raw data to `output/`, plus the layout and step-by-step run order |

The official validator (`utils/validate_submission.py --check-ids`) passes on `output/matching_results.tsv` and `output/candidate_pairs.tsv`, with no warnings.

### B. Environment and licences

- **Libraries:** polars, pyarrow, LightGBM, rapidfuzz, scikit-learn and indic_transliteration, pinned in `requirements.txt`. Transformers run under PyTorch + CUDA, pinned in `requirements_transformer.txt`.
- **Models:**
  - LightGBM (MIT);
  - paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0, 118M);
  - multilingual-e5-base (MIT, 278M);
  - xlm-roberta-base (MIT, 278M), tested but not used in the final system.
- **Hardware:** consumer laptops with a single GPU (6–12 GB) and 16–32 GB RAM. No cloud resources were used.
- **Data use:** only the provided training and test files. No external data, lookups or augmentation.
