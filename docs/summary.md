# Business Entity Resolution: Methodology Summary

**Team:** Antidex (Tanmay Rawal, Shiviansh Yadav, Mudit Todi, Yash Jain)  
**Final submission:** v24, public leaderboard **0.989462**

## Our approach

**The task.** For each Source-1 business record, find every Source-2 and Source-3 record that describes the same business, possibly none. Each record is scored with F0.5, which rewards precision twice as much as recall. The test set adds France, a country with no training labels.

**What the data taught us.** We began by auditing the data, and one observation shaped everything that followed:
- **True copies** of a business are *messy but faithful*: typos, dropped words, website names, acronyms, blank addresses, Indian-script spellings.
- **Decoys** are *clean but wrong in one specific way*: a swapped legal form, one changed house-number digit, a different street, or one common word replaced.

We also found that test has about twice training's share of decoys, and that every record belongs to at most one business.

**How we built the system:**
1. **Two independent pipelines.**
   - Pipeline A proposes candidates through 11 blocking channels and a second hop, and ranks the best 60 with LightGBM.
   - Pipeline B proposes candidates through 8 passes, including character-trigram TF-IDF search, and ranks the best 30.
   - In each pipeline, a calibrated LightGBM matcher is followed by a stacked LightGBM that reads the S1's whole candidate list. Decoys often stand out only against their siblings.
2. **Transformers where they help.** Pipeline A re-scores uncertain pairs with three fine-tuned multilingual cross-encoders. Pipeline B feeds two fine-tuned MiniLM pair judges into its stacked stage.
3. **Combination and correction.** We average the two pipelines' probabilities, then a small out-of-fold "second opinion" model corrects the average in list context.
4. **Exact decisions.** Each record can belong to only one business, so probabilities are made consistent across competing S1s. Each list is then the subset with the highest *expected* F0.5, where an empty list is a valid answer.
5. **France.** We worked out which generator operations create decoys and which create copies. We then added rules for the patterns our models missed: acronyms, website names with accents deleted, invented brand names, and owner detection from inherited formatting. Each rule was checked on the same pattern in the labelled countries (98–99.9% correct) and against the model's own confident French decisions.

**How we trained.** We hid some training S1s so the decoy density matched test, and cross-fitted every model so that stacked features were always out of fold. We kept one fold sealed for honest estimates, and no change was uploaded before it passed a held-out or label-free check.

## Models

| Model | Licence | Size |
|---|---|---|
| LightGBM (rankers, matchers, link, correction) | MIT | – |
| paraphrase-multilingual-MiniLM-L12-v2 | Apache-2.0 | 118M |
| multilingual-e5-base | MIT | 278M |

All models were fine-tuned only on the provided data. We used no external data or lookups.

## Key experiments

| Change | Effect |
|---|---|
| Decoy-aware list features | +0.024 LB (0.926 → 0.950) |
| Ranker similarity features, name × address blocking | 0.961 → 0.981 |
| Second pipeline and averaging | +0.0024 LB |
| Second-opinion correction | +0.0009 held out |
| France rules (suffix copies, decoy removal, domains, acronyms, ties, brand names) | about +0.0017 LB in total |
| Tried and rejected: more candidates, dense retrieval, new cross-encoder blends, moving the French threshold, tie-breaking models | no gain, or negative, on held-out data |

## Results

- **Held-out macro F0.5** (US + India, test-like sealed fold): **0.9918**.
- **Public leaderboard:** 0.9812 → 0.98216 → 0.985328 → 0.987776 → 0.989165 → 0.989359 → **0.989462** (final).

## Conclusion

Our gains came mainly from understanding the data:
- making training look like test;
- letting the models read each candidate list as a whole;
- deciding each list by exact expected F0.5;
- reasoning carefully about France, which has no labels.

Two independent pipelines and a learned correction added the rest. What remains is mostly blank-address records shared by businesses with identical names, where the data holds no signal about the true owner.
