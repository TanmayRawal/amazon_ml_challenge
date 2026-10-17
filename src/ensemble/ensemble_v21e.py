"""v21e: ensemble of the pipeline A (v20/V21) link model with our v18 stack for US/India, plus a chosen France file.

  US/India  p = logit-average (0.5/0.5) of
              - v20/V21 linked probability (stack p re-scored by the 3-transformer link; stack p outside the band)
              - our v18 stage-2 probability (test/scores_c.parquet of our pipeline)
            (a pair scored by only one model keeps that model's probability), then v20's own test-time corrections:
            decoy_shift x0.5 (same street, different house number) and bw_boost x3 (blank-address / website
            targets), then v20's decision (dta_own, missed 0.031).
  France    the rows of a France variant file (default: v20_fr_novocab_suffix), unchanged.

Validated on 1.43M train S1s scored by both models out of fold (scripts/ens_eval.py): fold 4 0.99027 vs v20 alone
0.98900, dev folds 0.99018 vs 0.98859 (US +0.0007, India +0.0022).

Usage (Python with pipeline A's requirements; run from anywhere):
  python ensemble_v21e.py --code <pipeline A code>/business_entity_resolution/src --work <pipeline A work dir, e.g. work_v21>
         --v18-scores <bundle>/v18_model/test_scores_c_v18.parquet --data-dir <student_resource/dataset>
         --france <France variant>/matching_results.tsv --version v21e [--compare <existing matching_results.tsv>]
  --work must hold test/{scores_c,xenc_band,xenc_scores,xenc_scores2,xenc_scores3,s1,tgt}.parquet,
  models/xlink/{model.txt,meta.json}, models/matcher_c/meta.json and train/s1.parquet (only its country column is
  read; any file with US and India rows works).
Writes <work>/submissions/<version>/ (matching_results.tsv, candidate_pairs.tsv = union of both models' candidates,
manifest.json; the official validator runs), unless --compare is given (then it only compares the predictions).
"""
import argparse
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--code", required=True)
ap.add_argument("--work", required=True)
ap.add_argument("--v18-scores", required=True)
ap.add_argument("--data-dir", required=True)
ap.add_argument("--france", required=True)
ap.add_argument("--version", default="v21e")
ap.add_argument("--weight", type=float, default=0.5, help="weight of the v20/V21 logit")
ap.add_argument("--compare", default=None)
ap.add_argument("--second", default=None, help="second-opinion model file (ens_second); default off")
args = ap.parse_args()

os.environ.update(BER_DECOY_SHIFT="1", BER_BW_BOOST="1", BER_TEXT_RULES="0", BER_SECOND="0")   # shifts are applied after averaging
sys.path.insert(0, args.code)
import polars as pl  # noqa: E402
import xlink  # noqa: E402
from config import from_args  # noqa: E402
from matcher_b import apply_policy  # noqa: E402

POLICY = {"name": "dta_own", "missed": 0.031}
cfg = from_args(args.data_dir, args.work)
log = lambda m: print(m, flush=True)  # noqa: E731
t0 = time.time()

# 1. v20/V21 linked probabilities, before the test-time shifts (capture the pool link_predict hands to the policy)
cap = {}
orig = xlink.apply_policy
xlink.apply_policy = lambda pool, policy, *a, **k: cap.setdefault("pool", pool)
xlink.link_predict(cfg, "test", log)
xlink.apply_policy = orig
s1 = pl.read_parquet(os.path.join(args.work, "test", "s1.parquet"), columns=["s1", "entity_id", "country"])
tg = pl.read_parquet(os.path.join(args.work, "test", "tgt.parquet"), columns=["t", "entity_id"])
usin = s1.filter(pl.col("country").is_in(["US", "India"])).select("s1")
p20 = cap["pool"].join(usin, on="s1", how="semi").select("s1", "t", p20="p")

# 2. our v18 stage-2 probabilities (same test S1/target indexes: both pipelines number records in file order)
p18 = (pl.scan_parquet(args.v18_scores).select("s1", "t", p18="p").filter(pl.col("p18") > 1e-4).collect()
       .join(usin, on="s1", how="semi"))

# 3. logit average; a pair scored by one model only keeps its probability
lg = lambda c: (pl.col(c).clip(1e-6, 1 - 1e-6) / (1 - pl.col(c).clip(1e-6, 1 - 1e-6))).log()  # noqa: E731
z = args.weight * lg("p20") + (1 - args.weight) * lg("p18")
pool = (p20.join(p18, on=["s1", "t"], how="full", coalesce=True)
        .select("s1", "t", p=pl.when(pl.col("p20").is_null()).then(pl.col("p18"))
                .when(pl.col("p18").is_null()).then(pl.col("p20")).otherwise(1 / (1 + (-z).exp())))
        .filter(pl.col("p") > 1e-4))
log(f"US/India: v20 {p20.height:,} pairs, v18 {p18.height:,}, union {pool.height:,} ({time.time() - t0:.0f}s)")

# 4. v20's test-time corrections, then its decision
if args.second:   # second opinion (experiments/decision2/ens_second.py): bw -> owner -> correction -> shift
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # ens_second.py sits next to this script
    import ens_second as es
    pool = xlink.bw_boost(cfg, "test", pool, 3.0, log)
    feat = es.features(args.work, "test", pool, p20, p18)
    feat.write_parquet(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"test_feat_{args.version}.parquet"))
    log(f"second opinion on {feat.height:,} test pairs ({args.second})")
    pred_usin = es.decide(pool, feat, es.load(args.second), shift=lambda s: xlink.decoy_shift(cfg, "test", s, 0.5, log))
else:
    pool = xlink.decoy_shift(cfg, "test", pool, 0.5, log)
    pool = xlink.bw_boost(cfg, "test", pool, 3.0, log)
    pred_usin = apply_policy(pool, POLICY)

# 5. France rows from the chosen variant file
fr = (pl.read_csv(args.france, separator="\t", quote_char=None, schema_overrides={"matched_entity_ids": pl.String})
      .with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
      .filter(pl.col("matched_entity_ids") != "")
      .join(s1.filter(pl.col("country") == "France").select("s1", source1_entity_id="entity_id"), on="source1_entity_id")
      .join(tg.select("t", matched_entity_ids="entity_id"), on="matched_entity_ids").select("s1", "t"))
pred = pl.concat([pred_usin, fr])
log(f"predicted: US/India {pred_usin.height:,} + France {fr.height:,} = {pred.height:,}")

if args.compare:
    ref = (pl.read_csv(args.compare, separator="\t", quote_char=None, schema_overrides={"matched_entity_ids": pl.String})
           .with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
           .filter(pl.col("matched_entity_ids") != "").select(s="source1_entity_id", m="matched_entity_ids"))
    mine = pred.join(s1, on="s1").join(tg, on="t", suffix="_t").select(s="entity_id", m="entity_id_t")
    log(f"compare with {args.compare}: reference {ref.height:,}, mine {mine.height:,}, "
        f"only mine {mine.join(ref, on=['s', 'm'], how='anti').height:,}, only reference {ref.join(mine, on=['s', 'm'], how='anti').height:,}")
    sys.exit(0)

from make_submission import make_submission  # noqa: E402
cand = pl.concat([pl.scan_parquet(os.path.join(args.work, "test", "scores_c.parquet")).select("s1", "t").collect(),
                  pl.scan_parquet(args.v18_scores).select("s1", "t").collect().join(usin, on="s1", how="semi")]).unique()
model = {"policy": {**POLICY, "usin": f"logit avg w{args.weight} v20/V21 link + v18 stack, decoy x0.5, bw x3",
                    "france": args.france},
         "sealed": {"overall": 0.990269, "note": "shared-S1 fold-4 ensemble estimate (US/India), scripts/ens_eval.py"}}
make_submission(cfg, args.version, pred, cand, model, log)
