"""Stage 3 (V13x): transformer link on the pairs the stack is unsure about.

About 3% of candidate pairs have 0.01 < p < 0.99. A multilingual cross-encoder
(sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2: Apache-2.0, 118M parameters, run offline,
fine-tuned only on the provided pairs; see xenc_cli.py) reads "S1 name | address" against
"target name | address" for those pairs, and a small LightGBM combines its logit with the stack's p.
Honest sealed test on V12 scores: +0.0041 F0.5 with the transformer never seeing a scored target.

Leak control: two transformers train on dev-fold pairs whose TARGET hashes to half 0 / 1. Each train
pair is scored by the model that never saw its target, test pairs by the mean of both. The combiner
trains on dev folds only, so the sealed fold stays a clean local score.

Flow: xenc-export (train, test) -> xenc_cli.py train / score (CUDA Python) -> link-train -> submit-x.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from config import Config
from matcher_b import apply_policy, owned
from prepare import load
from score import per_entity
from stack import _dir, _parts, level_scores

BAND = (0.01, 0.99)
LINK_FEATURES = ["p", "logit", "xenc"]
CTX = ["x_rank", "x_gap", "x_t_gap", "n_band"]
# B+ (BER_LINK_EXTRA=1): structured stack features next to the transformer scores, read from <split>/feat_d of
# BER_FEAT_FROM (a work dir, default this one). Sealed A/B: experiments/link_feat.py.
EXTRA = ["num_conflict", "num_match", "num_near", "sfx_conflict", "st_tset", "st_cont", "addr_tset", "addr_wjac",
         "name_tset", "legal_conflict", "t_blank", "s_blank", "rk_p", "log_name_tdf", "t_n_s1", "c_other_max", "c_gap",
         "s_rank", "sib_p", "src_n_conf_same", "src_n_conf_other", "d_raw_sim", "num_view", "tok_view", "addr_eq_same",
         "nw_new_df_min", "leg_swap", "num_off"]
PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
          "feature_fraction": 0.9, "verbosity": -1, "seed": 7, "deterministic": True, "force_row_wise": True}
ROUNDS = 400
_TXT = pl.concat_str(pl.col("business_name").fill_null(""), pl.lit(" | "), pl.col("business_address").fill_null(""))


def band_texts(scores: pl.DataFrame, s1: pl.DataFrame, tg: pl.DataFrame) -> pl.DataFrame:
    """scores: s1, t, p (+label, fold) for every candidate pair; s1/tg: id, business_name,
    business_address (+src for tg). Returns the uncertain pairs with the transformer inputs:
    text_a = S1, text_b = target, text_b_ctx = target + the S1's best confident other candidate
    (p > 0.9, same source first) + the best other S1 claiming this target (p > 0.01)."""
    txt = _TXT
    s_txt, t_txt = s1.select("s1", s_txt=txt), tg.select("t", "src", t_txt=txt)
    sc = scores.filter(pl.col("p") > BAND[0])
    band = sc.filter(pl.col("p") < BAND[1]).join(t_txt, on="t")
    top2 = (sc.select("t", "s1", "p").sort(["t", "p"], descending=[False, True]).group_by("t", maintain_order=True)
            .agg(r1=pl.col("s1").first(), r2=pl.col("s1").slice(1, 1).first()))
    band = (band.join(top2, on="t", how="left")
            .with_columns(rival=pl.when(pl.col("r1") == pl.col("s1")).then(pl.col("r2")).otherwise(pl.col("r1")))
            .join(s_txt.rename({"s1": "rival", "s_txt": "rival_txt"}), on="rival", how="left"))
    conf = sc.filter(pl.col("p") > 0.9).select("s1", t2="t", p2="p").join(
        t_txt.rename({"t": "t2", "src": "src2", "t_txt": "sib_txt"}), on="t2")
    sib = (band.select("s1", "t", "src").join(conf, on="s1").filter(pl.col("t2") != pl.col("t"))
           .with_columns(same=(pl.col("src2") == pl.col("src")).cast(pl.Int8))
           .sort(["s1", "t", "same", "p2"], descending=[False, False, True, True])
           .group_by("s1", "t", maintain_order=True).agg(pl.col("sib_txt").first()))
    keep = [c for c in ("label", "fold") if c in scores.columns]
    return (band.join(sib, on=["s1", "t"], how="left").join(s_txt, on="s1")
            .select("s1", "t", "p", *keep, thalf=(pl.col("t").hash(seed=9) % 2).cast(pl.Int8),
                    text_a="s_txt", text_b="t_txt",
                    text_b_ctx=pl.concat_str(pl.col("t_txt"), pl.lit(" [SIB] "), pl.col("sib_txt").fill_null("none"),
                                             pl.lit(" [RIVAL] "), pl.col("rival_txt").fill_null("none"))))


def export_band(cfg: Config, split: str, log=print) -> None:
    """Write the uncertain-band pairs (0.01 < p < 0.99 of the level-2 stack) with their transformer texts
    to <split>/xenc_band.parquet."""
    t0 = time.time()
    scores = level_scores(cfg, split, 2, log)
    x = band_texts(scores, load(cfg, split, "s1", ["s1", "business_name", "business_address"]),
                   load(cfg, split, "tgt", ["t", "business_name", "business_address", "src"]))
    path = cfg.split_dir(split) / "xenc_band.parquet"
    x.write_parquet(path)
    log(f"[xlink] {split}: {x.height:,} uncertain pairs of {scores.height:,} -> {path} ({time.time() - t0:.0f}s)")


PSEUDO_MAX = 150_000   # pseudo positives and negatives each (French pairs)


def pseudo_pairs(cfg: Config, log=print) -> None:
    """F5: transformer training pairs for test countries absent from train (France), labelled without
    ground truth (self-training on test is allowed). Positive: stack p > 0.99 and the street agrees (same
    first house number, no suffix conflict, street-name similarity >= 90). Negative: an uncertain pair
    (0.01 < p < 0.99) whose street disagrees (other house number, suffix conflict, or same number but
    street-name similarity <= 50); on train such pairs are 0.1-4% true. The same number of dev-fold
    train pairs of the uncertain band is replayed so US/India are not forgotten. Writes
    models/xenc_f5/pairs.parquet for xenc_cli.py train (fold -1 = pseudo)."""
    from rapidfuzz import fuzz, process
    from features import street_fields
    t0 = time.time()
    seen = load(cfg, "train", "s1", ["country"])["country"].unique().implode()
    s1 = load(cfg, "test", "s1", ["s1", "country", "business_name", "business_address", "addr_nums"]).filter(
        ~pl.col("country").is_in(seen))
    sc = level_scores(cfg, "test", 2, log).select("s1", "t", "p").filter(pl.col("p") > BAND[0]).join(
        s1.select("s1"), on="s1", how="semi")
    tg = load(cfg, "test", "tgt", ["t", "business_name", "business_address", "addr_nums"]).join(
        sc.select("t").unique(), on="t", how="semi")
    num = pl.col("addr_nums").str.split(" ").list.first().fill_null("")
    side = lambda df, idc: (df.select(idc, "business_name", "business_address", n=num)  # noqa: E731
                            .join(street_fields(df.select(idc, "business_address"), cfg.n_jobs), on=idc)
                            .with_columns(txt=_TXT).drop("business_name", "business_address"))
    x = sc.join(side(s1, "s1"), on="s1").join(side(tg, "t"), on="t", suffix="_t")
    st = process.cpdist(x["street_core"].to_list(), x["street_core_t"].to_list(), scorer=fuzz.token_set_ratio,
                        workers=-1, dtype=np.float32)
    st[((x["street_core"] == "") | (x["street_core_t"] == "")).to_numpy()] = -1.0
    x = x.with_columns(st=pl.Series(st))
    both_num = (pl.col("n") != "") & (pl.col("n_t") != "")
    sfx_conf = (pl.col("street_sfx") != "") & (pl.col("street_sfx_t") != "") & (pl.col("street_sfx") != pl.col("street_sfx_t"))
    pos = (pl.col("p") > 0.99) & both_num & (pl.col("n") == pl.col("n_t")) & ~sfx_conf & (pl.col("st") >= 90)
    neg = (pl.col("p") < BAND[1]) & ((both_num & (pl.col("n") != pl.col("n_t"))) | sfx_conf
                                    | (both_num & (pl.col("n") == pl.col("n_t")) & (pl.col("st") >= 0) & (pl.col("st") <= 50)))
    x = x.with_columns(label=pl.when(pos).then(1).when(neg).then(0).otherwise(None).cast(pl.UInt8)).drop_nulls("label")
    n_pos, n_neg = int(x["label"].sum()), int((x["label"] == 0).sum())
    fr = pl.concat([x.filter(pl.col("label") == v).sample(min(PSEUDO_MAX, max(1, (x["label"] == v).sum())), seed=cfg.seed)
                    for v in (0, 1) if (x["label"] == v).sum()])
    fr = fr.select("label", fold=pl.lit(-1, pl.Int16), thalf=(pl.col("t").hash(seed=9) % 2).cast(pl.Int8),
                   text_a="txt", text_b="txt_t")
    replay = (pl.read_parquet(cfg.split_dir("train") / "xenc_band.parquet", columns=["label", "fold", "thalf", "text_a", "text_b"])
              .filter(pl.col("fold") != 4))
    replay = replay.sample(min(replay.height, fr.height), seed=cfg.seed).with_columns(pl.col("fold").cast(pl.Int16))
    out = _dir(cfg, 2).parent / "xenc_f5"
    out.mkdir(parents=True, exist_ok=True)
    pl.concat([fr, replay.select(fr.columns)]).write_parquet(out / "pairs.parquet")
    log(f"[xlink F5] pseudo pairs: {n_pos:,} positive, {n_neg:,} negative of {sc.height:,} uncertain+ pairs -> "
        f"kept {fr.height:,} + {replay.height:,} replayed train pairs ({time.time() - t0:.0f}s)")


def fr_band(cfg: Config, log=print) -> None:
    """Test uncertain pairs of countries absent from train -> test/xenc_band_fr.parquet, to be scored by the
    French fine-tuned transformer. US/India keep the base transformer: the French fine-tune costs them
    (v15 sealed link gain +0.0029 with it vs +0.0036 without)."""
    seen = load(cfg, "train", "s1", ["country"])["country"].unique().implode()
    fr = load(cfg, "test", "s1", ["s1", "country"]).filter(~pl.col("country").is_in(seen)).select("s1")
    band = pl.read_parquet(cfg.split_dir("test") / "xenc_band.parquet").join(fr, on="s1", how="semi")
    band.write_parquet(cfg.split_dir("test") / "xenc_band_fr.parquet")
    log(f"[xlink] test: {band.height:,} uncertain pairs of countries absent from train")


def fr_merge(cfg: Config, log=print) -> None:
    """test/xenc_scores.parquet (base transformer, kept as xenc_scores_base.parquet) with the rows of
    xenc_scores_fr.parquet (French fine-tuned transformer) replacing their pairs."""
    d = cfg.split_dir("test")
    base = d / "xenc_scores_base.parquet"
    if not base.exists():
        (d / "xenc_scores.parquet").rename(base)
    fr = pl.read_parquet(d / "xenc_scores_fr.parquet").select("s1", "t", x_fr="xenc")
    out = (pl.read_parquet(base).join(fr, on=["s1", "t"], how="left")
           .with_columns(xenc=pl.coalesce("x_fr", "xenc")).drop("x_fr"))
    out.write_parquet(d / "xenc_scores.parquet")
    log(f"[xlink] test: {fr.height:,} of {out.height:,} transformer scores from the French model")


def _linked(cfg: Config, split: str) -> pl.DataFrame:
    """Band pairs of a split joined with their transformer scores (plus optional extra transformers and stack
    features), the context features and logit(p). Raises if a band pair has no transformer score."""
    band = pl.read_parquet(cfg.split_dir(split) / "xenc_band.parquet").drop("text_a", "text_b", "text_b_ctx")
    xs = pl.read_parquet(cfg.split_dir(split) / "xenc_scores.parquet").select("s1", "t", "xenc")
    x = band.join(xs, on=["s1", "t"], how="left")
    if x["xenc"].null_count():
        raise RuntimeError(f"{split}: {x['xenc'].null_count():,} uncertain pairs have no transformer score")
    for k in (2, 3):      # B: more transformers' opinions (ensemble): xenc_scores2.parquet -> xenc2, ...
        extra = cfg.split_dir(split) / f"xenc_scores{k}.parquet"
        if extra.exists():
            x = x.join(pl.read_parquet(extra).select("s1", "t", pl.col("xenc").alias(f"xenc{k}")), on=["s1", "t"], how="left")
    if os.environ.get("BER_LINK_EXTRA") == "1":
        src = Path(os.environ.get("BER_FEAT_FROM", str(cfg.work_dir))) / split / "feat_d"
        fd = (pl.scan_parquet(str(src / "part_*.parquet")).select(["s1", "t"] + EXTRA)
              .join(x.lazy().select("s1", "t"), on=["s1", "t"], how="semi").collect())
        x = x.join(fd, on=["s1", "t"], how="left")
    return context(x).with_columns(logit=(pl.col("p") / (1 - pl.col("p"))).log())


def context(x: pl.DataFrame) -> pl.DataFrame:
    """C: the pair among its S1's uncertain pairs (transformer rank, gap to the S1's best, number of uncertain
    pairs) and against the best OTHER S1's transformer score on the same target. Sealed A/B on v15:
    +0.0004 alone, +0.0006 with a second transformer (experiments/link_bcd.py)."""
    x = x.with_columns(x_rank=pl.col("xenc").rank("ordinal", descending=True).over("s1").cast(pl.Float32),
                       x_gap=(pl.col("xenc") - pl.col("xenc").max().over("s1")).cast(pl.Float32),
                       n_band=pl.len().over("s1").cast(pl.Float32))
    top = (x.select("t", "s1", "xenc").sort(["t", "xenc"], descending=[False, True]).group_by("t", maintain_order=True)
           .agg(a_s1=pl.col("s1").first(), a=pl.col("xenc").first(), b=pl.col("xenc").slice(1, 1).first()))
    other = pl.when(pl.col("a_s1") == pl.col("s1")).then(pl.col("b")).otherwise(pl.col("a"))
    return (x.join(top, on="t", how="left").with_columns(x_t_gap=(pl.col("xenc") - other).fill_null(10.0).cast(pl.Float32))
            .drop("a_s1", "a", "b"))


def _replace(scores: pl.DataFrame, new: pl.DataFrame) -> pl.DataFrame:
    """Stack scores (s1, t, p) with the link probabilities p_link of new replacing p for its pairs."""
    return (scores.select("s1", "t", "p").join(new.select("s1", "t", p_new="p_link"), on=["s1", "t"], how="left")
            .with_columns(p=pl.coalesce("p_new", "p")).drop("p_new"))


def _fit(d: pl.DataFrame, feats: list) -> lgb.Booster:
    """Train the LightGBM link combiner on d's feature columns feats and its label."""
    return lgb.train(PARAMS, lgb.Dataset(d.select(feats).to_numpy(), d["label"].to_numpy()), num_boost_round=ROUNDS)


def _xfit(x: pl.DataFrame, feats: list, m: lgb.Booster) -> list:
    """Link probabilities of every train band pair: sealed fold by the saved dev model m, each dev fold by a model
    trained on the other dev folds. Returns [sealed, dev folds...] frames with p_link."""
    dev = x.filter(pl.col("fold") != 4)
    sea = x.filter(pl.col("fold") == 4)
    xfit = [sea.with_columns(p_link=pl.Series(m.predict(sea.select(feats).to_numpy())))]
    for f in sorted(dev["fold"].unique().to_list()):
        part = x.filter(pl.col("fold") == f)
        xfit.append(part.with_columns(p_link=pl.Series(_fit(dev.filter(pl.col("fold") != f), feats)
                                                       .predict(part.select(feats).to_numpy()))))
    return xfit


def link_train(cfg: Config, log=print) -> None:
    """Train the link combiner on dev-fold band pairs and report sealed-fold F0.5 (stack vs linked).

    Saves models/xlink/model.txt and meta.json; unless BER_SECOND=0 also trains the second-opinion model."""
    x = _linked(cfg, "train")
    feats = LINK_FEATURES + [f"xenc{k}" for k in (2, 3) if f"xenc{k}" in x.columns] + CTX + [f for f in EXTRA if f in x.columns]
    dev = x.filter(pl.col("fold") != 4)
    m = _fit(dev, feats)
    out = _dir(cfg, 2).parent / "xlink"
    out.mkdir(parents=True, exist_ok=True)
    m.save_model(str(out / "model.txt"))
    policy = json.loads((_dir(cfg, 2) / "meta.json").read_text())["policy"]
    scores = level_scores(cfg, "train", 2, log)
    ids = load(cfg, "train", "folds", ["s1", "fold"]).filter(pl.col("fold") == 4).select("s1")   # audit: incl. no-candidate S1s
    gt = load(cfg, "train", "gt").join(ids, on="s1", how="semi")
    # Audit (V19): at submission every train-country S1 is re-scored, so the rival owners of a sealed target carry
    # linked probabilities too; "linked" leaves dev-fold rivals at stack p. "linked_xfit" also links dev folds,
    # each by a model trained on the other dev folds, so single-owner competition matches test time.
    xfit = _xfit(x, feats, m)
    sealed = xfit[0]
    res = {}
    for name, pool in (("stack", scores.select("s1", "t", "p")), ("linked", _replace(scores, sealed)),
                       ("linked_xfit", _replace(scores, pl.concat(xfit)))):
        pred = apply_policy(pool.filter(pl.col("p") > 1e-4), policy)
        res[name] = float(per_entity(ids, pred.join(ids, on="s1", how="semi"), gt)["f"].mean())
    res["overall"] = res["linked_xfit"]   # the manifest reports sealed["overall"]: the test-like number (audit)
    meta = {"features": feats, "band": BAND, "policy": policy, "sealed": res, "train_rows": dev.height}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"[xlink] sealed F0.5: stack {res['stack']:.5f} -> linked (test-like, cross-fitted rival owners) "
        f"{res['linked_xfit']:.5f} ({res['linked_xfit'] - res['stack']:+.5f}); old protocol {res['linked']:.5f}")
    if os.environ.get("BER_SECOND", "1") == "1":
        meta["second"] = second_train(cfg, _replace(scores, pl.concat(xfit)).filter(pl.col("p") > 1e-4), x, policy, log)
        (out / "meta.json").write_text(json.dumps(meta, indent=2))


def decoy_flags(cfg: Config, split: str, pool: pl.DataFrame) -> pl.DataFrame:
    """s1, t of the pairs decoy_shift moves: p > 0.01, train country, both first house numbers present and different
    (not a reordering), address word Jaccard >= 0.5."""
    seen = load(cfg, "train", "s1", ["country"])["country"].unique().implode()
    s1 = load(cfg, split, "s1", ["s1", "country", "addr_nums", "addr_toks"]).filter(pl.col("country").is_in(seen))
    tg = load(cfg, split, "tgt", ["t", "addr_nums", "addr_toks"])
    first = lambda c: pl.col(c).str.split(" ").list.first()  # noqa: E731
    toks = lambda c: pl.col(c).str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() > 1))  # noqa: E731
    y = (pool.filter(pl.col("p") > 0.01).select("s1", "t").join(s1, on="s1").join(tg, on="t", suffix="_t")
         .with_columns(a=toks("addr_toks"), b=toks("addr_toks_t"), n1=first("addr_nums"), n2=first("addr_nums_t"),
                       m1=pl.col("addr_nums").str.split(" "), m2=pl.col("addr_nums_t").str.split(" ")))
    jac = pl.col("a").list.set_intersection("b").list.len() / pl.col("a").list.set_union("b").list.len().clip(1, None)
    # audit: "Unit 1416, 6515 Belcrest Rd" vs "6515 Belcrest Rd, Unit 1416" is a reordering, not another house number
    reorder = pl.col("m2").list.contains(pl.col("n1")) | pl.col("m1").list.contains(pl.col("n2"))
    return y.filter((pl.col("n1").fill_null("") != "") & (pl.col("n2").fill_null("") != "") & (pl.col("n1") != pl.col("n2"))
                    & ~reorder & (jac >= 0.5)).select("s1", "t")


def decoy_shift(cfg: Config, split: str, pool: pl.DataFrame, r: float, log=print, flag: pl.DataFrame | None = None) -> pl.DataFrame:
    """Test holds 1.2-1.5x more "same street, different house number" pairs per S1 than train (US 0.54 -> 0.83,
    India 0.51 -> 0.63 per S1 among same-name targets; every other address pattern equal or lower), all the extra
    ones decoys: the stratum's true odds fall to ~0.5x of train's in both countries (decoy_types.py). Train's
    density matching hides owners instead, so no local score sees it. Odds x r for those pairs of train
    countries (decoy_shift.py simulates the test excess on the sealed fold with twin decoys). flag: decoy_flags."""
    flag = (decoy_flags(cfg, split, pool) if flag is None else flag).select("s1", "t", shift=pl.lit(True))
    odds = pl.col("p") / (1.0 - pl.col("p")).clip(1e-12, None)
    out = (pool.join(flag, on=["s1", "t"], how="left")
           .with_columns(p=pl.when(pl.col("shift")).then(odds * r / (1.0 + odds * r)).otherwise(pl.col("p"))).drop("shift"))
    log(f"[xlink] {split}: decoy shift x{r} on {flag.height:,} same-street different-number pairs (train countries)")
    return out


def bw_boost(cfg: Config, split: str, pool: pl.DataFrame, k: float, log=print) -> pl.DataFrame:
    """Blank-address / website-name targets per S1 on test equal train's OWNED ones (blank: India 0.138 vs 0.136,
    US 0.167 vs 0.164; website: 0.174 vs 0.171, 0.217 vs 0.210), not train's total (+27%): on test they are nearly
    all true copies, while on train ~22% are copies of S1s hidden by density matching, which the models learned to
    distrust. Odds x k for those pairs of train countries. Test-like sealed fold (orphan blank/website targets
    removed): x3 +0.00028 India, +0.00019 US; above x5 it falls (most misses there are same-name ties;
    experiments/blank_boost.py, blank_ties.py)."""
    seen = load(cfg, "train", "s1", ["country"])["country"].unique().implode()
    ok = load(cfg, split, "s1", ["s1", "country"]).filter(pl.col("country").is_in(seen)).select("s1", _a=pl.lit(True))
    bw = (load(cfg, split, "tgt", ["t", "addr_blank", "is_domain"]).filter(pl.col("addr_blank") | pl.col("is_domain"))
          .select("t", _b=pl.lit(True)))
    odds = pl.col("p") / (1.0 - pl.col("p")).clip(1e-12, None)
    hit = (pl.col("_a") & pl.col("_b")).fill_null(False)
    out = pool.join(ok, on="s1", how="left").join(bw, on="t", how="left").with_columns(_h=hit)
    log(f"[xlink] {split}: blank/website odds x{k} on {int(out['_h'].sum()):,} pairs (train countries)")
    return out.with_columns(p=pl.when("_h").then(odds * k / (1.0 + odds * k)).otherwise(pl.col("p"))).drop("_a", "_b", "_h")


# Text-edit rules (experiments/decoys, stack p on V17, tuned on dev folds): odds multipliers for
# pairs with 0.05 < p < 0.999 whose S1 -> target edit the model does not see. A letter appended at a name word's end
# (DMD -> VMD): 43-69% true at p ~0.93; the only differing number goes UP by 1-9: decoy-prone (US 86% true), DOWN
# by 1-9: 99.5% true; India "pvt ltd" -> "pvt": 99% true at p 0.945.
TEXT_RULES = {"ins_tail": {"India": 0.03, "US": 0.03}, "num_up": {"India": 0.7, "US": 0.5},
              "num_down": {"India": 8.0, "US": 16.0}, "pvt_only": {"India": 4.0}}


def _ins_tail(sc: str, tc: str) -> bool:
    """One name word differs (edit distance <= 2, not an abbreviation) and the edit is one letter inserted at the
    word's end."""
    from rapidfuzz.distance import Levenshtein
    a, b = sc.split(), tc.split()
    new, gone = [w for w in b if w not in a], [w for w in a if w not in b]
    if len(new) != 1 or len(gone) != 1 or not set(a) & set(b):
        return False
    g, w = gone[0], new[0]
    if g[0] == w or (len(w) <= 3 and len(g) > len(w) and g.startswith(w)) or Levenshtein.distance(w, g) > 2:
        return False
    ops = Levenshtein.editops(g, w)
    return len(ops) == 1 and ops[0].tag == "insert" and ops[0].src_pos >= len(g) - 1


def _num_dir(sn: str, tn: str) -> int:
    """+1 / -1: the one number only the S1 has and the one only the target has differ by +1..9 / -9..-1."""
    a, b = (sn or "").split(), (tn or "").split()
    s_only, t_only = [v for v in a if v not in b], [v for v in b if v not in a]
    if len(s_only) != 1 or len(t_only) != 1 or s_only[0].lstrip("0") == t_only[0].lstrip("0"):
        return 0
    try:
        d = int(t_only[0]) - int(s_only[0])
    except ValueError:
        return 0
    return 1 if 1 <= d <= 9 else (-1 if -9 <= d <= -1 else 0)


def text_rules(pool: pl.DataFrame, s1: pl.DataFrame, tg: pl.DataFrame, log=print) -> pl.DataFrame:
    """pool: s1, t, p. s1: s1, country, core, core_key, script, legal, addr_nums; tg: t, core, core_key, script,
    is_domain, legal, addr_nums, addr_blank. Odds x TEXT_RULES for the pairs of countries the rules were tuned on."""
    tg = tg.rename({c: f"{c}_t" for c in tg.columns if c != "t"})
    x = (pool.filter((pl.col("p") > 0.05) & (pl.col("p") < 0.999)).select("s1", "t")
         .join(s1.filter(pl.col("country").is_in(["India", "US"])), on="s1").join(tg, on="t"))
    name_ok = ((pl.col("script") == pl.col("script_t")) & ~pl.col("is_domain_t") & (pl.col("core") != pl.col("core_t"))
               & (pl.col("core_key") != pl.col("core_key_t")))
    cand = x.filter(name_ok)
    tail = [_ins_tail(a or "", b or "") for a, b in zip(cand["core"].to_list(), cand["core_t"].to_list())]
    num = [0 if bl else _num_dir(a, b) for a, b, bl in zip(x["addr_nums"].to_list(), x["addr_nums_t"].to_list(),
                                                            x["addr_blank_t"].fill_null(False).to_list())]
    x = (x.with_columns(num=pl.Series(num, dtype=pl.Int8), pvt=(pl.col("legal") == "ltd pvt") & (pl.col("legal_t") == "pvt"))
         .join(cand.select("s1", "t", tail=pl.Series(tail, dtype=pl.Boolean)), on=["s1", "t"], how="left"))
    r = lambda rule: pl.col("country").replace_strict(TEXT_RULES[rule], default=1.0)  # noqa: E731
    m = (pl.when(pl.col("tail").fill_null(False)).then(r("ins_tail")).otherwise(1.0)
         * pl.when(pl.col("num") == 1).then(r("num_up")).when(pl.col("num") == -1).then(r("num_down")).otherwise(1.0)
         * pl.when(pl.col("pvt").fill_null(False)).then(r("pvt_only")).otherwise(1.0))
    x = x.select("s1", "t", m=m).filter(pl.col("m") != 1.0)
    log(f"[xlink] text rules on {x.height:,} pairs")
    odds = pl.col("p") / (1.0 - pl.col("p")).clip(1e-12, None) * pl.col("m").fill_null(1.0)
    return pool.join(x, on=["s1", "t"], how="left").with_columns(p=odds / (1.0 + odds)).drop("m")


# V21 second opinion (experiments/decision2): a small LightGBM that corrects the probabilities the
# set decision sees (after single owner) with what the link cannot see: the S1's other candidates and the target's rivals
# on the FINAL probabilities, the transformer scores and the stack's number / street / sibling-agreement signals.
# Boosted from logit(p) (init_score), trained on dev folds of the test-like train pool (orphan blank/website targets
# dropped, bw_boost applied, link cross-fitted). Sealed fold 4, V20 data, test-like: India +0.00030, US +0.00030 (dev
# folds, cross-fitted: +0.00026 / +0.00037); with every ownerless target claimed by a sealed S1 twinned (2x test decoy
# density) +0.00024 / +0.00036. Both directions help: lowering alone +0.00018 / +0.00015, raising +0.00010 / +0.00011.
# Production order (link_decide, test-time decoy shift x0.5 after it) on the sealed fold: India +0.00030, US +0.00026.
SECOND_STACK = ["num_conflict", "num_match", "num_near", "num_delta", "num_off", "num_s_only", "num_t_only", "sfx_conflict",
                "st_tset", "addr_wjac", "num_view", "num_new_min", "tok_view", "tok_new_min", "sa_same", "sa_other",
                "addr_eq_same", "sup_num_share", "s_num_share", "t_num_mode", "nm_edit", "name_tset", "leg_swap"]
SECOND_CTX = ["l_rank", "l_sum_o", "l_nconf_o", "l_nconf_same", "l_nconf_oth", "l_t_n", "l_t_odds_o", "l_best_o"]
SECOND = ["in_band", "bw"] + SECOND_STACK + ["xenc1", "xenc2", "xenc3"] + SECOND_CTX + ["l_logit"]
SECOND_PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 2000,
                 "feature_fraction": 0.8, "bagging_fraction": 0.5, "bagging_freq": 1, "lambda_l2": 10.0,
                 "verbosity": -1, "seed": 11, "deterministic": True, "force_row_wise": True}
SECOND_ROUNDS = 400
SECOND_MIN_P = 0.01
_LOGIT = (pl.col("p").clip(1e-6, 1 - 1e-6) / (1 - pl.col("p").clip(1e-6, 1 - 1e-6))).log()


def second_features(cfg: Config, split: str, pool: pl.DataFrame, x: pl.DataFrame) -> pl.DataFrame:
    """pool: s1, t, p = the decision input (link + test corrections); x: s1, t, xenc[, xenc2, xenc3] of the re-scored
    band pairs. Returns s1, t and SECOND (without l_logit) for the pairs with p > SECOND_MIN_P of train-country S1s
    (the only ones the second opinion corrects). List context is computed on p over every S1."""
    seen = load(cfg, "train", "s1", ["country"])["country"].unique().implode()
    ok = load(cfg, split, "s1", ["s1", "country"]).filter(pl.col("country").is_in(seen)).select("s1")
    tg = load(cfg, split, "tgt", ["t", "src", "addr_blank", "is_domain"])
    conf = (pl.col("p") > 0.5).cast(pl.Float32)
    odds = pl.col("p").clip(0, 1 - 1e-6) / (1 - pl.col("p").clip(0, 1 - 1e-6))
    y = (pool.filter(pl.col("p") > SECOND_MIN_P).select("s1", "t", "p").join(tg.select("t", "src"), on="t", how="left")
         .with_columns(l_rank=pl.col("p").rank("ordinal", descending=True).over("s1").cast(pl.Float32),
                       l_sum_o=(pl.col("p").sum().over("s1") - pl.col("p")).cast(pl.Float32),
                       l_nconf_o=conf.sum().over("s1") - conf,
                       l_nconf_same=conf.sum().over("s1", "src") - conf,
                       l_nconf_oth=conf.sum().over("s1") - conf.sum().over("s1", "src"),
                       l_t_n=pl.len().over("t").cast(pl.Float32),
                       l_t_odds_o=(odds.sum().over("t") - odds).cast(pl.Float32)))
    top2 = pl.col("p").sort(descending=True).slice(1, 1).first().over("s1")
    y = (y.with_columns(l_best_o=pl.when(pl.col("l_rank") == 1).then(top2).otherwise(pl.col("p").max().over("s1"))
                        .fill_null(0.0).cast(pl.Float32))
         .join(ok, on="s1", how="semi").drop("p", "src"))
    xs = x.select("s1", "t", xenc1="xenc", **{f"xenc{k}": f"xenc{k}" for k in (2, 3) if f"xenc{k}" in x.columns})
    keys = y.select("s1", "t")
    fd = pl.concat([pl.scan_parquet(p).select(["s1", "t"] + SECOND_STACK).join(keys.lazy(), on=["s1", "t"], how="semi")
                    .collect().with_columns([pl.col(c).cast(pl.Float32) for c in SECOND_STACK]) for p in _parts(cfg, split, 2)])
    y = (y.join(xs, on=["s1", "t"], how="left").join(fd, on=["s1", "t"], how="left")
         .join(tg.select("t", bw=(pl.col("addr_blank") | pl.col("is_domain")).cast(pl.Float32)), on="t", how="left")
         .with_columns(in_band=pl.col("xenc1").is_not_null().cast(pl.Float32)))
    return y.with_columns([pl.lit(None, pl.Float32).alias(c) for c in SECOND if c not in y.columns and c != "l_logit"])


def _second_x(y: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """y: s1, t, p (owned) + second_features columns -> (X in SECOND order, offset logit p)."""
    y = y.with_columns(l_logit=_LOGIT.cast(pl.Float32))
    return y.select(SECOND).to_numpy().astype(np.float32), y["l_logit"].to_numpy().astype(np.float64)


def second_adjust(feat: pl.DataFrame, booster, down_only: pl.DataFrame | None = None):
    """The apply_policy hook: owned p -> sigmoid(logit p + booster raw score) for the pairs in feat. down_only (s1, t):
    pairs the correction may lower but not raise."""
    def adjust(s: pl.DataFrame) -> pl.DataFrame:
        """Replace the owned p of pairs in feat by the second-opinion probability (never raised for down_only)."""
        y = s.join(feat, on=["s1", "t"], how="inner")
        if y.height == 0:
            return s
        X, off = _second_x(y)
        pc = 1.0 / (1.0 + np.exp(-(off + booster.predict(X, raw_score=True))))
        y = y.select("s1", "t", "p", pc=pl.Series(pc))
        if down_only is not None:
            y = (y.join(down_only.select("s1", "t", _d=pl.lit(True)), on=["s1", "t"], how="left")
                 .with_columns(pc=pl.when(pl.col("_d")).then(pl.min_horizontal("pc", "p")).otherwise(pl.col("pc"))))
        return (s.join(y.select("s1", "t", "pc"), on=["s1", "t"], how="left")
                .with_columns(p=pl.coalesce("pc", "p")).drop("pc"))
    return adjust


def second_train(cfg: Config, pool: pl.DataFrame, x: pl.DataFrame, policy: dict, log=print, out: Path | None = None) -> dict:
    """pool: s1, t, p of every train S1 with cross-fitted link probabilities (link_train); x: the _linked train frame.
    Trains the second opinion on dev folds of the test-like pool -> <out, default models/xlink>/second.txt; reports the
    sealed fold (test-like) with and without it."""
    t0 = time.time()
    gt = load(cfg, "train", "gt")
    tg = load(cfg, "train", "tgt", ["t", "addr_blank", "is_domain"])
    orphan = tg.filter(pl.col("addr_blank") | pl.col("is_domain")).join(gt.select("t").unique(), on="t", how="anti")
    pool = pool.select("s1", "t", "p").join(orphan.select("t"), on="t", how="anti")   # absent on test (bw_boost doc)
    pool = _corrections(cfg, "train", pool, False, log)   # as link_decide sees it (the decoy shift comes after)
    feat = second_features(cfg, "train", pool, x)
    folds = load(cfg, "train", "folds", ["s1", "fold"])
    y = (owned(pool, policy).join(feat, on=["s1", "t"]).join(folds, on="s1")
         .join(gt.select("s1", "t", label=pl.lit(1.0)), on=["s1", "t"], how="left").with_columns(pl.col("label").fill_null(0.0))
         .filter(pl.col("fold") != 4))
    X, off = _second_x(y)
    m = lgb.train(SECOND_PARAMS, lgb.Dataset(X, y["label"].to_numpy(), init_score=off, feature_name=SECOND),
                  num_boost_round=SECOND_ROUNDS)
    out = out or _dir(cfg, 2).parent / "xlink"
    m.save_model(str(out / "second.txt"))
    ids = folds.filter(pl.col("fold") == 4).select("s1")
    gt4 = gt.join(ids, on="s1", how="semi")
    country = load(cfg, "train", "s1", ["s1", "country"])
    res = {}
    for name, adj in (("test_like", None), ("second", second_adjust(feat, m))):
        f = per_entity(ids, apply_policy(pool, policy, adj).join(ids, on="s1", how="semi"), gt4).join(country, on="s1")
        res[name] = {c: float(v) for c, v in f.group_by("country").agg(pl.col("f").mean()).iter_rows()}
        res[name]["overall"] = float(f["f"].mean())
    d = {c: res["second"][c] - res["test_like"][c] for c in res["second"]}
    log(f"[xlink second] {y.height:,} dev pairs -> {out / 'second.txt'}; sealed test-like F0.5 "
        + ", ".join(f"{c} {res['test_like'][c]:.5f} -> {res['second'][c]:.5f} ({d[c]:+.5f})" for c in sorted(d))
        + f" ({time.time() - t0:.0f}s)")
    return res


def second_retrain(cfg: Config, log=print) -> None:
    """Second opinion alone, after a link-train that ran without it: rebuilds the cross-fitted link probabilities
    (same models as link_train, deterministic), trains models/xlink/second.txt and adds "second" to its meta.json."""
    out = _dir(cfg, 2).parent / "xlink"
    meta = json.loads((out / "meta.json").read_text())
    x = _linked(cfg, "train")
    xfit = _xfit(x, meta["features"], lgb.Booster(model_file=str(out / "model.txt")))
    pool = _replace(level_scores(cfg, "train", 2, log), pl.concat(xfit)).filter(pl.col("p") > 1e-4)
    meta["second"] = second_train(cfg, pool, x, meta["policy"], log)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


def _corrections(cfg: Config, split: str, pool: pl.DataFrame, shift: bool, log=print) -> pl.DataFrame:
    """Odds multipliers on the linked pool (env: BER_DECOY_SHIFT, BER_BW_BOOST, BER_TEXT_RULES)."""
    r = float(os.environ.get("BER_DECOY_SHIFT", "1"))
    if shift and r != 1.0:
        pool = decoy_shift(cfg, split, pool, r, log)
    k = float(os.environ.get("BER_BW_BOOST", "3"))   # on by default in v19 and v19ns (both test-only corrections)
    if k != 1.0:
        pool = bw_boost(cfg, split, pool, k, log)
    if os.environ.get("BER_TEXT_RULES", "1") == "1":   # linked V16 sealed, test-like: India +0.00004, US +0.00009
        pool = text_rules(pool, load(cfg, split, "s1", ["s1", "country", "core", "core_key", "script", "legal", "addr_nums"]),
                          load(cfg, split, "tgt", ["t", "core", "core_key", "script", "is_domain", "legal", "addr_nums",
                                                   "addr_blank"]), log)
    return pool


def link_decide(cfg: Config, split: str, pool: pl.DataFrame, x: pl.DataFrame, policy: dict, log=print,
                second_path: Path | None = None) -> pl.DataFrame:
    """pool: s1, t, p with the link applied (p > 1e-4); x: the re-scored band pairs. Returns the predicted s1, t.
    Order: bw_boost / text_rules -> single owner -> second opinion (BER_SECOND, default on; trained on the same
    bw-boosted test-like input) -> decoy shift -> expected-F0.5 sets. The decoy shift (BER_DECOY_SHIFT) is a test-only
    prior correction the second opinion never saw in training, so with the second opinion on it is applied after it
    (odds x r on top of the corrected p). BER_SECOND=0 gives the V20 path exactly (shift, bw, text rules, policy)."""
    r = float(os.environ.get("BER_DECOY_SHIFT", "1"))
    path = second_path or _dir(cfg, 2).parent / "xlink" / "second.txt"
    second = os.environ.get("BER_SECOND", "1") == "1"
    if second and not path.exists():
        log(f"[xlink] WARNING {split}: BER_SECOND=1 but {path} is missing (run link-train): no second opinion")
        second = False
    pool = _corrections(cfg, split, pool, not second, log)
    adjust = None
    if second:
        feat = second_features(cfg, split, pool, x)
        # BER_SECOND_CAP=1 (conservative): decoy-shift pairs may be lowered, never raised. Sealed, shift x0.5: US +0.00017
        # instead of +0.00026 (India +0.00028 / +0.00030); 2x twin decoys: US +0.00023 instead of +0.00032. On test the
        # uncapped model adds 3.8 US same-street different-number matches per 1k S1s (sealed 1.9), capped 0.65.
        cap = os.environ.get("BER_SECOND_CAP", "0") == "1"
        flag = decoy_flags(cfg, split, pool) if cap or r != 1.0 else None
        sec = second_adjust(feat, lgb.Booster(model_file=str(path)), flag if cap else None)
        adjust = sec if r == 1.0 else (lambda s: decoy_shift(cfg, split, sec(s), r, log, flag))
        log(f"[xlink] {split}: second opinion on {feat.height:,} pairs" + (f", then decoy shift x{r}" if r != 1.0 else ""))
    return apply_policy(pool, policy, adjust)


def link_predict(cfg: Config, split: str, log=print) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Test-time link: re-score train-country band pairs with the combiner, then apply the test-time
    corrections, second opinion and decision policy (link_decide).

    Returns: (predicted s1, t matches; all candidate pairs; model info with policy and sealed metrics)."""
    out = _dir(cfg, 2).parent / "xlink"
    meta = json.loads((out / "meta.json").read_text())
    m = lgb.Booster(model_file=str(out / "model.txt"))
    # Only countries the transformer and combiner were trained on. On a 2% decoy sample of test (most true
    # targets removed) the link cut US/India S1s with a predicted match from 9.5/14.1% to 8.8/9.0%, but
    # raised France's from 26.7% to 35.3%: out of distribution, it pushes French pairs up.
    # BER_LINK_ALL=1 re-scores every country (for a transformer that saw French pseudo pairs, see pseudo_pairs).
    seen = load(cfg, "train", "s1", ["country"])["country"].unique().implode()
    ok = load(cfg, split, "s1", ["s1", "country"])
    if os.environ.get("BER_LINK_ALL") != "1":
        ok = ok.filter(pl.col("country").is_in(seen))
    ok = ok.select("s1")
    x = _linked(cfg, split)
    n_all = x.height
    x = x.join(ok, on="s1", how="semi")
    missing = [f for f in meta["features"] if f not in x.columns]
    if missing:
        raise RuntimeError(f"{split}: link features {missing} missing (xenc_scores2.parquet not scored?)")
    x = x.with_columns(p_link=pl.Series(m.predict(x.select(meta["features"]).to_numpy())))
    scores = level_scores(cfg, split, 2, log)
    pred = link_decide(cfg, split, _replace(scores, x).filter(pl.col("p") > 1e-4), x, meta["policy"], log)
    log(f"[xlink] {split}: {x.height:,} of {n_all:,} uncertain pairs re-scored "
        f"({'all countries' if os.environ.get('BER_LINK_ALL') == '1' else 'train countries only'}), "
        f"{scores.height:,} candidates -> {pred.height:,} matches")
    return pred, scores.select("s1", "t"), {"policy": meta["policy"], "sealed": meta["sealed"]}


if __name__ == "__main__":
    s = pl.DataFrame({"s1": [0, 0, 1], "t": [5, 6, 5], "p": [0.5, 0.995, 0.3]})
    s1 = pl.DataFrame({"s1": [0, 1], "business_name": ["A", "B"], "business_address": ["1 x", None]})
    tg = pl.DataFrame({"t": [5, 6], "business_name": ["a", "a2"], "business_address": ["1 x", "1 x"], "src": [2, 2]})
    r = band_texts(s, s1, tg).sort("s1")
    assert r["t"].to_list() == [5, 5], r
    assert r["text_b_ctx"][0] == "a | 1 x [SIB] a2 | 1 x [RIVAL] B | ", r["text_b_ctx"][0]
    assert r["text_b_ctx"][1] == "a | 1 x [SIB] none [RIVAL] A | 1 x", r["text_b_ctx"][1]
    print("xlink self-check ok")
