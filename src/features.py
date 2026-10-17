"""Plan A pair features for every candidate pair.

Set overlaps and flags are vectorized in polars; string ratios use rapidfuzz's
multi-threaded element-wise cpdist. Rank features are computed per S1 over the full
candidate list (each candidate part holds complete S1 lists).
"""
from __future__ import annotations

import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from blocking import PASS_COLS
from config import Config
from prepare import load

S1_COLS = ["s1", "country", "core", "core_key", "legal", "concat", "initials",
           "addr_nums", "addr_toks", "addr_blank"]
T_COLS = ["t", "country", "src", "core", "core_key", "legal", "concat", "acro_self", "domain",
          "is_domain", "script", "addr_nums", "addr_toks", "addr_blank"]
FEATURES = [
    "name_jac", "name_exact", "name_ratio", "name_tset", "concat_lvl", "concat_jw", "acro",
    "legal_match", "legal_conflict",
    "addr_jac", "addr_tset", "num_match", "num_conflict", "t_blank", "s_blank",
    "log_name_tdf", "log_name_rep",
    "rare_idf", "rare_n", "addr_n", "f_exact", "f_concat",
    "src3", "t_indic", "t_domain",
    "cheap_rank", "cheap_rel", "log_n_cand",
]
# Stage 3 (variant "c", ranked candidates): ranker score, rare-address-word overlap,
# extra string views, near-miss house numbers and competition for the same target.
FEATURES_C = FEATURES + [
    "rk_p", "a2_n", "a2_idf", "name_pratio", "name_jw", "addr_ratio",
    "num_first_eq", "num_near", "num_suffix", "t_n_s1", "t_best_gap", "t_is_best",
    # Stage 5: cross-script name skeletons (normalize.name_skeleton) and the ranker's skeleton pass
    "sk_n", "sk_idf", "sk_tset", "sk_jw",
    # V12: new ranker channels (name word pairs, truncated numbers, website grams), skeleton-word
    # overlap, any house number equal to the other side's minus its last digit (true copies drop a
    # trailing digit; decoys shift the number instead), share of S1 words spelled inside the website.
    "at_n", "np_n", "dm_n", "sim_sk_jac", "num0_tr", "num_trunc", "dom_cover",
    # V13: address word-pair and typo-key channels; first numbers a small numeric shift apart that is
    # not a dropped digit (decoys shift the number: 73 -> 74; true copies drop a digit instead).
    "ap_n", "fz_n", "num_shift",
    # V15 (France): house-number suffix conflict ("155 K" vs "155 J": 2% true on train against 52% when equal)
    # and street-name agreement without the city/region words (same number, street similarity <= 50: 0.1-4%
    # true against 81-84% above 80). IDF-weighted address overlap, so words most records of a country carry
    # (regions, states, "rue") count little; name evidence as the IDF of the shared name words (French names
    # are generic: "bordeaux union") and the S1's most distinctive name word.
    "sfx_conflict", "sfx_one", "st_tset", "st_cont", "addr_wjac", "nm_shared_idf", "nm_min_df",
    # V19: second-hop candidates (ranker.second_hop): key (0 ranked, 1 exact address, 2 house number + street words),
    # best seed probability, number of confident seed siblings sharing the key, key block size (0 for ranked pairs)
    "hop", "hop_p", "hop_n", "hop_bs",
    # V19 audit: num_first_eq & co. now compare house numbers (street segment); the old first-number agreement stays
    "num_first_raw",
    # V20 (decoy study, experiments/decoys): the one differing house number's signed change
    # (UP by 1-9 is decoy-prone, DOWN is a true copy's noise), the one differing name word's edit type (a letter
    # appended at the word's end is decoy-prone), India legal form "pvt ltd" -> "pvt" / -> "ltd"
    "num_delta", "nm_edit", "leg_pvt_only", "leg_ltd_only",
]
CAND_DIR = {"a": "cand_a", "b": "cand_b", "c": "cand_b"}


def _toks(col: str) -> pl.Expr:
    """Split a space-separated string column into a list of non-empty tokens."""
    return pl.col(col).str.split(" ").list.eval(pl.element().filter(pl.element() != ""))


def _inter(a: str, b: str) -> pl.Expr:
    """Size of the set intersection of two list columns."""
    return pl.col(a).list.set_intersection(pl.col(b)).list.len()


def _jac(a: str, b: str) -> pl.Expr:
    """Jaccard similarity of two list columns (0.0 when both are empty)."""
    union = pl.col(a).list.set_union(pl.col(b)).list.len()
    return pl.when(union > 0).then(_inter(a, b) / union).otherwise(0.0)


def _prefix_levels(a: str, b: str) -> pl.Expr:
    """Count of prefix lengths (4, 6, 8, 12 chars) at which two string columns agree (Int8, 0-4)."""
    total = pl.lit(0, pl.Int8)
    for length in (4, 6, 8, 12):
        ok = (
            (pl.col(a).str.len_chars() >= length)
            & (pl.col(b).str.len_chars() >= length)
            & (pl.col(a).str.slice(0, length) == pl.col(b).str.slice(0, length))
        )
        total = total + ok.cast(pl.Int8)
    return total


def _ratio(df: pl.DataFrame, a: str, b: str, scorer, scale: float) -> np.ndarray:
    """Row-wise rapidfuzz similarity between string columns `a` and `b`.

    Args:
        df: pair frame holding both columns.
        scorer: rapidfuzz scorer (e.g. fuzz.ratio, JaroWinkler.normalized_similarity).
        scale: divisor bringing the score to [0, 1] (100.0 for fuzz scorers, 1.0 otherwise).
    Returns:
        float32 array of length df.height; 0.0 where either side is empty.
    """
    left, right = df[a].to_list(), df[b].to_list()
    out = process.cpdist(left, right, scorer=scorer, workers=-1, dtype=np.float32) / scale
    empty = ((df[a] == "") | (df[b] == "")).to_numpy()
    out[empty] = 0.0
    return out


def name_frequencies(s1: pl.DataFrame, tg: pl.DataFrame) -> pl.DataFrame:
    """How many same-country targets / S1 records carry exactly this core name."""
    tdf = tg.filter(pl.col("core_key") != "").group_by("country", "core_key").agg(name_tdf=pl.len())
    rep = s1.filter(pl.col("core_key") != "").group_by("country", "core_key").agg(name_rep=pl.len())
    return (
        s1.join(tdf, on=["country", "core_key"], how="left")
        .join(rep, on=["country", "core_key"], how="left")
        .with_columns(pl.col("name_tdf", "name_rep").fill_null(0))
        .sort("s1")
    )


def _near_numbers(df: pl.DataFrame) -> np.ndarray:
    """First house numbers that differ by exactly one edit (9914 vs 914, 103 vs 303)."""
    left, right = df["_s0"].to_list(), df["_t0"].to_list()
    dist = process.cpdist(left, right, scorer=Levenshtein.distance, workers=-1, dtype=np.int32)
    empty = ((df["_s0"] == "") | (df["_t0"] == "")).to_numpy()
    return ((dist == 1) & ~empty).astype(np.float32)


def _dom_cover(df: pl.DataFrame) -> np.ndarray:
    """Share of the S1's Latin core words (3+ letters) found inside the target's website label."""
    out = np.zeros(df.height, dtype=np.float32)
    idx = np.flatnonzero((df["_t_dom"] != "").to_numpy())
    for k, core, dom in zip(idx, df["s_core"].gather(idx).to_list(), df["_t_dom"].gather(idx).to_list()):
        words = [w for w in core.split() if len(w) >= 3 and w.isascii()]
        if words:
            out[k] = sum(w in dom for w in words) / len(words)
    return out


def idf_table(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """df: country + space-separated token column. Per-country log(N / document frequency)."""
    n = df.group_by("country").agg(N=pl.len())
    return (df.select("country", tok=pl.col(col).str.split(" ").list.unique()).explode("tok")
            .filter(pl.col("tok").is_not_null() & (pl.col("tok") != ""))
            .group_by("country", "tok").agg(df=pl.len()).join(n, on="country")
            .select("country", "tok", idf=(pl.col("N") / pl.col("df")).log().cast(pl.Float32), df="df"))


def _idf_sum(df: pl.DataFrame, lists: pl.Expr, idf: pl.DataFrame) -> pl.Series:
    """Sum of per-country IDF over each row's token list (s_country gives the country)."""
    x = (df.select(_i=pl.int_range(pl.len(), dtype=pl.UInt32), country=pl.col("s_country"), tok=lists).explode("tok")
         .join(idf.select("country", "tok", "idf"), on=["country", "tok"], how="left")
         .group_by("_i").agg(pl.col("idf").sum()).sort("_i"))
    if x.height != df.height:
        raise RuntimeError("idf sum lost rows")
    return x["idf"].fill_null(0.0)


def v15_record_fields(s1: pl.DataFrame, tg: pl.DataFrame, s_addr: pl.DataFrame, t_addr: pl.DataFrame,
                      n_jobs: int) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Per-record inputs of the V15 features. s1/tg: sorted frames with country, core, addr_toks;
    s_addr/t_addr: id + business_address. IDF tables are unsupervised per-country counts of this split."""
    idf = {"addr": idf_table(pl.concat([s1.select("country", "addr_toks"), tg.select("country", "addr_toks")]), "addr_toks"),
           "name": idf_table(tg.select("country", "core"), "core")}

    def addr_sum(df: pl.DataFrame, idc: str) -> pl.DataFrame:
        """Add addr_idf: the summed per-country IDF of each record's address tokens."""
        x =(df.select(idc, "country", tok=_toks("addr_toks")).explode("tok")
             .join(idf["addr"].select("country", "tok", "idf"), on=["country", "tok"], how="left")
             .group_by(idc).agg(addr_idf=pl.col("idf").sum().fill_null(0.0)))
        return df.join(x, on=idc, how="left")

    nm = (s1.select("s1", "country", tok=_toks("core")).explode("tok")
          .join(idf["name"].select("country", "tok", "df"), on=["country", "tok"], how="left")
          .group_by("s1").agg(nm_min_df=pl.col("df").fill_null(0).min().cast(pl.Float32).log1p()))
    s1 = (addr_sum(s1, "s1").join(nm, on="s1", how="left").join(street_fields(s_addr, n_jobs), on="s1", how="left")
          .with_columns(pl.col("nm_min_df").fill_null(0.0)).sort("s1"))
    tg = addr_sum(tg, "t").join(street_fields(t_addr, n_jobs), on="t", how="left").sort("t")
    return s1, tg, idf


def street_fields(df: pl.DataFrame, n_jobs: int) -> pl.DataFrame:
    """df: business_address. Adds street_sfx, street_core and street_house (normalize.norm_street)."""
    from normalize import parallel_normalize
    u = df["business_address"].unique().to_list()
    r = parallel_normalize("street", u, min(n_jobs, 8))   # cheap parse; few workers keep memory low
    tab = pl.DataFrame({"business_address": u, "street_sfx": [a for a, _, _ in r], "street_core": [b for _, b, _ in r],
                        "street_house": [c for _, _, c in r]},
                       schema={"business_address": pl.String, "street_sfx": pl.String, "street_core": pl.String,
                               "street_house": pl.String})
    return df.join(tab, on="business_address", how="left", maintain_order="left").drop("business_address")


def _v15(df: pl.DataFrame, idf: dict) -> pl.DataFrame:
    """Add the V15 street / IDF features to a pair frame.

    Adds sfx_conflict, sfx_one (house-number suffix disagreement), st_tset, st_cont
    (street-name similarity without city/region words), addr_wjac (IDF-weighted address
    Jaccard), nm_shared_idf (IDF of shared name words) and nm_min_df.

    Args:
        df: pair frame with s_/t_ street fields, token lists and addr_idf columns.
        idf: {"addr": ..., "name": ...} per-country IDF tables from v15_record_fields.
    """
    sa, sb = pl.col("s_street_sfx"), pl.col("t_street_sfx")
    ca, cb = _toks("s_street_core"), _toks("t_street_core")
    df = df.with_columns(_sca=ca, _scb=cb)
    small = pl.min_horizontal(pl.col("_sca").list.len(), pl.col("_scb").list.len())
    st = _ratio(df, "s_street_core", "t_street_core", fuzz.token_set_ratio, 100.0)
    st[((df["s_street_core"] == "") | (df["t_street_core"] == "")).to_numpy()] = -1.0
    shared = _idf_sum(df, pl.col("s_atok").list.set_intersection(pl.col("t_atok")), idf["addr"])
    union = pl.col("s_addr_idf") + pl.col("t_addr_idf") - pl.col("_ai")
    return df.with_columns(_ai=shared, nm_shared_idf=_idf_sum(df, pl.col("s_ctok").list.set_intersection(pl.col("t_ctok")), idf["name"])).with_columns(
        sfx_conflict=(sa != "") & (sb != "") & (sa != sb),
        sfx_one=(sa != "") != (sb != ""),
        st_tset=pl.Series(st),
        st_cont=pl.when(small > 0).then(pl.col("_sca").list.set_intersection("_scb").list.len() / small).otherwise(-1.0),
        addr_wjac=pl.when(union > 0).then(pl.col("_ai") / union).otherwise(0.0),
        nm_min_df=pl.col("s_nm_min_df"),
    )


def _edit_code(sc: str, tc: str) -> int:
    """The one differing name word's edit: 0 none/other, 1 letter appended at the word's end (DMD -> DMDX),
    2 inserted elsewhere, 3 last letter replaced, 4 replaced elsewhere, 5 deleted."""
    a, b = sc.split(), tc.split()
    new, gone = [w for w in b if w not in a], [w for w in a if w not in b]
    if len(new) != 1 or len(gone) != 1 or not set(a) & set(b) or Levenshtein.distance(gone[0], new[0]) > 2:
        return 0
    ops = Levenshtein.editops(gone[0], new[0])
    if len(ops) != 1:
        return 0
    o, g = ops[0], gone[0]
    if o.tag == "insert":
        return 1 if o.src_pos >= len(g) - 1 else 2
    if o.tag == "replace":
        return 3 if o.src_pos == len(g) - 1 else 4
    return 5


def _decoy_signals(df: pl.DataFrame) -> pl.DataFrame:
    """Add the V20 decoy-detection features to a pair frame.

    num_delta: signed change (clipped to +-20) of the single differing house number, else null;
    nm_edit: edit type of the single differing name word (see _edit_code);
    leg_pvt_only / leg_ltd_only: India "pvt ltd" legal form reduced to "pvt" / "ltd".
    """
    so = pl.col("s_num").list.set_difference("t_num")
    to = pl.col("t_num").list.set_difference("s_num")
    one = (so.list.len() == 1) & (to.list.len() == 1) & ~pl.col("t_addr_blank")
    delta = to.list.first().cast(pl.Int64, strict=False) - so.list.first().cast(pl.Int64, strict=False)
    word = ((pl.col("s_ctok").list.set_difference("t_ctok").list.len() == 1)
            & (pl.col("t_ctok").list.set_difference("s_ctok").list.len() == 1)).fill_null(False)
    ii = np.flatnonzero(df.select(word.alias("w"))["w"].to_numpy())
    code = np.zeros(df.height, dtype=np.float32)
    code[ii] = [_edit_code(a or "", b or "") for a, b in zip(df["s_core"].gather(ii).to_list(), df["t_core"].gather(ii).to_list())]
    return df.with_columns(
        num_delta=pl.when(one).then(delta.clip(-20, 20)).otherwise(None).cast(pl.Float32),
        nm_edit=pl.Series(code),
        leg_pvt_only=(pl.col("s_legal") == "ltd pvt") & (pl.col("t_legal") == "pvt"),
        leg_ltd_only=(pl.col("s_legal") == "ltd pvt") & (pl.col("t_legal") == "ltd"),
    )


def _extended(df: pl.DataFrame) -> pl.DataFrame:
    """Add the stage-3+ (variant "c") pair features on top of the base FEATURES.

    Covers house-number comparisons (num_first_eq, num_first_raw, num_suffix, num_near,
    num_trunc, num_shift), target competition (t_n_s1, t_best_gap, t_is_best), extra
    string views (name_pratio, name_jw, addr_ratio, sk_tset, sk_jw), dom_cover, rk_p and
    the decoy signals from _decoy_signals.
    """
    # Audit 1: the house number of the street segment, not the address's first number ("Unit 1416, 6515 Belcrest
    # Road" is house 6515); the first number only when the street segment has none. num_match keeps any overlap.
    house = lambda c: pl.when(pl.col(f"{c}_street_house") != "").then(pl.col(f"{c}_street_house"))  # noqa: E731
    s0 = house("s").otherwise(pl.col("s_num").list.first())
    t0 = house("t").otherwise(pl.col("t_num").list.first())
    long = pl.element().filter(pl.element().str.len_chars() >= 3)
    def cut(c: str) -> pl.Expr:   # V12: last digit dropped; V13: or the first
        """Truncated variants of each 3+ digit number in list column `c` (last or first digit removed)."""
        return pl.concat_list(pl.col(c).list.eval(long.str.head(-1)),
                              pl.col(c).list.eval(long.str.tail(-1).str.strip_chars_start("0")))
    df = df.with_columns(_s_tr=cut("s_num"), _t_tr=cut("t_num"))
    n_s, n_t = s0.cast(pl.Int64, strict=False), t0.cast(pl.Int64, strict=False)
    df = df.with_columns(
        num_trunc=(pl.col("_s_tr").list.set_intersection("t_num").list.len() > 0)
        | (pl.col("_t_tr").list.set_intersection("s_num").list.len() > 0),
        dom_cover=pl.Series(_dom_cover(df)),
        num_shift=((n_s != n_t) & ((n_s - n_t).abs() <= 10) & ~s0.str.starts_with(t0) & ~t0.str.starts_with(s0)
                   & ~s0.str.ends_with(t0) & ~t0.str.ends_with(s0)).fill_null(False),
    )
    df = df.with_columns(
        rk_p=pl.col("cheap"),
        num_first_eq=(s0 == t0).fill_null(False),
        num_first_raw=(pl.col("s_num").list.first() == pl.col("t_num").list.first()).fill_null(False),
        num_suffix=((s0 != t0) & (s0.str.ends_with(t0) | t0.str.ends_with(s0))).fill_null(False),
        t_n_s1=pl.col("t_n_s1").cast(pl.Float32).log1p(),
        t_best_gap=pl.col("cheap") - pl.col("t_max_rk"),
        t_is_best=pl.col("cheap") >= pl.col("t_max_rk"),
        _s0=s0.fill_null(""),
        _t0=t0.fill_null(""),
    )
    return df.with_columns(
        name_pratio=pl.Series(_ratio(df, "s_core", "t_core", fuzz.partial_ratio, 100.0)),
        name_jw=pl.Series(_ratio(df, "s_core", "t_core", JaroWinkler.normalized_similarity, 1.0)),
        addr_ratio=pl.Series(_ratio(df, "s_addr_toks", "t_addr_toks", fuzz.ratio, 100.0)),
        num_near=pl.Series(_near_numbers(df)),
        sk_tset=pl.Series(_ratio(df, "s_name_sk", "t_name_sk", fuzz.token_set_ratio, 100.0)),
        sk_jw=pl.Series(_ratio(df, "s_name_sk", "t_name_sk", JaroWinkler.normalized_similarity, 1.0)),
    ).pipe(_decoy_signals)


def competition(cfg: Config, split: str) -> pl.DataFrame:
    """Per target: how many S1s list it among their ranked candidates, and the best ranker score."""
    parts = sorted((cfg.split_dir(split) / "cand_b").glob("part_*.parquet"))
    c = pl.concat([
        pl.read_parquet(p, columns=["t", "cheap", "rrank"]).filter(pl.col("rrank") <= cfg.rk_cand_k).select("t", "cheap")
        for p in parts
    ])
    return c.group_by("t").agg(t_n_s1=pl.len(), t_max_rk=pl.col("cheap").max())


def pair_features(cand: pl.DataFrame, s1: pl.DataFrame, tg: pl.DataFrame, extended: bool = False,
                  idf: dict | None = None) -> pl.DataFrame:
    """cand: s1, t, pass columns, cheap, rank features. s1/tg must be sorted by index."""
    left = s1[cand["s1"].to_numpy()].select(pl.all().name.prefix("s_")).drop("s_s1")
    right = tg[cand["t"].to_numpy()].select(pl.all().name.prefix("t_")).drop("t_t")
    df = pl.concat([cand, left, right], how="horizontal")
    df = df.with_columns(
        s_ctok=_toks("s_core"), t_ctok=_toks("t_core"),
        s_atok=_toks("s_addr_toks"), t_atok=_toks("t_addr_toks"),
        s_num=_toks("s_addr_nums"), t_num=_toks("t_addr_nums"),
        s_leg=_toks("s_legal"), t_leg=_toks("t_legal"),
    )
    num_inter = _inter("s_num", "t_num")
    leg_inter = _inter("s_leg", "t_leg")
    df = df.with_columns(
        name_jac=_jac("s_ctok", "t_ctok"),
        name_exact=(pl.col("s_core_key") == pl.col("t_core_key")) & (pl.col("s_core_key") != ""),
        concat_lvl=pl.max_horizontal(_prefix_levels("s_concat", "t_concat"), _prefix_levels("s_concat", "t_domain")),
        acro=(pl.col("t_acro_self") != "") & (pl.col("t_acro_self") == pl.col("s_initials")),
        legal_match=leg_inter > 0,
        legal_conflict=(pl.col("s_leg").list.len() > 0) & (pl.col("t_leg").list.len() > 0) & (leg_inter == 0),
        addr_jac=_jac("s_atok", "t_atok"),
        num_match=num_inter > 0,
        num_conflict=(pl.col("s_num").list.len() > 0) & (pl.col("t_num").list.len() > 0) & (num_inter == 0),
        t_blank=pl.col("t_addr_blank"),
        s_blank=pl.col("s_addr_blank"),
        log_name_tdf=pl.col("s_name_tdf").log1p(),
        log_name_rep=pl.col("s_name_rep").log1p(),
        src3=pl.col("t_src") == 3,
        t_indic=pl.col("t_script").is_in(["indic", "mixed"]),
        t_domain=pl.col("t_is_domain"),
        _t_dom=pl.col("t_domain"),   # V12: the website label itself, before t_domain becomes the flag
    )
    df = df.with_columns(
        name_ratio=pl.Series(_ratio(df, "s_core", "t_core", fuzz.ratio, 100.0)),
        name_tset=pl.Series(_ratio(df, "s_core", "t_core", fuzz.token_set_ratio, 100.0)),
        concat_jw=pl.Series(_ratio(df, "s_concat", "t_concat", JaroWinkler.normalized_similarity, 1.0)),
        addr_tset=pl.Series(_ratio(df, "s_addr_toks", "t_addr_toks", fuzz.token_set_ratio, 100.0)),
    )
    feats = FEATURES
    if extended:
        df = _v15(_extended(df), idf)
        feats = FEATURES_C
    keep = ["s1", "t"] + [c for c in ("label", "fold") if c in df.columns]
    return df.select(keep + [pl.col(f).cast(pl.Float32) for f in feats])


def build_features(cfg: Config, split: str, log=print, variant: str = "a") -> None:
    """variant "a": Plan A candidates; "b": ranked candidates cut to the top rk_cand_k per S1."""
    t0 = time.time()
    s1 = name_frequencies(load(cfg, split, "s1", S1_COLS), load(cfg, split, "tgt", ["t", "country", "core_key"]))
    tg = load(cfg, split, "tgt", T_COLS).sort("t")
    gt = folds = None
    if split == "train":
        gt = load(cfg, split, "gt").with_columns(label=pl.lit(1, pl.UInt8))
        folds = load(cfg, split, "folds", ["s1", "fold"]).sort("s1")
    out_dir = cfg.split_dir(split) / f"feat_{variant}"
    out_dir.mkdir(exist_ok=True)
    for old in out_dir.glob("part_*.parquet"):
        old.unlink()
    parts = sorted((cfg.split_dir(split) / CAND_DIR[variant]).glob("part_*.parquet"))
    extended = variant == "c"
    comp = competition(cfg, split) if extended else None
    idf = None
    if extended:
        s1 = s1.join(load(cfg, split, "s1_sk"), on="s1", how="left").sort("s1")
        tg = tg.join(load(cfg, split, "tgt_sk"), on="t", how="left").sort("t")
        s1, tg, idf = v15_record_fields(s1, tg, load(cfg, split, "s1", ["s1", "business_address"]),
                                        load(cfg, split, "tgt", ["t", "business_address"]), cfg.n_jobs)
        log(f"[{split}] V15 record fields ({time.time() - t0:.0f}s)")
    for i, part in enumerate(parts):
        cand = pl.read_parquet(part)
        if "rrank" in cand.columns:
            cand = cand.filter(pl.col("rrank") <= cfg.rk_cand_k)
        if extended:
            cand = cand.with_columns([pl.lit(0.0, pl.Float32).alias(h) for h in ("hop", "hop_p", "hop_n", "hop_bs")
                                      if h not in cand.columns])
        if comp is not None:
            cand = cand.join(comp, on="t", how="left")
        cand = cand.with_columns(
            cheap_rank=pl.col("cheap").rank("min", descending=True).over("s1"),
            cheap_rel=pl.col("cheap") / pl.col("cheap").max().over("s1"),
            log_n_cand=pl.len().over("s1").cast(pl.Float32).log1p(),
        )
        if gt is not None:
            cand = (
                cand.join(gt, on=["s1", "t"], how="left")
                .with_columns(pl.col("label").fill_null(0))
                .join(folds, on="s1", how="left")
            )
        outs = []
        for lo in range(0, cand.height, cfg.feat_chunk):
            outs.append(pair_features(cand.slice(lo, cfg.feat_chunk), s1, tg, extended, idf))
        pl.concat(outs).write_parquet(out_dir / f"part_{i:03d}.parquet")
        log(f"[{split}] features part {i}: {cand.height:,} pairs ({time.time() - t0:.0f}s)")


def feature_parts(cfg: Config, split: str, variant: str = "a") -> list:
    """Sorted list of feature parquet paths written by build_features for a split and variant."""
    return sorted((cfg.split_dir(split) / f"feat_{variant}").glob("part_*.parquet"))
