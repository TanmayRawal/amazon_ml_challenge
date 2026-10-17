"""v12: translate Indic-script target names back to Latin with a dictionary learned from training pairs.

Indic-script targets are the English business name written word by word in an Indian script
("लाइफ इंडस्ट्रीज प्राइवेट लिमिटेड" = "Life Industries Private Limited"), drawn from a small
vocabulary. Aligning the words of true (Latin S1, Indic target) pairs position by position, when
both names have the same number of words, gives a dictionary of ~1,300 words that reproduces the
S1 name exactly for 84.5% of sealed-fold Indic true pairs and covers 91.7% of test Indic words.
The rule-based consonant skeletons (Stage 5) only approximate this.

Only dev-fold pairs are used to learn the dictionary. The translated name replaces the name
fields of Indic/mixed targets in tgt.parquet (core, legal, concat, ...), so every blocking pass
and feature sees Latin words; `script` keeps the original script, and `name_tr` holds the
translation. Words not in the dictionary stay in their script.
"""
from __future__ import annotations

import time

import polars as pl

from config import Config
from normalize import NAME_FIELDS, parallel_normalize
from prepare import NAME_SCHEMA, load

MIN_PURITY = 0.6
MIN_SUPPORT = 2
_INDIC_SCRIPTS = ["indic", "mixed"]


def _words(col: str, latin: bool) -> pl.Expr:
    """Lower-case a name column and split it into words (Latin alphanumerics, or any-script letters)."""
    pattern = r"[^a-z0-9]+" if latin else r"[^\p{L}\p{M}\p{N}]+"
    return pl.col(col).str.to_lowercase().str.replace_all(pattern, " ").str.strip_chars().str.split(" ")


def _path(cfg: Config):
    """Path of the learned dictionary, <work_dir>/models/indic_dict.parquet."""
    path = cfg.work_dir / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path / "indic_dict.parquet"


def learn(cfg: Config, log=print) -> pl.DataFrame:
    """Learn the Indic-word -> Latin-word dictionary from dev-fold true pairs of equal word count.

    Keeps words with purity >= MIN_PURITY and support >= MIN_SUPPORT, saves and returns the
    table (columns wi, lat, n, tot, purity).
    """
    t0 = time.time()
    folds = load(cfg, "train", "folds", ["s1", "fold"]).filter(pl.col("fold") != cfg.sealed_fold)
    tg = load(cfg, "train", "tgt", ["t", "business_name", "script"]).filter(pl.col("script").is_in(_INDIC_SCRIPTS))
    s1 = load(cfg, "train", "s1", ["s1", "business_name", "script"]).filter(pl.col("script") == "latin")
    pairs = (load(cfg, "train", "gt").join(folds.select("s1"), on="s1", how="semi")
             .join(tg.select("t", t_name="business_name"), on="t")
             .join(s1.select("s1", s_name="business_name"), on="s1")
             .with_columns(wi=_words("t_name", latin=False), wl=_words("s_name", latin=True)))
    aligned = (pairs.filter(pl.col("wi").list.len() == pl.col("wl").list.len()).select("wi", "wl")
               .explode(["wi", "wl"]).filter(~pl.col("wi").str.contains(r"^[a-z0-9]*$")))
    counts = aligned.group_by("wi", "wl").agg(n=pl.len())
    table = (counts.sort(["n", "wl"], descending=[True, False]).group_by("wi").agg(lat=pl.col("wl").first(),
                                                                                  n=pl.col("n").first())
             .join(counts.group_by("wi").agg(tot=pl.col("n").sum()), on="wi")
             .with_columns(purity=pl.col("n") / pl.col("tot"))
             .filter((pl.col("purity") >= MIN_PURITY) & (pl.col("tot") >= MIN_SUPPORT))
             .sort("wi"))
    table.write_parquet(_path(cfg))
    log(f"[indic-dict] {pairs.height:,} dev Indic true pairs, {aligned.height:,} aligned words -> "
        f"{table.height:,} dictionary entries ({time.time() - t0:.0f}s)")
    return table


def translate_names(names: pl.Series, table: pl.DataFrame) -> pl.Series:
    """Word-by-word translation; unknown words keep their script."""
    lut = dict(zip(table["wi"].to_list(), table["lat"].to_list()))
    df = pl.DataFrame({"name": names}).with_row_index("_r").with_columns(w=_words("name", latin=False))
    out = (df.explode("w").with_columns(w=pl.col("w").replace(lut))
           .group_by("_r", maintain_order=True).agg(pl.col("w").drop_nulls().str.join(" ")).sort("_r"))
    return out["w"].alias("name_tr")


def apply(cfg: Config, split: str, table: pl.DataFrame, log=print) -> None:
    """Translate Indic/mixed target names of a split and rewrite their name fields in tgt.parquet."""
    t0 = time.time()
    path = cfg.split_dir(split) / "tgt.parquet"
    tg = pl.read_parquet(path)
    if "name_tr" in tg.columns:
        tg = tg.drop("name_tr")
    ind = tg.filter(pl.col("script").is_in(_INDIC_SCRIPTS)).select("t", "business_name")
    ind = ind.with_columns(translate_names(ind["business_name"], table))
    uniq = ind["name_tr"].unique().to_list()
    rows = parallel_normalize("name", uniq, cfg.n_jobs)
    fields = [f for f in NAME_FIELDS if f != "script"]
    norm = (pl.DataFrame(rows, schema=NAME_SCHEMA, orient="row").drop("script")
            .with_columns(pl.Series("name_tr", uniq, dtype=pl.String)))
    new = ind.join(norm, on="name_tr", how="left").select(["t", "name_tr"] + fields)
    out = (tg.join(new, on="t", how="left", suffix="_new")
           .with_columns([pl.coalesce(f"{f}_new", f).alias(f) for f in fields])
           .drop([f"{f}_new" for f in fields]))
    if out.height != tg.height or not (out["t"] == tg["t"]).all():
        raise RuntimeError("translation changed the target table's rows")
    tmp = path.with_suffix(".tmp.parquet")
    out.write_parquet(tmp)
    tmp.replace(path)
    covered = ind.select(pl.col("name_tr").str.contains(r"[^\x00-\x7f]").not_().mean()).item()
    log(f"[indic-dict] {split}: translated {ind.height:,} Indic/mixed target names; fully Latin after "
        f"translation {covered:.3f} ({time.time() - t0:.0f}s)")


def run(cfg: Config, log=print) -> None:
    """Learn the dictionary and apply it to the train and test targets."""
    table = learn(cfg, log)
    for split in ("train", "test"):
        apply(cfg, split, table, log)
