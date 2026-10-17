"""Refined P1 on train, test-like (blank copies of hidden S1s removed, as test blank targets are ~95% owned)."""
import polars as pl

W = r"work"
D = r"data/dataset"
OUT = r"work/scratch/france2"
pl.Config.set_tbl_cols(30); pl.Config.set_fmt_str_lengths(60); pl.Config.set_tbl_width_chars(260); pl.Config.set_tbl_rows(80)

x = pl.read_parquet(f"{OUT}/p1_train.parquet")
raw = pl.read_csv(f"{D}/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema_length=0)
hid = (raw.filter((pl.col("source1_entity_id").str.extract(r"(\d+)$").cast(pl.Int64) % 100) < 20)
       .with_columns(e=pl.col("matched_entity_ids").str.split(",")).explode("e").filter(pl.col("e").is_not_null() & (pl.col("e") != ""))
       .select(entity_id="e").unique())
tid = pl.scan_parquet(f"{W}/train/tgt.parquet").select("t", "entity_id").join(x.select("t").unique().lazy(), on="t", how="semi").collect()
hc = tid.join(hid, on="entity_id", how="semi").select("t")
print("blank tie-targets that are hidden-S1 copies:", hc.height, "of", x["t"].n_unique())
x = x.join(hc, on="t", how="anti")


def add_rules(x):
    """Add candidate owner rules for tie groups: L1 (unique S1 with the target's legal form), L2 (L1 and every S1 has a
    legal form), F1 (unique folded-exact claimant, target has a legal form), N1 (target without legal form, unique
    S1 without one)."""
    return x.with_columns(
        n_same_leg=((pl.col("leg") == "same") & (pl.col("t_legal") != "")).sum().over("t"),
        n_noleg=(pl.col("s_legal") == "").sum().over("t"),
    ).with_columns(
        # target carries a legal form; exactly one S1 in the tie group carries the same one
        L1=(pl.col("leg") == "same") & (pl.col("t_legal") != "") & (pl.col("n_same_leg") == 1),
        # same, and no S1 in the group lacks a legal form (so 'legal added' cannot explain the target)
        L2=(pl.col("leg") == "same") & (pl.col("t_legal") != "") & (pl.col("n_same_leg") == 1) & (pl.col("n_noleg") == 0),
        # unique fold-exact (ordered tokens incl. legal) claimant, target has a legal form
        F1=pl.col("fold_eq") & (pl.col("n_fold") == 1) & (pl.col("t_legal") != ""),
        # target has no legal: unique no-legal S1
        N1=(pl.col("leg") == "same") & (pl.col("t_legal") == "") & (pl.col("n_same") == 1),
    )


x = add_rules(x)
x = x.with_columns(t_owned_in=pl.col("y").max().over("t"), t_claim=(pl.col("p") >= 0.5).max().over("t"))
tt = x.group_by("t").agg(pl.first("country"), ng=pl.first("ng"), own=pl.first("t_owned_in"), claim=pl.first("t_claim"))
print(tt.with_columns(g=pl.col("ng").clip(1, 4)).group_by("country", "g").agg(n=pl.len(), own=pl.col("own").mean(), claim=pl.col("claim").mean()).sort("country", "g"))
for r in ["L1", "L2", "F1", "N1", "R_leg", "R_top_leg"]:
    z = x.filter(pl.col(r) & (pl.col("ng") >= 2))
    print(r, z.group_by("country").agg(n=pl.len(), true=pl.col("y").mean(), own_in=pl.col("t_owned_in").mean(),
          true_given_own=pl.col("y").filter(pl.col("t_owned_in") == 1).mean(),
          un_n=(~pl.col("t_claim")).sum(), true_un=pl.col("y").filter(~pl.col("t_claim")).mean(),
          own_un=pl.col("t_owned_in").filter(~pl.col("t_claim")).mean(),
          true_given_own_un=pl.col("y").filter(~pl.col("t_claim") & (pl.col("t_owned_in") == 1)).mean()).sort("country"))
# who owns when L1 claimant is wrong (owned in group)?
z = x.filter(pl.col("ng") >= 2)
wrong_t = z.filter(pl.col("L1") & (pl.col("y") == 0) & (pl.col("t_owned_in") == 1)).select("t")
print("L1 wrong but owned-in-group: legal relation of the true owner")
print(z.join(wrong_t, on="t", how="semi").filter(pl.col("y") == 1).group_by("country", "leg").len().sort("country", "len", descending=True))
print(z.join(wrong_t, on="t", how="semi").filter(pl.col("y") == 1).group_by("country", "s_legal", "t_legal").len().sort("len", descending=True).head(20))
# India only by legal forms
print(x.filter(pl.col("L1") & (pl.col("ng") >= 2)).group_by("country", "t_legal").agg(n=pl.len(), true=pl.col("y").mean(), own=pl.col("t_owned_in").mean(),
      un=(~pl.col("t_claim")).sum(), true_un=pl.col("y").filter(~pl.col("t_claim")).mean()).sort("n", descending=True).head(20))
x.write_parquet(f"{OUT}/p1b_train.parquet")
