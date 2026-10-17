"""Combined final: base file + additions (s1,t parquet lists, test index) - removals. Additions skip targets already
claimed by any S1 (after removals) and targets added twice (first list wins). Writes matching_results.tsv, the change
lists and candidate_pairs.tsv (base candidates + missing added pairs)."""
import argparse, os, csv, collections
import polars as pl
ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--base-cand", required=True)
ap.add_argument("--add", nargs="*", default=[])
ap.add_argument("--remove", nargs="*", default=[])
ap.add_argument("--out", required=True)
a = ap.parse_args()
W = r"work"
s1 = pl.read_parquet(rf"{W}\test\s1.parquet", columns=["s1", "entity_id", "country"])
tg = pl.read_parquet(rf"{W}\test\tgt.parquet", columns=["t", "entity_id"])
base = pl.read_csv(a.base, separator="\t", quote_char=None, schema_overrides={"matched_entity_ids": pl.String}).with_columns(pl.col("matched_entity_ids").fill_null(""))
pairs = (base.with_row_index("r").with_columns(m=pl.col("matched_entity_ids").str.split(",")).explode("m").filter(pl.col("m").is_not_null() & (pl.col("m") != ""))
         .with_columns(k=pl.int_range(pl.len()).over("r")).join(s1.select("s1", source1_entity_id="entity_id"), on="source1_entity_id")
         .join(tg.select("t", m="entity_id"), on="m").select("r", "k", "s1", "t"))
n0 = pairs.height
rem = pl.concat([pl.read_parquet(f).select("s1", "t") for f in a.remove]).unique() if a.remove else pl.DataFrame(schema={"s1": pl.UInt32, "t": pl.UInt32})
rem = rem.with_columns(pl.col("s1").cast(pl.UInt32), pl.col("t").cast(pl.UInt32))
pairs2 = pairs.join(rem, on=["s1", "t"], how="anti")
print(f"removals requested {rem.height}, applied {n0 - pairs2.height}")
claimed = pairs2.select("t").unique()
adds = []
for i, f in enumerate(a.add):
    x = pl.read_parquet(f).select(pl.col("s1").cast(pl.UInt32), pl.col("t").cast(pl.UInt32)).unique().with_columns(src=pl.lit(os.path.basename(f)), o=pl.lit(i))
    adds.append(x)
if adds:
    ad = pl.concat(adds).sort("o").unique("t", keep="first", maintain_order=True)
    ad = ad.join(claimed, on="t", how="anti").join(pairs2.select("s1", "t"), on=["s1", "t"], how="anti")
    print("additions applied by list:", dict(ad.group_by("src").len().iter_rows()))
else:
    ad = pl.DataFrame(schema={"s1": pl.UInt32, "t": pl.UInt32, "src": pl.String, "o": pl.Int32})
r_of = base.with_row_index("r").select("r", "source1_entity_id").join(s1.select("s1", source1_entity_id="entity_id"), on="source1_entity_id").select("r", "s1")
ad2 = ad.join(r_of, on="s1").with_columns(k=pl.lit(10**6) + pl.int_range(pl.len()).over("r").cast(pl.Int64))
allp = pl.concat([pairs2.with_columns(pl.col("k").cast(pl.Int64)), ad2.select("r", "k", "s1", "t")]).join(tg.select("t", m="entity_id"), on="t")
lists = allp.sort("r", "k").group_by("r", maintain_order=True).agg(matched_entity_ids=pl.col("m").str.join(","))
out = base.with_row_index("r").select("r", "source1_entity_id").join(lists, on="r", how="left").sort("r").select("source1_entity_id", pl.col("matched_entity_ids").fill_null(""))
assert out.height == base.height
os.makedirs(a.out, exist_ok=True)
out.write_csv(os.path.join(a.out, "matching_results.tsv"), separator="\t", quote_style="never")
chg_add = ad.join(s1.select("s1", s_id="entity_id", country="country"), on="s1").join(tg.select("t", t_id="entity_id"), on="t")
chg_add.select("s_id", "t_id", "country", "src").write_csv(os.path.join(a.out, "added.csv"))
chg_rem = pairs.join(rem, on=["s1", "t"], how="semi").join(s1.select("s1", s_id="entity_id", country="country"), on="s1").join(tg.select("t", t_id="entity_id"), on="t")
chg_rem.select("s_id", "t_id", "country").write_csv(os.path.join(a.out, "removed.csv"))
print(f"pairs: base {n0:,} -> {pairs2.height + ad.height:,} (removed {n0 - pairs2.height:,}, added {ad.height:,}); by country added",
      dict(chg_add.group_by("country").len().iter_rows()), "removed", dict(chg_rem.group_by("country").len().iter_rows()))
# candidates: base candidates + missing added pairs
addmap = collections.defaultdict(list)
for s, t in chg_add.select("s_id", "t_id").iter_rows():
    addmap[s].append(t)
n_app = 0
with open(a.base_cand, encoding="utf-8", newline="") as fi, open(os.path.join(a.out, "candidate_pairs.tsv"), "w", encoding="utf-8", newline="") as fo:
    fo.write(fi.readline())
    for line in fi:
        sid = line.split("\t", 1)[0]
        if sid in addmap:
            s, ids = line.rstrip("\n").split("\t", 1)
            have = set(ids.split(",")) if ids else set()
            miss = [t for t in addmap[sid] if t not in have]
            if miss:
                n_app += len(miss); ids = ",".join(([ids] if ids else []) + miss)
            line = f"{s}\t{ids}\n"
        fo.write(line)
print("candidate pairs appended:", n_app)
