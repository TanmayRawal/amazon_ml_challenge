"""v21e3bx_fr2 = v21e3b (v21e3 US/India + France variant B)
   - the 1,602 US pairs removed in v21e3x (v21e3 pairs not in v21e3x)
   + FRANCE-2 additions (rules A, D; changes.csv), skipping any whose target another S1 already claims.
Row order and existing list order are kept; new ids are appended."""
import os
import polars as pl
SUB = r"submissions"
DS = r"data/dataset/test"
E3 = r"work/w20/submissions/v21e3/matching_results.tsv"
E3B = r"work/w20/submissions/v21e3b/submission_v21e3b/matching_results.tsv"
OUT = os.path.join(SUB, "v21e3bx_fr2")
rd = lambda f: pl.read_csv(f, separator="\t", quote_char=None, schema_overrides={"matched_entity_ids": pl.String}).with_columns(pl.col("matched_entity_ids").fill_null(""))
ex = lambda df: (df.select("source1_entity_id", m=pl.col("matched_entity_ids").str.split(",")).explode("m")
                 .filter(pl.col("m").is_not_null() & (pl.col("m") != "")))
e3, e3x, e3b = rd(E3), rd(os.path.join(SUB, "v21e3x", "matching_results.tsv")), rd(E3B)
s1 = pl.read_csv(DS + r"\test_source1.tsv", separator="\t", quote_char=None, columns=["entity_id", "country"]).rename({"entity_id": "source1_entity_id"})
assert e3b.height == e3.height == 1732544 and (e3b["source1_entity_id"] == e3["source1_entity_id"]).all()
d = e3.join(e3b, on="source1_entity_id", suffix="_b").filter(pl.col("matched_entity_ids") != pl.col("matched_entity_ids_b")).join(s1, on="source1_entity_id")
print("v21e3b vs v21e3 changed rows by country:", dict(d.group_by("country").len().iter_rows()))
rem = ex(e3).join(ex(e3x), on=["source1_entity_id", "m"], how="anti")
print("US removals (v21e3 - v21e3x):", rem.height)
pb = ex(e3b)
print("  of which present in v21e3b:", rem.join(pb, on=["source1_entity_id", "m"], how="semi").height)
ch = pl.read_csv(os.path.join(SUB, "v21e3x_fr2", "changes.csv"), infer_schema_length=0).select(source1_entity_id="s_id", m="t_id", rule="rule")
claimed = pb.join(rem, on=["source1_entity_id", "m"], how="anti")
already = ch.join(claimed, on=["source1_entity_id", "m"], how="semi")
other = ch.join(already, on=["source1_entity_id", "m"], how="anti").join(claimed.select("m", own="source1_entity_id"), on="m", how="inner")
add = ch.join(claimed, on=["source1_entity_id", "m"], how="anti").join(other.select("m"), on="m", how="anti")
print(f"FRANCE-2 adds: {ch.height}, already in v21e3b {already.height}, target owned by another S1 {other.height}, added {add.height}",
      dict(add.group_by("rule").len().iter_rows()))
final = pl.concat([claimed.with_columns(o=pl.lit(0)), add.select("source1_entity_id", "m", o=pl.lit(1))])
# keep existing order within each list, new ids appended
e3b_idx = e3b.with_row_index("r").select("source1_entity_id", "r")
lists = (final.with_row_index("k").sort("k").group_by("source1_entity_id", maintain_order=True)
         .agg(matched_entity_ids=pl.col("m").str.join(",")))
out = e3b_idx.join(lists, on="source1_entity_id", how="left").sort("r").select("source1_entity_id", pl.col("matched_entity_ids").fill_null(""))
os.makedirs(OUT, exist_ok=True)
out.write_csv(os.path.join(OUT, "matching_results.tsv"), separator="\t", quote_style="never")
add.write_csv(os.path.join(OUT, "fr2_added.csv")); rem.write_csv(os.path.join(OUT, "us_removed.csv"))
fp = ex(out)
print("pairs: v21e3b", pb.height, "final", fp.height, "| multi-claimed targets v21e3b", pb.group_by("m").len().filter(pl.col("len") > 1).height,
      "final", fp.group_by("m").len().filter(pl.col("len") > 1).height)
