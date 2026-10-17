"""French ties by stop words (v22), exploration: copy counts of French S1s.

Using the current French predictions (final_pairs + acro_add + fr_tie_add), prints how many S2 / S3 copies each French
S1 gets, the target count by source and blank address, and sample blank-address names per source. Prints only.
"""
import polars as pl
W = r"work"
s1 = pl.read_parquet(rf"{W}\test\s1.parquet", columns=["s1","country","business_name","core_key","legal"])
tg = pl.read_parquet(rf"{W}\test\tgt.parquet", columns=["t","country","src","business_name","core_key","legal","addr_blank","is_domain"])
fin = pl.concat([d.select(pl.col("s1").cast(pl.Int64), pl.col("t").cast(pl.Int64)) for d in [pl.read_parquet(rf"{W}\w20\final_pairs.parquet"), pl.read_parquet(rf"{W}\w20\acro_add.parquet").select("s1","t"), pl.read_parquet(rf"{W}\w20\fr_tie_add.parquet").select("s1","t")]]).unique()
fr = s1.filter(pl.col("country")=="France")
p = fin.join(fr.select(pl.col("s1").cast(pl.Int64)), on="s1").join(tg.select(pl.col("t").cast(pl.Int64),"src","addr_blank"), on="t")
print("french pairs", p.height)
c = p.group_by("s1","src").len()
print(c.group_by("src","len").agg(n=pl.len()).sort("src","len"))
c2 = p.group_by("s1").agg(n2=(pl.col("src")==2).sum(), n3=(pl.col("src")==3).sum())
print(c2.group_by("n2","n3").len().sort("len",descending=True).head(15))
print("french S1:", fr.height, "with any pred:", c2.height)
print(tg.filter(pl.col("country")=="France").group_by("src","addr_blank").len())
# raw name samples by src
print(tg.filter((pl.col("country")=="France")&pl.col("addr_blank")).group_by("src").agg(pl.col("business_name").sample(12, seed=3)).explode("business_name"))
