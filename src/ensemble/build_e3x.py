"""v21e3x = v21e3 minus the US exact-name, same-street, digit-substituted-number pairs that v21e3's correction added
over v21e (the class where test acceptance exceeds train truth). Writes matching_results.tsv in v21e3's row order."""
import polars as pl
W = r"work"
rm = pl.read_parquet(rf"{W}\w20\e3_us_numsub_added.parquet")
s1 = pl.read_parquet(rf"{W}\test\s1.parquet", columns=["s1", "entity_id"])
tg = pl.read_parquet(rf"{W}\test\tgt.parquet", columns=["t", "entity_id"])
rm_ids = rm.join(s1, on="s1").join(tg, on="t", suffix="_t").select(s="entity_id", m="entity_id_t")
drop = {}
for s, m in rm_ids.iter_rows():
    drop.setdefault(s, set()).add(m)
src = rf"{W}\w20\submissions\v21e3\matching_results.tsv"
out_dir = rf"{W}\w20\submissions\v21e3x"
import os; os.makedirs(out_dir, exist_ok=True)
n_rows = n_pairs = 0
with open(src, "rb") as f, open(out_dir + r"\matching_results.tsv", "wb") as g:
    for i, line in enumerate(f):
        if i == 0:
            g.write(line); continue
        body = line.rstrip(b"\r\n"); end = line[len(body):]
        s, rest = body.split(b"\t", 1)
        k = s.decode()
        if k in drop:
            ids = [x for x in rest.decode().split(",") if x and x not in drop[k]]
            n_pairs += len(rest.decode().split(",")) - len(ids)
            body = (k + "\t" + ",".join(ids)).encode(); n_rows += 1
        g.write(body + end)
print(f"removed {n_pairs} pairs in {n_rows} rows (expected {rm.height})")
