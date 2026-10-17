"""candidate_pairs for v21e3x_fr2: v21e3 candidates + any added France pair not already a candidate (streamed)."""
import csv, collections
SUB = r"submissions/v21e3x_fr2"
add = collections.defaultdict(list)
with open(SUB + r"\changes.csv", encoding="utf-8", newline="") as f:
    for r in csv.DictReader(f):
        add[r["s_id"]].append(r["t_id"])
n_app = n_rows = 0
with open(r"work/w20/submissions/v21e3/candidate_pairs.tsv", encoding="utf-8", newline="") as fi, \
     open(SUB + r"\candidate_pairs.tsv", "w", encoding="utf-8", newline="") as fo:
    fo.write(fi.readline())
    for line in fi:
        n_rows += 1
        if line.split("\t", 1)[0] in add:
            s, ids = line.rstrip("\n").split("\t", 1)
            have = set(ids.split(",")) if ids else set()
            miss = [t for t in add[s] if t not in have]
            if miss:
                n_app += len(miss)
                ids = ",".join(([ids] if ids else []) + miss)
            line = f"{s}\t{ids}\n"
        fo.write(line)
print(f"rows {n_rows:,}, appended {n_app:,} of {sum(map(len, add.values())):,} added pairs")
