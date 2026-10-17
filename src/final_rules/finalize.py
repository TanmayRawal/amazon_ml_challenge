"""Turn the ensemble output (v21e3b) into the final submission (v24) by applying the shipped change lists.

The final submission is the two-pipeline ensemble with the second-opinion correction (v21e3b), followed by
the validated rule overlays documented in README_v22.md, README_v23.md and README_v24.md. This script applies
them in the same order they were built, using the exact pair lists that ship with the code:

  1. remove the 1,602 US digit-substitution pairs added by the second opinion   (../ensemble/us_removed.csv)
  2. add the 2,500 French rule-D (domain) and rule-A (legal-form tie) pairs     (../ensemble/fr2_added.csv)
  3. remove the v22-v24 removals (decoys, ensemble, duplicates, caps, singleton guard)  (changes/removed.csv)
  4. add the v22-v24 additions (acronyms, ties, domains, recovered pairs, ensemble, aliases)  (changes/added.csv)

An addition is skipped if another Source-1 record already claims the target, so every target keeps at most
one owner. Every added pair is also appended to that Source-1 record's candidate list, so all matches are
candidates. Pure Python, no third-party packages.

Usage:
    python finalize.py <ensemble_dir with matching_results.tsv and candidate_pairs.tsv> <out_dir>
Writes <out_dir>/matching_results.tsv and <out_dir>/candidate_pairs.tsv.
"""
import collections
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [
    ("remove", os.path.join(HERE, "..", "ensemble", "us_removed.csv"), "source1_entity_id", "m"),
    ("add", os.path.join(HERE, "..", "ensemble", "fr2_added.csv"), "source1_entity_id", "m"),
    ("remove", os.path.join(HERE, "changes", "removed.csv"), "s_id", "t_id"),
    ("add", os.path.join(HERE, "changes", "added.csv"), "s_id", "t_id"),
]


def read_lists(path):
    """Read a submission-style TSV into (row order, {source1_entity_id: [target ids]})."""
    order, lists = [], {}
    with open(path, encoding="utf-8", newline="") as f:
        header = f.readline()
        for line in f:
            s, _, ids = line.rstrip("\r\n").partition("\t")
            order.append(s)
            lists[s] = [t for t in ids.split(",") if t]
    return header, order, lists


def read_pairs(path, col_s, col_t):
    """Read (source1_entity_id, target_id) pairs from a change-list CSV, in file order."""
    with open(path, encoding="utf-8", newline="") as f:
        return [(r[col_s], r[col_t]) for r in csv.DictReader(f)]


def apply_steps(match):
    """Apply the removal and addition steps in order to `match` (dict S1 -> list of targets); return added pairs."""
    added = []
    for kind, path, cs, ct in STEPS:
        pairs = read_pairs(path, cs, ct)
        if kind == "remove":
            drop = collections.defaultdict(set)
            for s, t in pairs:
                drop[s].add(t)
            for s, ts in drop.items():
                if s in match:
                    match[s] = [t for t in match[s] if t not in ts]
            print(f"{os.path.basename(path)}: removal list of {len(pairs):,} pairs applied")
        else:
            owner = {t: s for s, ts in match.items() for t in ts}
            n_add = n_skip = 0
            for s, t in pairs:
                if s not in match or t in match[s]:
                    continue
                if owner.get(t, s) != s:
                    n_skip += 1
                    continue
                match[s].append(t)
                owner[t] = s
                added.append((s, t))
                n_add += 1
            print(f"{os.path.basename(path)}: {n_add:,} pairs added, {n_skip:,} skipped (target owned by another S1)")
    return added


def main():
    """Read the ensemble output, apply the change lists and write both final TSVs."""
    ens_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    header, order, match = read_lists(os.path.join(ens_dir, "matching_results.tsv"))
    added = apply_steps(match)
    with open(os.path.join(out_dir, "matching_results.tsv"), "w", encoding="utf-8", newline="") as g:
        g.write(header if header.endswith("\n") else header + "\n")
        for s in order:
            g.write(f"{s}\t{','.join(match[s])}\n")
    extra = collections.defaultdict(list)
    for s, t in added:
        extra[s].append(t)
    n_app = 0
    with open(os.path.join(ens_dir, "candidate_pairs.tsv"), encoding="utf-8", newline="") as f, \
            open(os.path.join(out_dir, "candidate_pairs.tsv"), "w", encoding="utf-8", newline="") as g:
        g.write(f.readline())
        for line in f:
            s, _, ids = line.rstrip("\r\n").partition("\t")
            if s in extra:
                have = set(ids.split(",")) if ids else set()
                miss = [t for t in extra[s] if t not in have]
                n_app += len(miss)
                ids = ",".join(([ids] if ids else []) + miss)
            g.write(f"{s}\t{ids}\n")
    print(f"wrote {out_dir}: {len(order):,} rows; {n_app:,} added pairs appended to the candidate file")


if __name__ == "__main__":
    main()
