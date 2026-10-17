"""Synthetic labelled pairs for countries absent from train (France), built from their own test S1 records
(synthetic pairs from provided records are allowed). Each seed record gets noisy true copies and decoys:
  copies  formatting noise seen in train's true pairs: typos, leet digits, case, hyphens, accents, legal form
          added / dropped / reordered, a word dropped; street-type abbreviations (Rue -> R.), region or city
          dropped, parts reordered, "N°" / "(n)" numbers, blank address; the house number is kept (leaderboard
          probe 2a: French pairs whose number differs are ~18% true)
  decoys  ONE meaningful change + light noise: house number shifted, suffix added or changed (bis, B), street
          name swapped for another street of the same city, one name word swapped, legal form changed
Output parquet (xenc_cli layout): s1 (seed), t (pair id), thalf, label, kind, text_a, text_b."""
from __future__ import annotations

import random
import re
import unicodedata

import polars as pl

LEGAL = ["SARL", "SAS", "SASU", "EURL", "SA", "SCI", "EI", "SNC"]
LEGAL_VARIANTS = {"SARL": ["S.A.R.L.", "Sàrl", "sarl"], "SAS": ["S.A.S.", "sas"], "SA": ["S.A."], "EURL": ["E.U.R.L."]}
STREET_ABBR = {"Rue": ["R.", "R", "RUE"], "Avenue": ["Av", "Ave", "AV."], "Boulevard": ["Bd", "Bld", "BD"],
               "Allée": ["All", "Allee", "ALL."], "Place": ["Pl", "PL."], "Impasse": ["Imp"], "Chemin": ["Ch", "Che"],
               "Quai": ["Q."], "Route": ["Rte"], "Cours": ["Crs"]}
LEET = {"o": "0", "e": "3", "a": "4", "i": "1", "s": "5"}
NUM = re.compile(r"\b(\d+)(\s*(?:bis|ter|[A-Za-z])\b)?")


def strip_accents(s: str) -> str:
    """Remove combining accent marks (e.g. an accented e becomes e)."""
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def typo(w: str, r: random.Random) -> str:
    """Apply one random substitution, deletion, insertion or transposition inside a word of 4+ letters."""
    if len(w) < 4:
        return w
    i = r.randrange(1, len(w) - 1)
    op = r.choice("sdit")
    c = r.choice("abcdefghijklmnopqrstuvwxyz")
    return {"s": w[:i] + c + w[i + 1:], "d": w[:i] + w[i + 1:], "i": w[:i] + c + w[i:],
            "t": w[:i - 1] + w[i] + w[i - 1] + w[i + 1:]}[op]


def noisy_name(name: str, r: random.Random) -> str:
    """Noisy true-copy variant of a business name: typos, leet digits, legal form dropped / added /
    respelled, a word dropped or merged, and case or accent changes."""
    words = name.split()
    legal = [w for w in words if w.upper().strip(".") in LEGAL]
    core = [w for w in words if w not in legal]
    for _ in range(r.choice([1, 1, 2])):
        op = r.random()
        if op < 0.25 and core:
            k = r.randrange(len(core)); core[k] = typo(core[k], r)
        elif op < 0.35 and core:
            k = r.randrange(len(core)); w = core[k]; ch = [c for c in w.lower() if c in LEET]
            if ch:
                c = r.choice(ch); core[k] = w.replace(c, LEET[c], 1)
        elif op < 0.5:
            legal = [] if legal else [r.choice(LEGAL)]      # copies drop or add a legal form, never swap it
        elif op < 0.6 and legal:
            legal = [r.choice(LEGAL_VARIANTS.get(legal[0].upper().strip("."), [legal[0]]))]
        elif op < 0.7 and len(core) > 2:
            core.pop(r.randrange(1, len(core)))
        elif op < 0.8 and len(core) > 1:
            k = r.randrange(len(core) - 1); core[k:k + 2] = [core[k] + r.choice(["-", ""]) + core[k + 1]]
    out = " ".join((legal + core) if legal and r.random() < 0.3 else (core + legal))
    x = r.random()
    return out.upper() if x < 0.2 else out.lower() if x < 0.35 else strip_accents(out) if x < 0.45 else out


def noisy_addr(addr: str, r: random.Random, num_typo: float = 0.0) -> str:   # probe 2a: French copies keep the number
    """Noisy true-copy variant of an address: blanked, a part dropped or rotated, street types abbreviated,
    number formatting changed. The house number is kept unless num_typo > 0."""
    if r.random() < 0.08:
        return ""
    parts = [p.strip() for p in addr.split(",") if p.strip()]
    droppable = [k for k, part in enumerate(parts) if not NUM.search(part)]   # never the house-number part
    if len(parts) > 2 and droppable and r.random() < 0.4:
        parts.pop(r.choice(droppable))
    if len(parts) > 1 and r.random() < 0.3:
        k = r.randrange(1, len(parts)); parts = parts[k:] + parts[:k]
    s = ", ".join(parts)
    for full, abbr in STREET_ABBR.items():
        if full in s and r.random() < 0.5:
            s = s.replace(full, r.choice(abbr), 1)
    m = NUM.search(s)
    if m and r.random() < 0.15:
        s = s[:m.start()] + r.choice(["N°{}", "({})", "{},"]).format(m.group(1)) + s[m.start() + len(m.group(1)):]
    m = NUM.search(s)
    if m and r.random() < num_typo:
        n = m.group(1); k = r.randrange(len(n))
        s = s[:m.start()] + n[:k] + str((int(n[k]) + r.randint(1, 9)) % 10) + n[k + 1:] + s[m.start() + len(n):]
    x = r.random()
    return s.upper() if x < 0.25 else strip_accents(s) if x < 0.35 else s


def decoy(name: str, addr: str, pool: list[tuple[str, str]], r: random.Random) -> tuple[str, str, str]:
    """Make a decoy of a record by ONE meaningful change: shifted house number, changed suffix, street taken
    from another pool record, one name word swapped, or a different legal form.

    Returns: (name, address, kind of change)."""
    kind = r.choice(["num_shift", "suffix", "street_swap", "name_word", "legal"])
    m = NUM.search(addr)
    if kind == "num_shift" and m:
        n = int(m.group(1)); n2 = max(1, n + r.choice([-1, 1]) * r.randint(1, 10))
        addr = addr[:m.start(1)] + str(n2) + addr[m.end(1):]
    elif kind == "suffix" and m:
        cur = (m.group(2) or "").strip()
        new = r.choice([s for s in ["bis", "ter", "A", "B", "C", "D", "K", "J"] if s != cur])
        addr = addr[:m.start()] + m.group(1) + " " + new + addr[m.end():]
    elif kind == "street_swap" and m:
        other = r.choice([p for p in r.sample(pool, min(5, len(pool))) if p[1] != addr] or pool)[1]
        m2 = NUM.search(other)
        if m2:
            street = other[m2.end():].split(",")[0]
            seg_end = addr.find(",", m.end())
            addr = addr[:m.end()] + street + (addr[seg_end:] if seg_end >= 0 else "")
    elif kind == "name_word":
        words = name.split()
        other = [w for w in r.choice(pool)[0].split() if w.upper().strip(".") not in LEGAL]
        core = [k for k, w in enumerate(words) if w.upper().strip(".") not in LEGAL]
        if len(core) > 1 and other:
            words[r.choice(core)] = r.choice(other); name = " ".join(words)
    else:
        kind = "legal"
        words = [w for w in name.split() if w.upper().strip(".") not in LEGAL]
        name = " ".join(words + [r.choice(LEGAL)])
    return name, addr, kind


def build(s1: pl.DataFrame, n_seeds: int, seed: int = 7) -> pl.DataFrame:
    """s1: business_name, business_address of seed records (one country, test S1). 2 copies + 2 decoys each."""
    r = random.Random(seed)
    recs = [(a or "", b or "") for a, b in s1.select("business_name", "business_address").iter_rows()]
    seeds = r.sample(range(len(recs)), min(n_seeds, len(recs)))
    rows = []
    for i in seeds:
        name, addr = recs[i]
        a = f"{name} | {addr}"
        for _ in range(2):
            rows.append((i, 1, "copy", a, f"{noisy_name(name, r)} | {noisy_addr(addr, r)}"))
        for _ in range(2):
            n2, a2, kind = decoy(name, addr, recs, r)
            rows.append((i, 0, kind, a, f"{noisy_name(n2, r) if r.random() < 0.5 else n2} | {noisy_addr(a2, r, 0.0) if r.random() < 0.5 else a2}"))
    df = pl.DataFrame(rows, schema=["s1", "label", "kind", "text_a", "text_b"], orient="row")
    return df.with_row_index("t").with_columns(pl.col("t").cast(pl.UInt32), pl.col("s1").cast(pl.UInt32),
                                               pl.col("label").cast(pl.UInt8), thalf=(pl.col("t") % 2).cast(pl.Int8),
                                               fold=pl.lit(-1, pl.Int16))


if __name__ == "__main__":
    demo = pl.DataFrame({"business_name": ["Lille Santé SARL", "Bordeaux Union"],
                         "business_address": ["28 Rue Etienne Dolet, Lille, Hauts-de-France", "40 Rue Belfort, Bordeaux, Nouvelle-Aquitaine"]})
    out = build(demo, 2)
    assert out.height == 8 and out["label"].sum() == 4
    for row in out.iter_rows(named=True):
        print(row["label"], row["kind"], "|", row["text_b"])
