"""Write the two submission files, verify every rule locally, run the official validator.

IDs are only ever copied from test records that the pipeline retrieved; nothing is
generated. Violations raise instead of warning.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import polars as pl

from config import Config
from data_io import CAND_HEADER, MATCH_HEADER, write_id_lists
from prepare import load


def _to_ids(pairs: pl.DataFrame, s1_ids: pl.Series, t_ids: pl.Series) -> pl.DataFrame:
    """Map integer (s1, t) index pairs to string entity ids (s1_id, t_id)."""
    return pl.DataFrame({"s1_id": s1_ids.gather(pairs["s1"]), "t_id": t_ids.gather(pairs["t"])})


def _read_back(path: Path, header: tuple[str, str], s1_order: list[str], valid: set[str]) -> dict:
    """Re-read a written submission file and assert its format rules.

    Checks LF line endings, trailing newline, header, row order, no duplicate ids and that every
    id is a valid S2-/S3- test id. Returns {s1_id: set of listed ids}.
    """
    text = path.read_bytes().decode("utf-8")
    if "\r" in text:
        raise AssertionError(f"{path.name}: contains carriage returns")
    lines = text.split("\n")
    if lines[-1] != "":
        raise AssertionError(f"{path.name}: must end with a newline")
    if lines[0] != "\t".join(header):
        raise AssertionError(f"{path.name}: header {lines[0]!r}")
    rows = lines[1:-1]
    if len(rows) != len(s1_order):
        raise AssertionError(f"{path.name}: {len(rows)} rows, expected {len(s1_order)}")
    mapping = {}
    for row, expected in zip(rows, s1_order):
        s1, tab, rest = row.partition("\t")
        if not tab or s1 != expected:
            raise AssertionError(f"{path.name}: row for {expected} malformed or out of order")
        ids = rest.split(",") if rest else []
        if len(ids) != len(set(ids)):
            raise AssertionError(f"{path.name}: duplicate id in list of {s1}")
        for i in ids:
            if i[:3] not in ("S2-", "S3-") or i not in valid:
                raise AssertionError(f"{path.name}: invalid id {i} for {s1}")
        mapping[s1] = set(ids)
    return mapping


def _src_hashes() -> dict:
    """Short SHA-256 of every .py file in this source folder, recorded in the manifest."""
    src = Path(__file__).resolve().parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:12] for p in sorted(src.glob("*.py"))}


def _code_zip(out_dir: Path, version: str) -> Path:
    """Code zip uploaded with each leaderboard submission, in the official package layout:
    code/business_entity_resolution/{src/, tests/, README.md, requirements.txt}."""
    pkg = Path(__file__).resolve().parents[1]
    root = Path("code") / "business_entity_resolution"
    path = out_dir / f"code_{version}.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted((pkg / "src").glob("*.py")):
            z.write(p, root / "src" / p.name)
        for name in ("README.md", "requirements.txt"):
            z.write(pkg / name, root / name)
        for p in sorted((pkg / "tests").glob("*.py")):
            z.write(p, root / "tests" / p.name)
    return path


def _upload_copy(out_dir: Path, version: str, log) -> None:
    """Copy the files needed for a leaderboard upload to $BER_UPLOAD_DIR/<version>, if set."""
    target = os.environ.get("BER_UPLOAD_DIR")
    if not target:
        return
    dest = Path(target) / version
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("matching_results.tsv", f"code_{version}.zip", "manifest.json"):
        shutil.copy2(out_dir / name, dest / name)
        if hashlib.sha256((out_dir / name).read_bytes()).digest() != hashlib.sha256((dest / name).read_bytes()).digest():
            raise AssertionError(f"copy of {name} differs from the original")
    log(f"[submit] upload files copied to {dest}")


def final_package(cfg: Config, version: str, team: str, doc: Path, log=print) -> Path:
    """The single zip of the README's "Final Submission Package": output/ with both TSVs of
    `version`, code/business_entity_resolution/ and the filled Documentation_template.md.

    The code comes from the version's own code zip, checked file by file against the source
    hashes in its manifest, so the zipped pipeline is the one that produced the outputs.
    """
    src_dir = cfg.work_dir / "submissions" / version
    manifest = json.loads((src_dir / "manifest.json").read_text())
    if not doc.exists():
        raise FileNotFoundError(doc)
    out = cfg.work_dir / "final" / f"{team}_submission.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with zipfile.ZipFile(src_dir / f"code_{version}.zip") as code:
        entries = {n: code.read(n) for n in code.namelist()}
    src = {Path(n).name: hashlib.sha256(b).hexdigest()[:12] for n, b in entries.items() if "/src/" in n}
    if src != manifest["source_sha256"]:
        raise AssertionError(f"code_{version}.zip src differs from the code that produced {version}")
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in ("matching_results.tsv", "candidate_pairs.tsv"):
            z.write(src_dir / name, Path("output") / name)
        for name, data in entries.items():
            z.writestr(name, data)
        z.write(doc, "Documentation_template.md")
    with zipfile.ZipFile(out) as z:
        bad = z.testzip()
        if bad is not None:
            raise AssertionError(f"{out.name}: CRC check failed for {bad}")
        names = z.namelist()
    log(f"[package] {out} ({out.stat().st_size / 1e6:.0f} MB, {len(names)} files, {time.time() - t0:.0f}s)")
    target = os.environ.get("BER_UPLOAD_DIR")
    if target:
        dest = Path(target) / "final"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out, dest / out.name)
        if hashlib.sha256((dest / out.name).read_bytes()).digest() != hashlib.sha256(out.read_bytes()).digest():
            raise AssertionError("copy of the final package differs from the original")
        log(f"[package] copied to {dest / out.name}")
    return out


def make_submission(cfg: Config, version: str, pred: pl.DataFrame, cand: pl.DataFrame, model: dict, log=print) -> Path:
    """Write and verify matching_results.tsv and candidate_pairs.tsv for the test split.

    Args:
        pred: predicted (s1, t) index pairs; cand: candidate (s1, t) index pairs.
        model: model metadata, partly copied into manifest.json.
    Returns:
        The submission directory <work_dir>/submissions/<version> (also holds the code zip).
    """
    t0 = time.time()
    out_dir = cfg.work_dir / "submissions" / version
    out_dir.mkdir(parents=True, exist_ok=True)
    s1 = load(cfg, "test", "s1", ["s1", "entity_id", "country"]).sort("s1")
    tg = load(cfg, "test", "tgt", ["t", "entity_id"]).sort("t")

    match_path = out_dir / "matching_results.tsv"
    cand_path = out_dir / "candidate_pairs.tsv"
    write_id_lists(match_path, MATCH_HEADER, s1["entity_id"], _to_ids(pred, s1["entity_id"], tg["entity_id"]))
    write_id_lists(cand_path, CAND_HEADER, s1["entity_id"], _to_ids(cand, s1["entity_id"], tg["entity_id"]))

    order = s1["entity_id"].to_list()
    valid = set(tg["entity_id"].to_list())
    matches = _read_back(match_path, MATCH_HEADER, order, valid)
    cands = _read_back(cand_path, CAND_HEADER, order, valid)
    outside = [s for s, ids in matches.items() if not ids <= cands[s]]
    if outside:
        raise AssertionError(f"{len(outside)} S1 rows have matches outside their candidates, e.g. {outside[:3]}")
    log(f"[submit] local checks passed ({time.time() - t0:.0f}s)")

    validator = cfg.data_dir.parent / "utils" / "validate_submission.py"
    proc = subprocess.run(
        [sys.executable, str(validator), "--matching", str(match_path), "--candidate", str(cand_path),
         "--test-dir", str(cfg.data_dir / "test"), "--check-ids"],
        capture_output=True, text=True, encoding="utf-8",
    )
    log("[submit] validator output:\n" + proc.stdout.strip())
    if proc.returncode != 0 or "PASS" not in proc.stdout:
        raise AssertionError("official validator did not pass:\n" + proc.stdout + proc.stderr)

    per_country = (
        s1.with_columns(n=pl.Series([len(matches[e]) for e in order]),
                        c=pl.Series([len(cands[e]) for e in order]))
        .group_by("country")
        .agg(rows=pl.len(), empty=(pl.col("n") == 0).mean(), mean_len=pl.col("n").mean(),
             mean_cands=pl.col("c").mean())
        .sort("country")
    )
    log(f"[submit] test monitors by country:\n{per_country}")
    manifest = {
        "version": version,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": cfg.fingerprint(),
        "source_sha256": _src_hashes(),
        "model": {k: v for k, v in model.items() if k in ("tau", "rho", "excl", "max_matches", "train_rows", "policy")},
        "sealed_fold": model.get("sealed", {}).get("overall"),
        "rows": len(order),
        "matched_pairs": sum(len(v) for v in matches.values()),
        "candidate_pairs": sum(len(v) for v in cands.values()),
        "test_monitors": per_country.to_dicts(),
        "validator": proc.stdout.strip().splitlines()[-1],
        "code_zip": _code_zip(out_dir, version).name,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    log(f"[submit] {version} written to {out_dir}")
    _upload_copy(out_dir, version, log)
    return out_dir
