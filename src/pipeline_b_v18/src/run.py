"""Command-line entry point.

    python src/run.py prepare  --split all      # Stage 0: parse, normalize, cache
    python src/run.py folds                     # Stage 0: entity-disjoint folds (train)
    python src/run.py block    --split train    # Plan A candidates (+ recall report on train)
    python src/run.py features --split train    # Plan A pair features
    python src/run.py train-a                   # fit + tune Plan A on dev folds, report sealed fold
    python src/run.py submit-a --version v1     # test inference, submission files, validation

  Stage 2 (learned candidate ranker; downstream commands take --variant b):
    python src/run.py rank-train                # union sample from dev folds, 2 cross-fitted rankers
    python src/run.py rank --split train        # ranked candidates (cand_b) + recall@K report
    python src/run.py features --split train --variant b
    python src/run.py train-a --variant b
    python src/run.py rank --split test && python src/run.py features --split test --variant b
    python src/run.py submit-a --variant b --version v2

Paths come from --data-dir / --work-dir or the BER_DATA_DIR / BER_WORK_DIR variables.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import polars as pl

from config import from_args


def _log(msg: str) -> None:
    """Print a time-stamped log line and flush."""
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def cmd_prepare(cfg, args):
    """prepare: parse, normalize and cache the source files of the chosen split(s)."""
    from prepare import prepare_split
    for split in (["train", "test"] if args.split == "all" else [args.split]):
        prepare_split(cfg, split, _log)


def cmd_translit(cfg, args):
    """translit: build cross-script name skeletons for the chosen split(s)."""
    from prepare import prepare_skeletons
    for split in (["train", "test"] if args.split == "all" else [args.split]):
        prepare_skeletons(cfg, split, _log)


def cmd_indic_dict(cfg, args):
    """indic-dict: learn the Indic-to-Latin word dictionary and translate target names."""
    from indic_dict import run
    run(cfg, _log)


def cmd_fuzzy(cfg, args):
    """fuzzy: run the char-trigram TF-IDF blocking pass for the chosen split(s)."""
    from fuzzy import build
    for split in (["train", "test"] if args.split == "all" else [args.split]):
        build(cfg, split, _log)


def cmd_pj_train(cfg, args):
    """Pair judge: training pairs from this run's stage-1 train scores (after train-b), then the two
    cross-fitted models (GPU, BER_GPU_PYTHON)."""
    import pair_judge
    if not args.pj_tag:                       # a further model set reuses the same training pairs
        pair_judge.export_train(cfg, Path(args.pj_scores) if args.pj_scores else None, _log)
    pair_judge.train(cfg, _log, args.pj_tag)


def cmd_pj_score(cfg, args):
    """Pair judge: score the uncertain stage-1 band of a split (after train-b / score-b)."""
    import pair_judge
    for split in (["train", "test"] if args.split == "all" else [args.split]):
        if not args.pj_tag:                   # a further model set scores the same exported pairs
            pair_judge.export_score(cfg, split, _log)
        pair_judge.score(cfg, split, _log, args.pj_tag)


def cmd_folds(cfg, args):
    """folds: assign entity-disjoint folds to train S1 records and print fold statistics."""
    from prepare import load
    from splits import make_folds
    folds = make_folds(
        load(cfg, "train", "s1", ["s1", "entity_id", "country"]),
        load(cfg, "train", "gt"),
        load(cfg, "train", "tgt", ["t", "script"]),
        cfg.n_folds,
        cfg.seed,
    )
    folds.write_parquet(cfg.split_dir("train") / "folds.parquet")
    _log("fold sizes: " + json.dumps(dict(folds.group_by("fold").len().sort("fold").iter_rows())))
    check = folds.group_by("fold").agg(pl.col("n_true").mean().alias("mean_true"),
                                       (pl.col("n_true") == 0).mean().alias("singleton_rate")).sort("fold")
    print(check)


def cmd_block(cfg, args):
    """block: Plan A candidate generation (plus a recall report on train)."""
    from blocking import build_candidates
    build_candidates(cfg, args.split, _log)
    if args.split == "train":
        from report import candidate_report
        candidate_report(cfg, _log)


def cmd_rank_train(cfg, args):
    """rank-train: train the two cross-fitted candidate rankers."""
    from ranker import train_ranker
    train_ranker(cfg, _log)


def cmd_rank(cfg, args):
    """rank: write ranked candidates (cand_b) for a split (plus recall@K report on train)."""
    from ranker import rank_split, ranked_report
    rank_split(cfg, args.split, _log)
    if args.split == "train":
        ranked_report(cfg, _log)


def cmd_features(cfg, args):
    """features: build pair features of the chosen variant for a split."""
    from features import build_features
    build_features(cfg, args.split, _log, args.variant)


def cmd_train_a(cfg, args):
    """train-a: fit and tune the Plan A logistic matcher."""
    from model_a import train_and_tune
    train_and_tune(cfg, _log, args.variant)


def cmd_report_a(cfg, args):
    """report-a: sealed-fold strata report for the saved Plan A model and decision rule."""
    from model_a import decide, load_model, scores_path
    from report import strata_report
    model = load_model(cfg, args.variant)
    scores = pl.read_parquet(scores_path(cfg, args.variant), columns=["s1", "t", "p"])
    pred = decide(scores.filter(pl.col("p") >= min(0.15, model["tau"])), model["tau"], model["rho"], model["excl"])
    res = strata_report(cfg, pred, cfg.sealed_fold, _log)
    (cfg.split_dir("train") / f"report_strata_a_{args.variant}.json").write_text(json.dumps(res, indent=2, default=str))


def cmd_submit_a(cfg, args):
    """submit-a: predict test with Plan A and write/validate the submission files."""
    from model_a import predict_split
    from make_submission import make_submission
    pred, cand, model = predict_split(cfg, "test", _log, args.variant)
    make_submission(cfg, args.version, pred, cand, model, _log)


def cmd_train_b(cfg, args):
    """train-b: train, calibrate and tune the stage-1 LightGBM matcher (matcher_b)."""
    from matcher_b import train_and_tune
    train_and_tune(cfg, _log)


def cmd_submit_b(cfg, args):
    """submit-b: predict test with matcher B and write/validate the submission files."""
    from make_submission import make_submission
    from matcher_b import predict_split
    pred, cand, model = predict_split(cfg, "test", _log)
    make_submission(cfg, args.version, pred, cand, model, _log)


def cmd_stack_features(cfg, args):
    """stack-features: build the stacked-stage features of a split at the given level."""
    from stack import build_stack_features
    build_stack_features(cfg, args.split, _log, args.level)


def cmd_train_c(cfg, args):
    """train-c: train and tune the stacked stage."""
    from stack import train_and_tune
    train_and_tune(cfg, _log, args.level)


def cmd_submit_c(cfg, args):
    """submit-c: predict test with the stacked stage and write/validate the submission files."""
    from make_submission import make_submission
    from stack import predict_split
    pred, cand, model = predict_split(cfg, "test", _log, args.level)
    make_submission(cfg, args.version, pred, cand, model, _log)


def cmd_package(cfg, args):
    """package: build the final submission zip for a version."""
    from make_submission import final_package
    final_package(cfg, args.version, args.team, Path(args.doc), _log)


def cmd_score_b(cfg, args):
    """score-b: compute and cache calibrated matcher-B scores for a split."""
    from matcher_b import split_scores
    split_scores(cfg, args.split, _log)


def cmd_stress(cfg, args):
    """stress: distractor-injection stress test (through both stages for variant "c")."""
    from stress import injection, injection_stack
    if args.variant == "c":          # through both stages (v4+); variant "a" keeps the v3 stage-1 test
        injection_stack(cfg, log=_log, pool_folds=args.pool)
    else:
        injection(cfg, log=_log)


def main(argv=None) -> int:
    """Parse the command line, build the Config and run the chosen command; returns the exit code."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["prepare", "folds", "block", "features", "train-a", "report-a",
                                        "submit-a", "rank-train", "rank", "train-b", "submit-b", "stress",
                                        "score-b", "stack-features", "train-c", "submit-c", "translit",
                                        "package", "indic-dict", "fuzzy", "pj-train", "pj-score"])
    ap.add_argument("--split", default="train", choices=["train", "test", "all"])
    ap.add_argument("--variant", default="a", choices=["a", "b", "c"],
                    help="a: Plan A candidates; b: ranked candidates; c: ranked candidates + Stage 3 features")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--level", type=int, default=2, choices=[2, 3],
                    help="stacked stage: 2 = on stage-1 scores (v4-v6); 3 = on stage-2 scores")
    ap.add_argument("--pool", default="sealed", choices=["sealed", "dev"],
                    help="stress: evaluate sealed-fold (report) or dev-fold (tuning) entities")
    ap.add_argument("--shift-country", default=None,
                    help='submit-c: per-country decision shift, e.g. "France=0.3" (overrides config)')
    ap.add_argument("--team", default="team", help="package: zip name prefix (<team>_submission.zip)")
    ap.add_argument("--doc", default="Documentation_template.md", help="package: filled methodology document")
    ap.add_argument("--noisy-boost", type=float, default=None,
                    help="submit-c: odds multiplier for blank-address/domain targets (config.noisy_boost)")
    ap.add_argument("--fr-swap-boost", type=float, default=None,
                    help="submit-c: odds multiplier for French one-word-swap pairs (config.fr_swap_boost)")
    ap.add_argument("--fr-list-words", default=None,
                    help='submit-c: comma-separated French noise-copy words, e.g. "fils,associes,developpement"')
    ap.add_argument("--fr-list-adds", action="store_true", help="submit-c: apply the list-word floor to one-word additions too")
    ap.add_argument("--fr-vocab-words", default=None, help="submit-c: comma-separated French vocabulary words to penalise")
    ap.add_argument("--fr-vocab-mult", type=float, default=None, help="submit-c: odds multiplier for --fr-vocab-words swaps")
    ap.add_argument("--pj-tag", default="", help='pj-train/pj-score: model set ("" = model_A/B, "2" = second seed)')
    ap.add_argument("--pj-sets", default=None, help='stack-features/train-c/submit-c: pair-judge sets to average, e.g. ",2"')
    ap.add_argument("--drop-noisy-hidden", action="store_true",
                    help="stack-features: drop blank/domain copies of hidden S1s from the stage-2 train inputs")
    ap.add_argument("--pj-scores", default=None,
                    help="pj-train: take training pairs from an earlier run's stage-1 train scores (v13: v10's)")
    ap.add_argument("--data-dir")
    ap.add_argument("--work-dir")
    ap.add_argument("--n-jobs", type=int)
    args = ap.parse_args(argv)
    shift_country = None
    if args.shift_country:
        shift_country = {k.strip(): float(v) for k, v in (kv.split("=") for kv in args.shift_country.split(","))}
    cfg = from_args(args.data_dir, args.work_dir, n_jobs=args.n_jobs, decision_shift_country=shift_country,
                    noisy_boost=args.noisy_boost, fr_swap_boost=args.fr_swap_boost,
                    fr_list_words=tuple(w for w in (args.fr_list_words or "").split(",") if w) or None,
                    fr_vocab_words=tuple(w for w in (args.fr_vocab_words or "").split(",") if w) or None,
                    fr_vocab_mult=args.fr_vocab_mult, fr_list_adds=args.fr_list_adds or None,
                    pj_tags=tuple(args.pj_sets.split(",")) if args.pj_sets is not None else None,
                    stack_drop_noisy_hidden=args.drop_noisy_hidden or None)
    _log(f"config {cfg.fingerprint()} data={cfg.data_dir} work={cfg.work_dir} n_jobs={cfg.n_jobs}")
    handler = {
        "prepare": cmd_prepare, "folds": cmd_folds, "block": cmd_block, "features": cmd_features,
        "train-a": cmd_train_a, "report-a": cmd_report_a, "submit-a": cmd_submit_a,
        "rank-train": cmd_rank_train, "rank": cmd_rank, "train-b": cmd_train_b, "submit-b": cmd_submit_b,
        "stress": cmd_stress, "score-b": cmd_score_b, "stack-features": cmd_stack_features,
        "train-c": cmd_train_c, "submit-c": cmd_submit_c, "translit": cmd_translit, "package": cmd_package,
        "indic-dict": cmd_indic_dict, "fuzzy": cmd_fuzzy, "pj-train": cmd_pj_train, "pj-score": cmd_pj_score,
    }[args.command]
    t0 = time.time()
    handler(cfg, args)
    _log(f"{args.command} done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
