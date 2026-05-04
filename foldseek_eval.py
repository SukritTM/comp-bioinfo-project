"""
foldseek_eval.py
----------------
Evaluates FoldSeek structural similarity results against DGEB ground-truth
labels (euk_retrieval task) and reports the same metrics as retrieval.py.

FoldSeek output format (default):
    query  target  fident  alnlen  mismatch  gapopen  qstart  qend  tstart  tend  evalue  bits

Or with TM-score:
    foldseek easy-search ... --format-output "query,target,alntmscore"

UniProt IDs are extracted from AlphaFold filenames automatically:
    AF-{UNIPROT_ID}-F1-model_v6.cif  ->  UNIPROT_ID
    AF-{UNIPROT_ID}-F1-model_v6      ->  UNIPROT_ID  (no extension)

Usage
-----
    # Using default FoldSeek output (ranked by bitscore, higher = better):
    python foldseek_eval.py --results foldseek_results.tsv --labels dgeb_euk_labels.json

    # Using TM-score output:
    python foldseek_eval.py --results foldseek_results.tsv --labels dgeb_euk_labels.json \\
        --score_col alntmscore --higher_is_better

    # Using e-value (lower = better):
    python foldseek_eval.py --results foldseek_results.tsv --labels dgeb_euk_labels.json \\
        --score_col evalue --no-higher_is_better

    # Custom column names (if you used --format-output):
    python foldseek_eval.py --results foldseek_results.tsv --labels dgeb_euk_labels.json \\
        --score_col alntmscore --higher_is_better --query_col query --target_col target

Output (written to --output_dir, default: results_foldseek/)
-------------------------------------------------------------
    summary_metrics.json
    per_query_scores.csv
    recall_curve.csv
    score_distribution_map@k.csv
    retrieval_log.txt
    run_manifest.jsonl

Dependencies
------------
    pip install numpy pandas tqdm
"""

import argparse
import datetime
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# UniProt ID extraction
# ---------------------------------------------------------------------------

def extract_uniprot_id(name: str) -> str:
    """
    Extract UniProt ID from an AlphaFold filename or bare ID.

    Examples
    --------
    AF-E2RU81-F1-model_v6.cif  ->  E2RU81
    AF-E2RU81-F1-model_v6      ->  E2RU81
    E2RU81                     ->  E2RU81
    """
    # Strip path and extension
    stem = Path(name).stem
    # AF-{ID}-F1-model_v6 pattern
    if stem.startswith("AF-"):
        parts = stem.split("-")
        if len(parts) >= 3:
            return parts[1]
    return stem


# ---------------------------------------------------------------------------
# Load FoldSeek results
# ---------------------------------------------------------------------------

def load_foldseek_results(results_path: Path, query_col: str, target_col: str,
                           score_col: str, higher_is_better: bool,
                           logger) -> dict:
    """
    Load FoldSeek TSV output and return a dict:
        { query_uniprot_id: [(target_uniprot_id, score), ...] }
    Results are pre-sorted best-first per query.
    """
    logger.info(f"Loading FoldSeek results from {results_path}")

    # Try to detect if there's a header line
    with open(results_path) as f:
        first_line = f.readline().strip()

    has_header = not first_line.split("\t")[0].startswith("AF-") and \
                 not first_line.split("\t")[0][0].isupper() or \
                 first_line.startswith(query_col)

    # Default FoldSeek columns (no header)
    default_cols = ["query", "target", "fident", "alnlen", "mismatch",
                    "gapopen", "qstart", "qend", "tstart", "tend", "evalue", "bits"]

    try:
        if has_header:
            df = pd.read_csv(results_path, sep="\t")
        else:
            df = pd.read_csv(results_path, sep="\t", header=None)
            # Assign default column names if column count matches
            if len(df.columns) == len(default_cols):
                df.columns = default_cols
            else:
                # Generic column names
                df.columns = [f"col{i}" for i in range(len(df.columns))]
                # Try to map the first two as query/target
                df = df.rename(columns={"col0": "query", "col1": "target"})
                if len(df.columns) > 2:
                    df = df.rename(columns={"col2": df.columns[2]})
    except Exception as e:
        logger.error(f"Failed to read FoldSeek results: {e}")
        sys.exit(1)

    logger.info(f"Loaded {len(df)} alignment rows")
    logger.info(f"Columns: {list(df.columns)}")

    if query_col not in df.columns:
        logger.error(f"Query column '{query_col}' not found. Available: {list(df.columns)}")
        sys.exit(1)
    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found. Available: {list(df.columns)}")
        sys.exit(1)
    if score_col not in df.columns:
        logger.error(f"Score column '{score_col}' not found. Available: {list(df.columns)}")
        sys.exit(1)

    # Extract UniProt IDs
    df["query_id"]  = df[query_col].apply(extract_uniprot_id)
    df["target_id"] = df[target_col].apply(extract_uniprot_id)
    df["score"]     = pd.to_numeric(df[score_col], errors="coerce")
    df = df.dropna(subset=["score"])

    # Remove self-hits
    df = df[df["query_id"] != df["target_id"]]

    # Sort best-first
    df = df.sort_values("score", ascending=not higher_is_better)

    # Build per-query ranked list
    ranked = {}
    for query_id, group in df.groupby("query_id"):
        ranked[query_id] = list(zip(group["target_id"], group["score"]))

    n_queries  = len(ranked)
    avg_hits   = np.mean([len(v) for v in ranked.values()]) if ranked else 0
    logger.info(f"Unique query proteins : {n_queries}")
    logger.info(f"Avg hits per query    : {avg_hits:.1f}")

    return ranked


# ---------------------------------------------------------------------------
# Ground-truth labels
# ---------------------------------------------------------------------------

def load_labels(label_path: Path, logger) -> dict:
    logger.info(f"Loading ground-truth labels from {label_path}")
    with open(label_path) as f:
        raw = json.load(f)
    labels = {k: set(v) for k, v in raw.items()}
    logger.info(f"Labels loaded for {len(labels)} query proteins")
    return labels


# ---------------------------------------------------------------------------
# Metric functions (same as retrieval.py)
# ---------------------------------------------------------------------------

def dcg_at_k(rel, k):
    return sum(r / np.log2(i + 2) for i, r in enumerate(rel[:k]))

def ndcg_at_k(rel, k):
    idcg = dcg_at_k(sorted(rel, reverse=True), k)
    return 0.0 if idcg == 0 else dcg_at_k(rel, k) / idcg

def mrr_at_k(rel, k):
    for i, r in enumerate(rel[:k]):
        if r > 0:
            return 1.0 / (i + 1)
    return 0.0

def map_at_k(rel, k):
    hits, s = 0, 0.0
    for i, r in enumerate(rel[:k]):
        if r > 0:
            hits += 1
            s += hits / (i + 1)
    n_rel = sum(1 for r in rel if r > 0)
    return 0.0 if n_rel == 0 else s / min(n_rel, k)

def precision_at_k(rel, k):
    return sum(1 for r in rel[:k] if r > 0) / k

def recall_at_k(rel, k, n_rel):
    return 0.0 if n_rel == 0 else sum(1 for r in rel[:k] if r > 0) / n_rel


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(ranked: dict, labels: dict, k_values: list, logger):
    max_k      = max(k_values)
    query_ids  = [qid for qid in ranked if qid in labels]

    if not query_ids:
        logger.error("No overlap between FoldSeek queries and label keys. "
                     "Check that UniProt IDs are being extracted correctly.")
        sys.exit(1)

    logger.info(f"Evaluating {len(query_ids)} queries with labels, max_k={max_k}")

    rows         = []
    recall_accum = np.zeros(max_k, dtype=np.float64)
    n_queries    = 0
    start        = time.time()

    for qid in tqdm(query_ids, desc="Evaluating"):
        relevant = labels[qid]
        n_rel    = len(relevant)
        if n_rel == 0:
            continue

        hits = ranked.get(qid, [])
        if len(hits) < max_k:
            logger.warning(f"{qid}: only {len(hits)} hits returned (need {max_k} for k={max_k})")

        retrieved_ids = [tid for tid, _ in hits[:max_k]]
        rel = [1.0 if rid in relevant else 0.0 for rid in retrieved_ids]

        # Pad with zeros if fewer hits than max_k
        rel += [0.0] * (max_k - len(rel))

        row = {"protein_id": qid, "n_relevant": n_rel, "n_retrieved": len(hits)}
        for k in k_values:
            row[f"ndcg@{k}"]      = ndcg_at_k(rel, k)
            row[f"mrr@{k}"]       = mrr_at_k(rel, k)
            row[f"map@{k}"]       = map_at_k(rel, k)
            row[f"precision@{k}"] = precision_at_k(rel, k)
            row[f"recall@{k}"]    = recall_at_k(rel, k, n_rel)

        for kk in range(1, max_k + 1):
            recall_accum[kk - 1] += recall_at_k(rel, kk, n_rel)

        rows.append(row)
        n_queries += 1

    elapsed = time.time() - start
    logger.info(f"Evaluation complete in {elapsed:.1f}s")

    per_query_df = pd.DataFrame(rows)

    summary = {"n_queries": n_queries, "k_values": k_values, "metrics": {}}
    for col in [c for c in per_query_df.columns
                if c not in ("protein_id", "n_relevant", "n_retrieved")]:
        summary["metrics"][col] = round(float(per_query_df[col].mean()), 6)

    recall_curve = pd.DataFrame({
        "k":      list(range(1, max_k + 1)),
        "recall": (recall_accum / max(n_queries, 1)).tolist(),
    })

    return per_query_df, recall_curve, summary


def score_distribution(per_query_df, k):
    col = f"map@{k}"
    if col not in per_query_df.columns:
        return pd.DataFrame()
    counts, edges = np.histogram(per_query_df[col].dropna(), bins=20, range=(0.0, 1.0))
    return pd.DataFrame({
        "bin_left":  edges[:-1].round(3),
        "bin_right": edges[1:].round(3),
        "count":     counts,
    })


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("foldseek_eval")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate FoldSeek results against DGEB euk_retrieval labels."
    )
    p.add_argument("--results",   required=True,
                   help="Path to FoldSeek output TSV file")
    p.add_argument("--labels",    required=True,
                   help="Path to DGEB ground-truth JSON (e.g. dgeb_euk_labels.json)")
    p.add_argument("--score_col", default="bits",
                   help="Column name to use as similarity score (default: bits). "
                        "Use 'alntmscore' for TM-score output, 'evalue' for e-value.")
    p.add_argument("--query_col",  default="query",
                   help="Column name for query protein (default: query)")
    p.add_argument("--target_col", default="target",
                   help="Column name for target protein (default: target)")
    p.add_argument("--higher_is_better", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Whether higher score = better match (default: True). "
                        "Use --no-higher_is_better for e-value.")
    p.add_argument("--k_values",  default="5,10,50",
                   help="Comma-separated k values for metrics (default: 5,10,50)")
    p.add_argument("--output_dir", default="results_foldseek",
                   help="Directory for output files (default: results_foldseek)")
    return p.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger   = setup_logger(output_dir / "retrieval_log.txt")
    k_values = [int(k.strip()) for k in args.k_values.split(",")]

    logger.info("=" * 60)
    logger.info("FoldSeek Structural Retrieval Evaluation")
    logger.info("=" * 60)
    logger.info(f"Results file      : {args.results}")
    logger.info(f"Labels file       : {args.labels}")
    logger.info(f"Score column      : {args.score_col}")
    logger.info(f"Higher is better  : {args.higher_is_better}")
    logger.info(f"k values          : {k_values}")
    logger.info(f"Primary metric    : map@5 (DGEB standard)")

    # 1. Load FoldSeek results
    ranked = load_foldseek_results(
        Path(args.results),
        query_col=args.query_col,
        target_col=args.target_col,
        score_col=args.score_col,
        higher_is_better=args.higher_is_better,
        logger=logger,
    )

    # 2. Load labels
    labels = load_labels(Path(args.labels), logger)

    # 3. Evaluate
    per_query_df, recall_curve, summary = evaluate(ranked, labels, k_values, logger)

    # 4. Save outputs
    (output_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2))
    per_query_df.to_csv(output_dir / "per_query_scores.csv", index=False)
    recall_curve.to_csv(output_dir / "recall_curve.csv", index=False)
    for k in k_values:
        score_distribution(per_query_df, k).to_csv(
            output_dir / f"score_distribution_map@{k}.csv", index=False
        )

    # Append to run manifest
    manifest_path = output_dir / "run_manifest.jsonl"
    manifest_entry = {
        "timestamp":      datetime.datetime.now().isoformat(timespec="seconds"),
        "results_file":   args.results,
        "score_col":      args.score_col,
        "higher_is_better": args.higher_is_better,
        "n_queries":      summary["n_queries"],
        "k_values":       k_values,
        "metrics":        summary["metrics"],
    }
    with open(manifest_path, "a") as f:
        f.write(json.dumps(manifest_entry) + "\n")
    logger.info(f"Run appended to manifest -> {manifest_path}")
    logger.info(f"All outputs saved to: {output_dir}/")

    # 5. Print summary table
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    header = f"{'Metric':<20}" + "".join(f"  k={k:<8}" for k in k_values)
    logger.info(header)
    logger.info("-" * len(header))
    for metric in ["map", "ndcg", "mrr", "precision", "recall"]:
        row_str = f"{metric:<20}"
        for k in k_values:
            val = summary["metrics"].get(f"{metric}@{k}", float("nan"))
            row_str += f"  {val:.4f}    "
        logger.info(row_str)
    logger.info("=" * 60)
    logger.info(f"  [*] Primary DGEB metric: map@5 = {summary['metrics'].get('map@5', float('nan')):.4f}")
    logger.info("=" * 60)

    # 6. Print comparison against known baselines
    logger.info("\nComparison against other methods:")
    logger.info(f"  FoldSeek ({args.score_col})  map@5 = {summary['metrics'].get('map@5', float('nan')):.4f}")
    logger.info(f"  Centroids + FAISS        map@5 = 0.1503")
    logger.info(f"  ESM-2 8M (best layer)    map@5 = 0.2147")


if __name__ == "__main__":
    main()
