"""
foldseek_compare.py
-------------------
Evaluates multiple FoldSeek ranking strategies against DGEB ground-truth
labels and produces a comparison table.

Strategies evaluated:
  1. alntmscore        - raw TM-score of the aligned region (higher = better)
  2. prob              - FoldSeek probability of true homolog (higher = better)
  3. alntmscore_x_qcov - TM-score * query coverage (penalises partial alignments)
  4. qcov              - query coverage alone (higher = better)
  5. prob then qcov    - sort by prob first, use qcov as tiebreaker

Usage
-----
    python foldseek_compare.py \
        --results data/foldseek-alignment-formatted.csv \
        --labels  dgeb_euk_labels.json \
        --output_dir results_foldseek
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def dcg_at_k(rel, k):
    return sum(r / np.log2(i + 2) for i, r in enumerate(rel[:k]))

def ndcg_at_k(rel, k):
    idcg = dcg_at_k(sorted(rel, reverse=True), k)
    return 0.0 if idcg == 0 else dcg_at_k(rel, k) / idcg

def map_at_k(rel, k):
    hits, s = 0, 0.0
    for i, r in enumerate(rel[:k]):
        if r > 0:
            hits += 1
            s += hits / (i + 1)
    n_rel = sum(1 for r in rel if r > 0)
    return 0.0 if n_rel == 0 else s / min(n_rel, k)

def mrr_at_k(rel, k):
    for i, r in enumerate(rel[:k]):
        if r > 0:
            return 1.0 / (i + 1)
    return 0.0

def precision_at_k(rel, k):
    return sum(1 for r in rel[:k] if r > 0) / k

def recall_at_k(rel, k, n_rel):
    return 0.0 if n_rel == 0 else sum(1 for r in rel[:k] if r > 0) / n_rel


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(df, score_col, higher_is_better, labels, k_values,
             secondary_col=None, secondary_higher=True):
    """
    Evaluate a ranking strategy.

    Parameters
    ----------
    score_col          : primary column to sort by
    higher_is_better   : whether higher score = better rank
    secondary_col      : optional tiebreaker column
    secondary_higher   : whether higher tiebreaker = better rank
    """
    max_k = max(k_values)
    rows = []

    for qid, group in df.groupby("query"):
        if qid not in labels:
            continue
        relevant = labels[qid]
        n_rel = len(relevant)
        if n_rel == 0:
            continue

        if secondary_col:
            ranked = group.sort_values(
                [score_col, secondary_col],
                ascending=[not higher_is_better, not secondary_higher]
            )
        else:
            ranked = group.sort_values(score_col, ascending=not higher_is_better)

        retrieved = ranked["target"].tolist()[:max_k]
        rel = [1.0 if rid in relevant else 0.0 for rid in retrieved]
        rel += [0.0] * (max_k - len(rel))

        row = {"protein_id": qid, "n_relevant": n_rel}
        for k in k_values:
            row[f"ndcg@{k}"]      = ndcg_at_k(rel, k)
            row[f"mrr@{k}"]       = mrr_at_k(rel, k)
            row[f"map@{k}"]       = map_at_k(rel, k)
            row[f"precision@{k}"] = precision_at_k(rel, k)
            row[f"recall@{k}"]    = recall_at_k(rel, k, n_rel)
        rows.append(row)

    result_df = pd.DataFrame(rows)
    metrics = {
        col: round(float(result_df[col].mean()), 6)
        for col in result_df.columns
        if col not in ("protein_id", "n_relevant")
    }
    return metrics, result_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare FoldSeek ranking strategies against DGEB labels."
    )
    p.add_argument("--results",    required=True,
                   help="FoldSeek alignment CSV")
    p.add_argument("--labels",     required=True,
                   help="DGEB ground-truth JSON (e.g. dgeb_euk_labels.json)")
    p.add_argument("--k_values",   default="5,10,50",
                   help="Comma-separated k values (default: 5,10,50)")
    p.add_argument("--output_dir", default="results_foldseek",
                   help="Directory for output files (default: results_foldseek)")
    return p.parse_args()


def main():
    args     = parse_args()
    k_values = [int(k.strip()) for k in args.k_values.split(",")]

    # 1. Load data
    print(f"Loading FoldSeek results from {args.results} ...")
    df = pd.read_csv(args.results, index_col=0)
    print(f"  {len(df)} alignment rows, {df['query'].nunique()} queries, "
          f"{df['target'].nunique()} targets")

    print(f"Loading labels from {args.labels} ...")
    with open(args.labels) as f:
        labels = {k: set(v) for k, v in json.load(f).items()}
    print(f"  {len(labels)} labelled queries")

    # 2. Add composite score
    df["alntmscore_x_qcov"] = df["alntmscore"] * df["qcov"]

    # 3. Define strategies:
    # (score_col, higher_is_better, display_name, secondary_col, secondary_higher)
    strategies = [
        ("alntmscore",        True,  "alntmscore",        None,   True),
        ("prob",              True,  "prob",               None,   True),
        ("alntmscore_x_qcov", True,  "alntmscore x qcov", None,   True),
        ("qcov",              True,  "qcov",               None,   True),
        ("prob",              True,  "prob then qcov",     "qcov", True),
    ]

    # 4. Evaluate each strategy
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    for score_col, higher, display_name, sec_col, sec_higher in strategies:
        print(f"Evaluating: {display_name} ...", flush=True)
        metrics, pq_df = evaluate(df, score_col, higher, labels, k_values,
                                  secondary_col=sec_col, secondary_higher=sec_higher)
        all_metrics[display_name] = metrics
        safe_name = display_name.replace(" ", "_")
        pq_df.to_csv(output_dir / f"per_query_{safe_name}.csv", index=False)

    # 5. Save summary JSON
    summary_path = output_dir / "foldseek_comparison.json"
    summary_path.write_text(json.dumps({"strategies": all_metrics}, indent=2))
    print(f"\nResults saved to {output_dir}/")

    # 6. Print comparison table
    col_w    = 20
    n_strats = len(strategies)
    sep      = "=" * (20 + (col_w + 2) * n_strats)

    print()
    print(sep)
    print("FOLDSEEK RANKING STRATEGY COMPARISON")
    print(sep)
    print(f"{'Metric':<20}", end="")
    for _, _, display_name, _, _ in strategies:
        print(f"  {display_name:<{col_w}}", end="")
    print()
    print("-" * (20 + (col_w + 2) * n_strats))

    for k in k_values:
        for metric in ["map", "ndcg", "mrr", "precision", "recall"]:
            key = f"{metric}@{k}"
            print(f"{key:<20}", end="")
            for _, _, display_name, _, _ in strategies:
                val = all_metrics[display_name].get(key, float("nan"))
                print(f"  {val:<{col_w}.4f}", end="")
            print()
        print()

    print(sep)
    print("KNOWN BASELINES")
    print(f"  Centroids + FAISS        map@5 = 0.1503")
    print(f"  Freq vectors + FAISS     map@5 = 0.0295")
    print(f"  ESM-2 8M (best layer)    map@5 = 0.2147")
    print(sep)


if __name__ == "__main__":
    main()
