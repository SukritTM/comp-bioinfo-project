"""
foldseek_compare.py - Exhaustive FoldSeek ranking strategy comparison.

Groups:
  A. Individual columns
  B. All pairwise (primary + tiebreaker)
  C. Pairwise products (higher-is-better cols, including 1/rmsd and 1/evalue)
  D. Triple products and other combos

Usage:
    python foldseek_compare.py \
        --results data/foldseek-alignment-formatted.csv \
        --labels dgeb_euk_labels.json \
        --output_dir results_foldseek
"""

import argparse
import json
import itertools
import numpy as np
import pandas as pd
from pathlib import Path


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


def evaluate(df, score_col, higher_is_better, labels, k_values,
             secondary_col=None, secondary_higher=True):
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


def build_strategies(df):
    strategies = []

    # Base columns: (col_name, higher_is_better)
    # qlen/tlen excluded - they are protein properties, not alignment quality
    base = [
        ("alntmscore", True),
        ("rmsd",       False),
        ("prob",       True),
        ("evalue",     False),
        ("alnlen",     True),
        ("qcov",       True),
        ("tcov",       True),
    ]

    # A. Individual columns
    for col, higher in base:
        strategies.append((col, col, higher, None, True))

    # B. All pairwise tiebreakers
    for (c1, h1), (c2, h2) in itertools.permutations(base, 2):
        name = f"{c1} -> {c2}"
        strategies.append((name, c1, h1, c2, h2))

    # C. Pairwise products
    # Invert lower-is-better columns so all are higher-is-better
    df["inv_rmsd"]   = 1.0 / (df["rmsd"]   + 1e-9)
    df["inv_evalue"] = 1.0 / (df["evalue"] + 1e-9)

    hb = [
        ("alntmscore", "alntmscore"),
        ("prob",       "prob"),
        ("alnlen",     "alnlen"),
        ("qcov",       "qcov"),
        ("tcov",       "tcov"),
        ("inv_rmsd",   "1/rmsd"),
        ("inv_evalue", "1/evalue"),
    ]

    for (c1, n1), (c2, n2) in itertools.combinations(hb, 2):
        col_name = f"{c1}_x_{c2}"
        df[col_name] = df[c1] * df[c2]
        strategies.append((f"{n1} x {n2}", col_name, True, None, True))

    # D. Triple products
    triples = [
        ("alntmscore", "qcov",  "tcov",       "alntmscore x qcov x tcov"),
        ("alntmscore", "qcov",  "prob",        "alntmscore x qcov x prob"),
        ("alntmscore", "qcov",  "inv_evalue",  "alntmscore x qcov x 1/evalue"),
        ("alntmscore", "prob",  "qcov",        "alntmscore x prob x qcov"),
        ("prob",       "qcov",  "tcov",        "prob x qcov x tcov"),
        ("prob",       "qcov",  "inv_evalue",  "prob x qcov x 1/evalue"),
        ("alntmscore", "qcov",  "inv_rmsd",    "alntmscore x qcov x 1/rmsd"),
        ("alntmscore", "prob",  "inv_evalue",  "alntmscore x prob x 1/evalue"),
        ("prob",       "qcov",  "inv_rmsd",    "prob x qcov x 1/rmsd"),
    ]
    seen = set()
    for c1, c2, c3, name in triples:
        key = tuple(sorted([c1, c2, c3]))
        if key in seen:
            continue
        seen.add(key)
        col_name = f"{c1}_x_{c2}_x_{c3}"
        df[col_name] = df[c1] * df[c2] * df[c3]
        strategies.append((name, col_name, True, None, True))

    return strategies


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results",    required=True)
    p.add_argument("--labels",     required=True)
    p.add_argument("--k_values",   default="5,10,50")
    p.add_argument("--output_dir", default="results_foldseek")
    return p.parse_args()


def main():
    args     = parse_args()
    k_values = [int(k.strip()) for k in args.k_values.split(",")]

    print(f"Loading {args.results} ...")
    df = pd.read_csv(args.results, index_col=0)
    print(f"  {len(df)} rows, {df['query'].nunique()} queries, {df['target'].nunique()} targets")

    print(f"Loading {args.labels} ...")
    with open(args.labels) as f:
        labels = {k: set(v) for k, v in json.load(f).items()}
    print(f"  {len(labels)} labelled queries")

    strategies = build_strategies(df)
    print(f"\nTotal strategies to evaluate: {len(strategies)}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    for i, (display_name, score_col, higher, sec_col, sec_higher) in enumerate(strategies):
        print(f"  [{i+1}/{len(strategies)}] {display_name}", flush=True)
        metrics, pq_df = evaluate(df, score_col, higher, labels, k_values,
                                  secondary_col=sec_col, secondary_higher=sec_higher)
        all_metrics[display_name] = metrics
        safe = display_name.replace(" ", "_").replace("/", "per").replace(">", "").replace("-", "")
        pq_df.to_csv(output_dir / f"per_query_{safe}.csv", index=False)

    summary_path = output_dir / "foldseek_comparison.json"
    summary_path.write_text(json.dumps({"strategies": all_metrics}, indent=2))
    print(f"\nResults saved to {output_dir}/")

    # Sort and print by map@5
    ranked = sorted(all_metrics.items(), key=lambda x: x[1].get("map@5", 0), reverse=True)

    print()
    print("=" * 80)
    print("ALL STRATEGIES RANKED BY map@5")
    print("=" * 80)
    print(f"{'Rank':<6} {'Strategy':<42} {'map@5':>7}  {'ndcg@5':>7}  {'mrr@5':>7}  {'rec@5':>7}")
    print("-" * 80)
    for rank, (name, m) in enumerate(ranked, 1):
        print(f"{rank:<6} {name:<42} "
              f"{m.get('map@5', float('nan')):>7.4f}  "
              f"{m.get('ndcg@5', float('nan')):>7.4f}  "
              f"{m.get('mrr@5', float('nan')):>7.4f}  "
              f"{m.get('recall@5', float('nan')):>7.4f}")

    print("=" * 80)
    print("KNOWN BASELINES")
    print("  Centroids + FAISS        map@5 = 0.1503")
    print("  Freq vectors + FAISS     map@5 = 0.0295")
    print("  ESM-2 8M (best layer)    map@5 = 0.2147")
    print("=" * 80)

    print(f"\nTop 10 by map@5:")
    for rank, (name, m) in enumerate(ranked[:10], 1):
        print(f"  {rank:2}. {name:<45}  map@5={m.get('map@5'):.4f}")


if __name__ == "__main__":
    main()
