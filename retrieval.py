"""
retrieval.py
------------
Loads *_centroids.csv files (from AlphaFold structures downloaded for DGEB),
builds pairwise-distance tensors on the fly, runs FAISS cosine-similarity
retrieval, and evaluates against DGEB ground-truth labels.

Protein IDs are extracted from AlphaFold filenames:
    AF-{UNIPROT_ID}-F1-model_v6_centroids.csv  ->  UNIPROT_ID

Usage
-----
    # Test pipeline with synthetic labels:
    python retrieval.py --centroids data/secondary_centroids --demo

    # Pull DGEB labels directly from HuggingFace and run:
    python retrieval.py --centroids data/secondary_centroids --dgeb

    # Run with a pre-saved labels JSON:
    python retrieval.py --centroids data/secondary_centroids --labels dgeb_euk_labels.json

Output (written to results/)
------------------------------
    summary_metrics.json          — aggregate metrics table
    per_query_scores.csv          — per-protein scores at every k
    recall_curve.csv              — recall@k for k=1..max_k  (for recall curve plot)
    score_distribution_map@k.csv  — histogram data for MAP score distribution
    retrieval_log.txt             — full human-readable log

Dependencies
------------
    pip install faiss-cpu numpy pandas tqdm datasets
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Conf-type vocabulary
# ---------------------------------------------------------------------------

CONF_TYPE_VOCAB = [
    "HELX_RH_AL_P",
    "HELX_RH_3T_P",
    "HELX_RH_PI_P",
    "HELX_LH_AL_P",
    "HELX_LH_PP_P",
    "TURN_TY1_P",
    "TURN_TY1P_P",
    "TURN_TY2_P",
    "TURN_TY2P_P",
    "TURN_TY3_P",
    "STRN",
    "BEND",
    "OTHER",
]
CONF_TYPE_TO_IDX = {ct: i for i, ct in enumerate(CONF_TYPE_VOCAB)}


def conf_type_idx(ct: str) -> int:
    return CONF_TYPE_TO_IDX.get(ct, CONF_TYPE_TO_IDX["OTHER"])


# ---------------------------------------------------------------------------
# CSV → tensor (on the fly)
# ---------------------------------------------------------------------------

def csv_to_vector(csv_path: Path, max_elements: int) -> tuple[np.ndarray, dict]:
    """
    Read one *_centroids.csv and return a flat upper-triangle vector + metadata.

    Tensor channels (before flattening):
      0 - pairwise Euclidean distance between centroids (Angstroms)
      1 - pairwise sequence separation  |beg_seq_i - beg_seq_j|
      2 - same conf_type indicator  (1 if both elements share the same type)

    Returns
    -------
    vector : np.float32 of shape (3 * n_pairs,)  where n_pairs = MAX*(MAX-1)/2
    meta   : dict with protein_id, n_elements, truncated flag
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["mean_x", "mean_y", "mean_z"]).reset_index(drop=True)

    n      = len(df)
    n_used = min(n, max_elements)
    sub    = df.iloc[:n_used].copy()

    # Extract UniProt ID from AlphaFold filename
    # AF-{UNIPROT_ID}-F1-model_v6_centroids → UNIPROT_ID
    stem       = csv_path.stem.replace("_centroids", "")
    parts      = stem.split("-")
    uniprot_id = parts[1] if len(parts) >= 3 and stem.startswith("AF-") else stem

    meta = {
        "protein_id":  uniprot_id,
        "af_filename": stem,
        "n_elements":  n,
        "n_used":      n_used,
        "truncated":   n > max_elements,
    }

    # Pad to max_elements rows with zeros so all tensors have the same shape
    if n_used < max_elements:
        pad = pd.DataFrame(
            {
                "mean_x":       [0.0] * (max_elements - n_used),
                "mean_y":       [0.0] * (max_elements - n_used),
                "mean_z":       [0.0] * (max_elements - n_used),
                "beg_seq":      [0]   * (max_elements - n_used),
                "conf_type_id": ["OTHER"] * (max_elements - n_used),
            }
        )
        sub = pd.concat([sub, pad], ignore_index=True)

    # Channel 0: pairwise Euclidean distance
    coords   = sub[["mean_x", "mean_y", "mean_z"]].values
    diff     = coords[:, None, :] - coords[None, :, :]
    dist_mat = np.sqrt((diff ** 2).sum(axis=-1)).astype(np.float32)

    # Channel 1: pairwise sequence separation
    seq_pos  = sub["beg_seq"].values.astype(float)
    seq_sep  = np.abs(seq_pos[:, None] - seq_pos[None, :]).astype(np.float32)

    # Channel 2: same-type indicator
    type_idx  = sub["conf_type_id"].map(conf_type_idx).values
    same_type = (type_idx[:, None] == type_idx[None, :]).astype(np.float32)

    # Upper triangle indices (excludes diagonal)
    idx = np.triu_indices(max_elements, k=1)

    # Flatten upper triangle of each channel and concatenate
    vector = np.concatenate([
        dist_mat[idx],
        seq_sep[idx],
        same_type[idx],
    ])

    return vector, meta


def load_all_csvs(centroids_dir: Path, max_elements: int, logger) -> tuple[np.ndarray, list, list]:
    """
    Load all *_centroids.csv files and return:
        vectors     : np.float32 (N, D)
        protein_ids : list of str
        all_meta    : list of dicts
    """
    csv_files = sorted(centroids_dir.glob("*_centroids.csv"))
    if not csv_files:
        logger.error(f"No *_centroids.csv files found in '{centroids_dir}'")
        sys.exit(1)

    logger.info(f"Found {len(csv_files)} centroid CSV(s) in {centroids_dir}")
    logger.info(f"MAX_ELEMENTS = {max_elements}")

    vectors, protein_ids, all_meta, failed = [], [], [], []

    for csv_path in tqdm(csv_files, desc="Loading CSVs"):
        try:
            vec, meta = csv_to_vector(csv_path, max_elements)
            vectors.append(vec)
            protein_ids.append(meta["protein_id"])
            all_meta.append(meta)
        except Exception as e:
            logger.warning(f"Failed on {csv_path.name}: {e}")
            failed.append(csv_path.name)

    if not vectors:
        logger.error("No vectors produced. Exiting.")
        sys.exit(1)

    vecs = np.stack(vectors, axis=0).astype(np.float32)
    logger.info(f"Vector matrix shape : {vecs.shape}")

    n_trunc = sum(1 for m in all_meta if m["truncated"])
    if n_trunc:
        logger.warning(f"{n_trunc} protein(s) were truncated to {max_elements} elements")
    if failed:
        logger.warning(f"{len(failed)} file(s) failed to load: {failed}")

    return vecs, protein_ids, all_meta


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------

def build_faiss_index(vecs: np.ndarray, logger) -> tuple[faiss.IndexFlatIP, np.ndarray]:
    logger.info("L2-normalising vectors and building FAISS cosine-similarity index")
    vecs_norm = vecs.copy()
    faiss.normalize_L2(vecs_norm)
    index = faiss.IndexFlatIP(vecs_norm.shape[1])
    index.add(vecs_norm)
    logger.info(f"Index built: {index.ntotal} vectors, dim={vecs_norm.shape[1]}")
    return index, vecs_norm


# ---------------------------------------------------------------------------
# Ground-truth labels
# ---------------------------------------------------------------------------

def load_dgeb_labels(label_path: Path, protein_ids: list, logger) -> dict:
    """
    Load DGEB ground-truth JSON.

    Expected format:
        { "protein_A": ["protein_B", "protein_C"], ... }
    """
    logger.info(f"Loading DGEB labels from {label_path}")
    with open(label_path) as f:
        raw = json.load(f)
    labels   = {k: set(v) for k, v in raw.items()}
    covered  = sum(1 for pid in protein_ids if pid in labels)
    logger.info(f"Labels cover {covered}/{len(protein_ids)} proteins in this dataset")
    return labels


def make_demo_labels(protein_ids: list, logger) -> dict:
    """Synthetic labels for pipeline testing — every protein's relevant set is the next 3."""
    logger.warning("DEMO MODE — synthetic labels, metrics are not meaningful")
    n      = len(protein_ids)
    labels = {}
    for i, pid in enumerate(protein_ids):
        labels[pid] = {protein_ids[(i + 1) % n],
                       protein_ids[(i + 2) % n],
                       protein_ids[(i + 3) % n]}
    return labels


def load_dgeb_labels_from_hf(protein_ids: list, logger) -> dict:
    """
    Pull euk_retrieval ground-truth labels directly from HuggingFace.

    The euk_retrieval task has:
      - queries  : test split  (311 eukaryotic proteins)
      - corpus   : train split (3202 bacterial proteins)
      - qrels    : query_id → corpus_id relevance pairs

    We only keep queries and corpus entries that exist in our CSV set
    so that retrieval is evaluated only over proteins we have structures for.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Install the 'datasets' package: pip install datasets")
        sys.exit(1)

    logger.info("Pulling DGEB euk_retrieval labels from HuggingFace...")

    qrels_ds = load_dataset(
        "tattabio/euk_retrieval_qrels",
        revision="a5aa01e9b9738074aba57fc07434e352c4c71e4b",
        split="train"
    )

    our_ids = set(protein_ids)

    labels = {}
    skipped_query   = 0
    skipped_corpus  = 0

    for row in qrels_ds:
        qid = row["query_id"]
        cid = row["corpus_id"]

        if qid not in our_ids:
            skipped_query += 1
            continue
        if cid not in our_ids:
            skipped_corpus += 1
            continue

        if qid not in labels:
            labels[qid] = set()
        labels[qid].add(cid)

    logger.info(f"Labels built for {len(labels)} query proteins")
    logger.info(f"Skipped {skipped_query} qrels (query not in our CSVs)")
    logger.info(f"Skipped {skipped_corpus} qrels (corpus entry not in our CSVs)")

    # Save labels JSON for reuse so HF doesn't need to be re-fetched
    labels_serialisable = {k: list(v) for k, v in labels.items()}
    labels_path = Path("dgeb_euk_labels.json")
    labels_path.write_text(json.dumps(labels_serialisable, indent=2))
    logger.info(f"Labels saved to {labels_path} (use --labels to reuse without re-fetching)")

    return labels


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def dcg_at_k(rel: list, k: int) -> float:
    return sum(r / np.log2(i + 2) for i, r in enumerate(rel[:k]))

def ndcg_at_k(rel: list, k: int) -> float:
    idcg = dcg_at_k(sorted(rel, reverse=True), k)
    return 0.0 if idcg == 0 else dcg_at_k(rel, k) / idcg

def mrr_at_k(rel: list, k: int) -> float:
    for i, r in enumerate(rel[:k]):
        if r > 0:
            return 1.0 / (i + 1)
    return 0.0

def map_at_k(rel: list, k: int) -> float:
    hits, s = 0, 0.0
    for i, r in enumerate(rel[:k]):
        if r > 0:
            hits += 1
            s    += hits / (i + 1)
    n_rel = sum(1 for r in rel if r > 0)
    return 0.0 if n_rel == 0 else s / min(n_rel, k)

def precision_at_k(rel: list, k: int) -> float:
    return sum(1 for r in rel[:k] if r > 0) / k

def recall_at_k(rel: list, k: int, n_rel: int) -> float:
    return 0.0 if n_rel == 0 else sum(1 for r in rel[:k] if r > 0) / n_rel


# ---------------------------------------------------------------------------
# Retrieval + evaluation
# ---------------------------------------------------------------------------

def run_retrieval(index, vecs_norm, protein_ids, labels, k_values, logger):
    max_k     = max(k_values)
    query_ids = [pid for pid in protein_ids if pid in labels]
    id_to_idx = {pid: i for i, pid in enumerate(protein_ids)}
    protein_ids_arr = np.array(protein_ids)

    logger.info(f"Running retrieval for {len(query_ids)} labelled queries, max_k={max_k}")

    rows         = []
    recall_accum = np.zeros(max_k, dtype=np.float64)
    n_queries    = 0
    start        = time.time()

    for qid in tqdm(query_ids, desc="Retrieving"):
        if qid not in id_to_idx:
            continue
        q_idx    = id_to_idx[qid]
        q_vec    = vecs_norm[q_idx: q_idx + 1]
        relevant = labels[qid]
        n_rel    = len(relevant)
        if n_rel == 0:
            continue

        distances, indices = index.search(q_vec, max_k + 1)
        distances = distances[0]
        indices   = indices[0]

        # Remove query protein from results
        mask      = indices != q_idx
        indices   = indices[mask][:max_k]
        distances = distances[mask][:max_k]

        retrieved_ids = protein_ids_arr[indices]
        rel           = [1.0 if rid in relevant else 0.0 for rid in retrieved_ids]

        row = {"protein_id": qid, "n_relevant": n_rel}
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
    logger.info(f"Retrieval complete in {elapsed:.1f}s")

    per_query_df = pd.DataFrame(rows)

    # Aggregate summary
    summary = {"n_queries": n_queries, "k_values": k_values, "metrics": {}}
    for col in [c for c in per_query_df.columns if c not in ("protein_id", "n_relevant")]:
        val = float(per_query_df[col].mean())
        summary["metrics"][col] = round(val, 6)

    # Recall curve
    recall_curve = pd.DataFrame({
        "k":      list(range(1, max_k + 1)),
        "recall": (recall_accum / max(n_queries, 1)).tolist(),
    })

    return per_query_df, recall_curve, summary


# ---------------------------------------------------------------------------
# Score distribution for histogram
# ---------------------------------------------------------------------------

def score_distribution(per_query_df: pd.DataFrame, k: int) -> pd.DataFrame:
    col = f"map@{k}"   # map_at_5 is DGEB's primary metric for retrieval
    if col not in per_query_df.columns:
        return pd.DataFrame()
    counts, edges = np.histogram(per_query_df[col].dropna(), bins=20, range=(0.0, 1.0))
    return pd.DataFrame({
        "bin_left":  edges[:-1].round(3),
        "bin_right": edges[1:].round(3),
        "count":     counts,
    })


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("retrieval")
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
        description="Structure-based protein retrieval using FAISS cosine similarity."
    )
    p.add_argument("--centroids",    required=True,
                   help="Directory containing *_centroids.csv files")
    p.add_argument("--labels",       default=None,
                   help="Path to pre-saved DGEB ground-truth JSON")
    p.add_argument("--dgeb",         action="store_true",
                   help="Pull euk_retrieval labels directly from HuggingFace")
    p.add_argument("--demo",         action="store_true",
                   help="Use synthetic labels to verify the pipeline end-to-end")
    p.add_argument("--max_elements", type=int, default=150,
                   help="Pad/truncate proteins to this many structural elements (default: 150)")
    p.add_argument("--k_values",     default="5,10,50",
                   help="Comma-separated k values for metrics (default: 5,10,50)")
    p.add_argument("--output_dir",   default="results",
                   help="Directory for output files (default: results)")
    return p.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger   = setup_logger(output_dir / "retrieval_log.txt")
    k_values = [int(k.strip()) for k in args.k_values.split(",")]

    logger.info("=" * 60)
    logger.info("Structure-based Protein Retrieval (FAISS)")
    logger.info("=" * 60)
    logger.info(f"Centroids dir : {args.centroids}")
    logger.info(f"Max elements  : {args.max_elements}")
    logger.info(f"k values      : {k_values}")
    logger.info(f"Primary metric: map@5 (DGEB standard)")

    # 1. Load CSVs → vectors (on the fly, no intermediate files)
    vecs, protein_ids, all_meta = load_all_csvs(
        Path(args.centroids), args.max_elements, logger
    )

    # 2. Build FAISS index
    index, vecs_norm = build_faiss_index(vecs, logger)

    # 3. Labels
    if args.demo:
        labels = make_demo_labels(protein_ids, logger)
    elif args.dgeb:
        labels = load_dgeb_labels_from_hf(protein_ids, logger)
    elif args.labels:
        labels = load_dgeb_labels(Path(args.labels), protein_ids, logger)
    else:
        logger.error("Provide one of: --dgeb  --labels <path>  --demo")
        sys.exit(1)

    # 4. Retrieval + evaluation
    per_query_df, recall_curve, summary = run_retrieval(
        index, vecs_norm, protein_ids, labels, k_values, logger
    )

    # 5. Save all outputs
    (output_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2))
    per_query_df.to_csv(output_dir / "per_query_scores.csv", index=False)
    recall_curve.to_csv(output_dir / "recall_curve.csv", index=False)

    for k in k_values:
        score_distribution(per_query_df, k).to_csv(
            output_dir / f"score_distribution_map@{k}.csv", index=False
        )

    # Save protein metadata
    (output_dir / "protein_meta.json").write_text(
        json.dumps({"proteins": all_meta}, indent=2)
    )

    # 5b. Append to persistent run manifest (never overwritten)
    import datetime
    manifest_path = output_dir / "run_manifest.jsonl"
    manifest_entry = {
        "timestamp":    datetime.datetime.now().isoformat(timespec="seconds"),
        "max_elements": args.max_elements,
        "labels_source": "demo" if args.demo else ("dgeb_hf" if args.dgeb else args.labels),
        "n_proteins":   len(protein_ids),
        "n_queries":    summary["n_queries"],
        "k_values":     k_values,
        "metrics":      summary["metrics"],
    }
    with open(manifest_path, "a") as f:
        f.write(json.dumps(manifest_entry) + "\n")
    logger.info(f"Run appended to manifest -> {manifest_path}")

    logger.info(f"\nAll outputs saved to: {output_dir}/")

    # 6. Print summary table
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


if __name__ == "__main__":
    main()