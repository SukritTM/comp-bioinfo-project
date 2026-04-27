"""
run_dgeb_baseline.py
--------------------
Runs the official DGEB evaluation pipeline for ESM-2 on the euk_retrieval task
and saves a clean summary for comparison with our structure-based method.

Must be run as a script (not in a Jupyter notebook) due to Windows multiprocessing
requirements. The if __name__ == "__main__" guard is mandatory on Windows.

Usage
-----
    python run_dgeb_baseline.py
    python run_dgeb_baseline.py --model facebook/esm2_t12_35M_UR50D
    python run_dgeb_baseline.py --model facebook/esm2_t30_150M_UR50D

Output
------
    dgeb_baseline_results/
        raw/esm2_*/euk_retrieval.json  -- raw DGEB output (per-layer metrics)
        baseline_summary.json          -- clean summary: best layer + final layer map@5

Dependencies
------------
    pip install dgeb torch transformers
"""

import os
import sys
import json
import argparse
import datetime
from pathlib import Path

# Must be set before any torch/transformers imports on Windows
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Monkey-patch torch DataLoader to force num_workers=0 on Windows.
# This prevents the multiprocessing pickling error.
import platform
if platform.system() == "Windows":
    import torch.utils.data
    _orig_init = torch.utils.data.DataLoader.__init__
    def _patched_init(self, *args, **kwargs):
        kwargs["num_workers"] = 0
        _orig_init(self, *args, **kwargs)
    torch.utils.data.DataLoader.__init__ = _patched_init


def parse_args():
    p = argparse.ArgumentParser(
        description="Run DGEB ESM-2 baseline on euk_retrieval task."
    )
    p.add_argument(
        "--model",
        default="facebook/esm2_t6_8M_UR50D",
        help=(
            "HuggingFace model ID to evaluate.\n"
            "  facebook/esm2_t6_8M_UR50D    (8M  params, fastest)\n"
            "  facebook/esm2_t12_35M_UR50D  (35M params)\n"
            "  facebook/esm2_t30_150M_UR50D (150M params)\n"
        )
    )
    p.add_argument(
        "--output_dir",
        default="dgeb_baseline_results",
        help="Directory to write results to. Default: dgeb_baseline_results"
    )
    return p.parse_args()


def parse_dgeb_json(results_dir: Path) -> tuple:
    """
    Find and parse the DGEB output JSON for euk_retrieval.

    DGEB v0.2.0 structure:
    {
      "task": {"id": "euk_retrieval", "primary_metric_id": "map_at_5", ...},
      "results": [
        {
          "layer_number": 3,
          "metrics": [{"id": "map_at_5", "value": 0.12}, ...]
        },
        {
          "layer_number": 5,
          "metrics": [{"id": "map_at_5", "value": 0.21}, ...]
        }
      ]
    }

    Returns:
        raw_data   : full JSON dict
        per_layer  : { layer_number (int): { metric_id (str): value (float) } }
        primary_metric : str, e.g. "map_at_5"
    """
    raw_data       = {}
    per_layer      = {}
    primary_metric = "map_at_5"

    for jf in results_dir.rglob("*.json"):
        try:
            data = json.loads(jf.read_text())
            task_id = data.get("task", {}).get("id", "")
            if "euk_retrieval" in task_id:
                raw_data       = data
                primary_metric = data.get("task", {}).get("primary_metric_id", "map_at_5")
                print(f"Parsed: {jf}")
                for layer in data.get("results", []):
                    ln = layer.get("layer_number")
                    per_layer[ln] = {
                        m["id"]: m["value"]
                        for m in layer.get("metrics", [])
                        if isinstance(m.get("value"), (int, float))
                    }
                break
        except Exception:
            continue

    return raw_data, per_layer, primary_metric


def main():
    args = parse_args()

    try:
        import dgeb
    except ImportError:
        print("ERROR: dgeb not installed. Run: pip install dgeb")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DGEB ESM-2 Baseline Evaluation")
    print("=" * 60)
    print(f"Model      : {args.model}")
    print(f"Task       : euk_retrieval")
    print(f"Output dir : {output_dir}")
    print("=" * 60)

    # Load model
    print(f"\nLoading model: {args.model} ...")
    model = dgeb.get_model(args.model)

    # Get only euk_retrieval task
    all_tasks = dgeb.get_tasks_by_modality(dgeb.Modality.PROTEIN)
    euk_tasks = [t for t in all_tasks if t.metadata.id == "euk_retrieval"]

    if not euk_tasks:
        print("ERROR: euk_retrieval task not found in DGEB.")
        sys.exit(1)

    print(f"Running evaluation on: {euk_tasks[0].metadata.id}")
    print("This may take 10-20 minutes on CPU...\n")

    # Run DGEB evaluation (run() only accepts model + output_folder in v0.2.0)
    evaluation = dgeb.DGEB(tasks=euk_tasks)
    evaluation.run(model, output_folder=str(output_dir / "raw"))

    print("\nEvaluation complete. Parsing results...")

    # Parse results
    raw_data, per_layer, primary_metric = parse_dgeb_json(output_dir / "raw")

    if not per_layer:
        print("WARNING: Could not parse results. Check raw JSON in dgeb_baseline_results/raw/")
        sys.exit(1)

    # Best layer = layer with highest primary metric value
    best_layer = max(per_layer, key=lambda ln: per_layer[ln].get(primary_metric, 0))
    last_layer = max(per_layer.keys())

    best_metrics = per_layer[best_layer]
    last_metrics = per_layer[last_layer]

    # Build clean summary
    summary = {
        "timestamp":        datetime.datetime.now().isoformat(timespec="seconds"),
        "model":            args.model,
        "task":             "euk_retrieval",
        "primary_metric":   primary_metric,
        "layers_evaluated": sorted(per_layer.keys()),
        "best_layer": {
            "layer_number":   best_layer,
            primary_metric:   best_metrics.get(primary_metric),
            "ndcg_at_5":      best_metrics.get("ndcg_at_5"),
            "ndcg_at_10":     best_metrics.get("ndcg_at_10"),
            "mrr_at_5":       best_metrics.get("mrr_at_5"),
            "precision_at_5": best_metrics.get("precision_at_5"),
            "recall_at_5":    best_metrics.get("recall_at_5"),
            "map_at_10":      best_metrics.get("map_at_10"),
            "map_at_50":      best_metrics.get("map_at_50"),
        },
        "last_layer": {
            "layer_number":   last_layer,
            primary_metric:   last_metrics.get(primary_metric),
            "ndcg_at_5":      last_metrics.get("ndcg_at_5"),
            "ndcg_at_10":     last_metrics.get("ndcg_at_10"),
            "mrr_at_5":       last_metrics.get("mrr_at_5"),
            "precision_at_5": last_metrics.get("precision_at_5"),
            "recall_at_5":    last_metrics.get("recall_at_5"),
            "map_at_10":      last_metrics.get("map_at_10"),
            "map_at_50":      last_metrics.get("map_at_50"),
        },
        "all_layers": {str(ln): per_layer[ln] for ln in sorted(per_layer.keys())},
    }

    summary_path = output_dir / "baseline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    # Print clean comparison table
    print("\n" + "=" * 60)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Primary metric: {primary_metric}")
    print()
    print(f"{'Layer':<10} {'map@5':<10} {'ndcg@5':<10} {'mrr@5':<10} {'prec@5':<10} {'rec@5':<10}")
    print("-" * 60)
    for ln in sorted(per_layer.keys()):
        m = per_layer[ln]
        marker = " <-- best" if ln == best_layer else ""
        print(
            f"{ln:<10} "
            f"{m.get('map_at_5', float('nan')):<10.4f} "
            f"{m.get('ndcg_at_5', float('nan')):<10.4f} "
            f"{m.get('mrr_at_5', float('nan')):<10.4f} "
            f"{m.get('precision_at_5', float('nan')):<10.4f} "
            f"{m.get('recall_at_5', float('nan')):<10.4f}"
            f"{marker}"
        )
    print()
    print(f"Best layer map@5  : {best_metrics.get(primary_metric):.4f}  (layer {best_layer})")
    print(f"Our method map@5  : 0.1503  (structure centroids + FAISS, max_elements=150)")
    print("=" * 60)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()