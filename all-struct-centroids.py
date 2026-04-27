"""
struct-centroids.py
------------------------
Reads a folder of mmCIF files, parses the _struct_conf secondary-structure
table and _atom_site coordinates, then computes the mean (x, y, z)
centroid of all atoms belonging to each structural element.

Output: One CSV per input file, saved to data/secondary_centroids/

Dependencies:
    pip install gemmi pandas
"""

import sys
import argparse
import gemmi
import pandas as pd
from pathlib import Path
from time import perf_counter as pf


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute centroids of struct_conf elements from a folder of mmCIF files."
    )
    p.add_argument("cif_dir", help="Path to folder containing mmCIF / PDBx files")
    p.add_argument(
        "-o", "--output_dir", default="data//euk_retrieval//secondary_centroids",
        help="Output directory for CSV files (default: data/secondary_centroids)"
    )
    p.add_argument(
        "--atoms", default="all",
        help=(
            "Which atoms to include in the centroid calculation. "
            "Options: 'all' (default), 'backbone' (N/CA/C/O only), "
            "or a comma-separated list of atom names e.g. 'CA,CB'"
        )
    )
    return p.parse_args()


BACKBONE_ATOMS = {"N", "CA", "C", "O"}


def load_struct_conf(block):
    """Return a DataFrame of struct_conf rows."""
    sc = block.find(
        "_struct_conf.",
        [
            "id",
            "conf_type_id",
            "beg_label_asym_id",
            "beg_label_seq_id",
            "beg_label_comp_id",
            "end_label_asym_id",
            "end_label_seq_id",
            "end_label_comp_id",
        ],
    )

    rows = []
    for row in sc:
        rows.append({
            "id":            row["_struct_conf.id"],
            "conf_type_id":  row["_struct_conf.conf_type_id"],
            "beg_chain":     row["_struct_conf.beg_label_asym_id"],
            "beg_seq":       int(row["_struct_conf.beg_label_seq_id"]),
            "beg_res":       row["_struct_conf.beg_label_comp_id"],
            "end_chain":     row["_struct_conf.end_label_asym_id"],
            "end_seq":       int(row["_struct_conf.end_label_seq_id"]),
            "end_res":       row["_struct_conf.end_label_comp_id"],
        })
    return pd.DataFrame(rows)


def load_atoms(block, atom_filter):
    """Return a DataFrame of atom coordinates from _atom_site."""
    at = block.find(
        "_atom_site.",
        [
            "label_asym_id",
            "label_seq_id",
            "label_atom_id",
            "Cartn_x",
            "Cartn_y",
            "Cartn_z",
        ],
    )

    rows = []
    for row in at:
        atom_name = row["_atom_site.label_atom_id"]
        if atom_filter != "all" and atom_name not in atom_filter:
            continue
        seq_id = row["_atom_site.label_seq_id"]
        if seq_id in (".", "?"):
            continue
        rows.append({
            "chain":     row["_atom_site.label_asym_id"],
            "seq":       int(seq_id),
            "atom_name": atom_name,
            "x":         float(row["_atom_site.Cartn_x"]),
            "y":         float(row["_atom_site.Cartn_y"]),
            "z":         float(row["_atom_site.Cartn_z"]),
        })
    return pd.DataFrame(rows)


def compute_centroids(sc_df, atom_df):
    """For each struct_conf element, compute mean x/y/z of matching atoms."""
    results = []
    for _, elem in sc_df.iterrows():
        mask = (
            (atom_df["chain"] == elem["beg_chain"]) &
            (atom_df["seq"]   >= elem["beg_seq"])   &
            (atom_df["seq"]   <= elem["end_seq"])
        )
        subset = atom_df[mask]
        n = len(subset)
        if n == 0:
            mean_x = mean_y = mean_z = float("nan")
        else:
            mean_x = subset["x"].mean()
            mean_y = subset["y"].mean()
            mean_z = subset["z"].mean()

        results.append({
            "id":           elem["id"],
            "conf_type_id": elem["conf_type_id"],
            "beg_chain":    elem["beg_chain"],
            "beg_seq":      elem["beg_seq"],
            "beg_res":      elem["beg_res"],
            "end_chain":    elem["end_chain"],
            "end_seq":      elem["end_seq"],
            "end_res":      elem["end_res"],
            "n_atoms":      n,
            "mean_x":       round(mean_x, 3) if n else float("nan"),
            "mean_y":       round(mean_y, 3) if n else float("nan"),
            "mean_z":       round(mean_z, 3) if n else float("nan"),
        })

    return pd.DataFrame(results)


def process_cif(cif_path, output_dir, atom_filter):
    """Process a single CIF file and save its centroid CSV."""
    print(f"\n--- Processing {cif_path.name} ---")
    try:
        doc = gemmi.cif.read(str(cif_path))
        block = doc.sole_block()

        sc_df = load_struct_conf(block)
        print(f"  Found {len(sc_df)} structural elements.")

        if sc_df.empty:
            print("  No struct_conf entries found, skipping.")
            return None

        atom_df = load_atoms(block, atom_filter)
        print(f"  Loaded {len(atom_df)} atoms.")

        result_df = compute_centroids(sc_df, atom_df)

        out_path = output_dir / (cif_path.stem + "_centroids.csv")
        result_df.to_csv(out_path, index=False)
        print(f"  Saved -> {out_path}")
        return result_df

    except Exception as e:
        print(f"  ERROR processing {cif_path.name}: {e}", file=sys.stderr)
        return None


def main():
    timer = pf()
    args = parse_args()

    # Resolve atom filter
    if args.atoms == "all":
        atom_filter = "all"
    elif args.atoms == "backbone":
        atom_filter = BACKBONE_ATOMS
    else:
        atom_filter = set(a.strip() for a in args.atoms.split(","))

    cif_dir = Path(args.cif_dir)
    if not cif_dir.is_dir():
        print(f"Error: '{cif_dir}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    cif_files = sorted(cif_dir.glob("*.cif"))
    if not cif_files:
        # Also try .mmcif extension
        cif_files = sorted(cif_dir.glob("*.mmcif"))
    if not cif_files:
        print(f"No .cif or .mmcif files found in '{cif_dir}'.", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"Found {len(cif_files)} CIF file(s) to process.")
    print(f"Atom filter: {args.atoms}")

    succeeded, failed = 0, 0
    for cif_path in cif_files:
        result = process_cif(cif_path, output_dir, atom_filter)
        if result is not None:
            succeeded += 1
        else:
            failed += 1

    elapsed = pf() - timer
    print(f"\n{'='*50}")
    print(f"Done. {succeeded} succeeded, {failed} failed.")
    print(f"Total time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()