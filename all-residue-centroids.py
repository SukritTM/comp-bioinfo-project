"""
residue-centroids.py
------------------------
Reads a folder of mmCIF files and computes the mean (x, y, z)
centroid of all atoms belonging to each amino acid residue.

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
        description="Compute per-residue centroids from a folder of mmCIF files."
    )
    p.add_argument("cif_dir", help="Path to folder containing mmCIF / PDBx files")
    p.add_argument(
        "-o", "--output_dir", default="data/residue_centroids",
        help="Output directory for CSV files (default: data/residue_centroids)"
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

# Standard 20 amino acid residue names
AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL",
}


def load_atoms(block, atom_filter):
    """Return a DataFrame of atom coordinates from _atom_site, amino acids only."""
    at = block.find(
        "_atom_site.",
        [
            "label_asym_id",
            "label_seq_id",
            "label_comp_id",
            "label_atom_id",
            "Cartn_x",
            "Cartn_y",
            "Cartn_z",
        ],
    )

    rows = []
    for row in at:
        res_name = row["_atom_site.label_comp_id"]
        if res_name not in AMINO_ACIDS:
            continue

        atom_name = row["_atom_site.label_atom_id"]
        if atom_filter != "all" and atom_name not in atom_filter:
            continue

        seq_id = row["_atom_site.label_seq_id"]
        if seq_id in (".", "?"):
            continue

        rows.append({
            "chain":    row["_atom_site.label_asym_id"],
            "seq":      int(seq_id),
            "res_name": res_name,
            "atom_name": atom_name,
            "x":        float(row["_atom_site.Cartn_x"]),
            "y":        float(row["_atom_site.Cartn_y"]),
            "z":        float(row["_atom_site.Cartn_z"]),
        })
    return pd.DataFrame(rows)


def compute_centroids(atom_df):
    """Compute mean x/y/z centroid for each (chain, seq, res_name) residue."""
    if atom_df.empty:
        return pd.DataFrame()

    grouped = atom_df.groupby(["chain", "seq", "res_name"], sort=False)

    result = grouped.agg(
        n_atoms=("x", "count"),
        mean_x=("x", "mean"),
        mean_y=("y", "mean"),
        mean_z=("z", "mean"),
    ).reset_index()

    result[["mean_x", "mean_y", "mean_z"]] = result[["mean_x", "mean_y", "mean_z"]].round(3)
    result = result.sort_values(["chain", "seq"]).reset_index(drop=True)

    return result


def process_cif(cif_path, output_dir, atom_filter):
    """Process a single CIF file and save its per-residue centroid CSV."""
    print(f"\n--- Processing {cif_path.name} ---")
    try:
        doc = gemmi.cif.read(str(cif_path))
        block = doc.sole_block()

        atom_df = load_atoms(block, atom_filter)
        print(f"  Loaded {len(atom_df)} atoms across "
              f"{atom_df.groupby(['chain','seq']).ngroups if not atom_df.empty else 0} residues.")

        if atom_df.empty:
            print("  No amino acid atoms found, skipping.")
            return None

        result_df = compute_centroids(atom_df)
        print(f"  Computed {len(result_df)} residue centroids.")

        out_path = output_dir / (cif_path.stem + "_residue_centroids.csv")
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