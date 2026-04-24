"""
struct-centroids.py
------------------------
Reads an mmCIF file, parses the _struct_conf secondary-structure
table and _atom_site coordinates, then computes the mean (x, y, z)
centroid of all atoms belonging to each structural element.
 
Output: CSV and a pretty-printed table to stdout.
 
Dependencies:
    pip install gemmi pandas
"""
 
import sys
import argparse
import gemmi
import pandas as pd
from time import perf_counter as pf

 
def parse_args():
    p = argparse.ArgumentParser(
        description="Compute centroids of struct_conf elements from an mmCIF file."
    )
    p.add_argument("cif", help="Path to the input mmCIF / PDBx file")
    p.add_argument(
        "-o", "--output", default="struct_conf_centroids.csv",
        help="Output CSV filename (default: struct_conf_centroids.csv)"
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
 
    print(f"Reading {args.cif} ...")
    doc = gemmi.cif.read(args.cif)
    block = doc.sole_block()
 
    print("Parsing _struct_conf ...")
    sc_df = load_struct_conf(block)
    print(f"  Found {len(sc_df)} structural elements.")
 
    print(f"Parsing _atom_site (filter: {args.atoms}) ...")
    atom_df = load_atoms(block, atom_filter)
    print(f"  Loaded {len(atom_df)} atoms.")
 
    print("Computing centroids ...")
    result_df = compute_centroids(sc_df, atom_df)
 
    # Save CSV
    result_df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")
 
    # Pretty-print to stdout
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", "{:.3f}".format)
    print("\n" + result_df.to_string(index=False))

    timer = pf() - timer
    print(f'time: {timer:2f}')
 
 
if __name__ == "__main__":
    main()