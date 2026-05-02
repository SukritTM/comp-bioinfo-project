import gemmi
import pandas as pd
from typing import List, Dict
import numpy as np
from argparse import ArgumentParser
import os

import datasets

parser = ArgumentParser()
parser.add_argument('-c', '--cif_dir', help='Path to folder containing mmCIF / PDBx files')
parser.add_argument('-d', '--dataset', default='tattabio/euk_retrieval', help='Huggingface Dataset name')
parser.add_argument('-o', '--output_dir', default='data\\euk_retrieval\\', help='Output directory')
# parser.add_argument('-o', '--output_dir', help='Output directory')

def count_secondary_structures(filepath: str) -> pd.Series:
    file = gemmi.cif.read_file(filepath)
    block = file.sole_block()
    unique, counts = np.unique(block.find_loop('_struct_conf.conf_type_id'), return_counts=True)
    return pd.Series(counts, index=unique)

def find_file(tgt_dir, uniprot_id):
    filenames = os.listdir(tgt_dir)
    for fname in filenames:
        if uniprot_id in fname:
            return os.path.join(tgt_dir, fname)


def main():
    args = parser.parse_args()

    print('Downloading dataset...', end='', flush=True)
    ds = datasets.load_dataset(args.dataset)
    print('Downloaded!')

    train = ds['train']
    test = ds['test']

    print('Counting train secondary structures...', end='', flush=True)
    serieses = []
    for uniprot_id in train['Entry']:
        filepath = find_file(args.cif_dir, uniprot_id)
        if filepath:
            series = count_secondary_structures(filepath)
            serieses.append(series)
    
    train_df = pd.DataFrame(serieses).fillna(0)
    train_df['Entry'] = train.to_pandas()['Entry']
    print('Done!')
    
    print('Counting test secondary structures...', end='', flush=True)
    serieses = []
    for uniprot_id in test['Entry']:
        filepath = find_file(args.cif_dir, uniprot_id)
        if filepath:
            series = count_secondary_structures(filepath)
            serieses.append(series)
    
    test_df = pd.DataFrame(serieses).fillna(0)
    test_df['Entry'] = test.to_pandas()['Entry']
    print('Done!')
    
    train_df.to_csv(os.path.join(args.output_dir, 'train_secondary_structure_frequencies.csv'), index=False)
    test_df.to_csv(os.path.join(args.output_dir, 'test_secondary_structure_frequencies.csv'), index=False)
    
    
    
if __name__ == "__main__":
    main()
    # Example usage
    # import sys
    
    # if len(sys.argv) > 1:
    #     filepath = sys.argv[1]
    #     structure_types = ['HELX_LH_PP_P', 'HELX_RH_3T_P', 'TURN_TY1_P', 'BEND', 'STRN', 'HELX_RH_AL_P', 'HELX_RH_PI_P']
        
    #     result = count_secondary_structures(filepath)
    #     print("\nSecondary Structure Counts:")
    #     print(result)
    #     print(f"\nTotal structures: {result.sum()}")
    # else:
    #     print("Usage: python secondary_structure_counter.py <path_to_cif_file>")
    #     print("\nExample:")
    #     print("  python secondary_structure_counter.py protein.cif")