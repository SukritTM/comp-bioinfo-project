import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-tsv', required=True, help='Path to the unformatted FoldSeek TSV file')
parser.add_argument('-o', '--output', required=True, help='Path to the output folder to store the CSV file')
args = parser.parse_args()

fs_df = pd.read_csv(args.input_tsv, sep='\t')
query_ids = fs_df['query'].map(lambda s: s.split('-')[1])
target_ids = fs_df['target'].map(lambda s: s.split('-')[1])
fs_df['query_id'] = query_ids
fs_df['target_id'] = target_ids

fs_df.to_csv(args.output)