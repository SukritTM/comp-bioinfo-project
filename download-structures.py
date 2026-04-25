import Bio.PDB.alphafold_db as adb
# from datasets import load_dataset

from datasets import load_dataset
from tqdm import tqdm

from time import perf_counter as pf
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-d', '--dataset', default='tattabio/euk_retrieval', help='Huggingface Dataset name')
parser.add_argument('-o', '--output_dir', default='data\\euk_retrieval\\structures\\', help='Output directory')
args = parser.parse_args()


ds = load_dataset(args.dataset)

train = ds['train'].to_pandas()
test = ds['test'].to_pandas()

num_multiple = 0
no_results = []
results = []
multiple = []

timer = pf()
print('Training subset')
for i, uniprot_id in enumerate(tqdm(train['Entry'])):
    # print(uniprot_id, flush=True)
    try:
        pred = list(adb.get_predictions(uniprot_id))
        results.append(uniprot_id)
        # adb.download_cif_for()
        if len(pred) > 1:
            num_multiple += 1
            multiple.append(uniprot_id)
            canonical_file = sorted(list(pred), key=lambda d: d['modelEntityId'])[-1]
        else:
            canonical_file = pred[0]

        adb.download_cif_for(canonical_file, args.output_dir)
    except:
        pred = []
        no_results.append(uniprot_id)
    # print(i, len(pred))


print('Testing subset')
for i, uniprot_id in enumerate(tqdm(test['Entry'])):
    # print(uniprot_id, flush=True)
    try:
        pred = list(adb.get_predictions(uniprot_id))
        results.append(uniprot_id)
        # adb.download_cif_for()
        if len(pred) > 1:
            num_multiple += 1
            multiple.append(uniprot_id)
            canonical_file = sorted(list(pred), key=lambda d: d['modelEntityId'])[-1]
        else:
            canonical_file = pred[0]

        adb.download_cif_for(canonical_file, args.output_dir)
    except:
        pred = []
        no_results.append(uniprot_id)
    # print(i, len(pred))



timer = pf() - timer


print(f'no_results: {no_results}')
print(f'multiple: {multiple}')

print(f'time: {timer:2f}')
print(f'num_multiple: {num_multiple}')
print(f'num with no results: {len(no_results)}')
print(f'num with results: {len(results)}')
