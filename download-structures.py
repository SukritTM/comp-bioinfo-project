import Bio.PDB.alphafold_db as adb
# from datasets import load_dataset

from datasets import load_dataset
from tqdm import tqdm

ds = load_dataset("tattabio/euk_retrieval")

train = ds['train'].to_pandas()
test = ds['test'].to_pandas()

num_multiple = 0
no_results = []
for i, uniprot_id in enumerate(train['Entry']):
    # print(uniprot_id, flush=True)
    try: 
        pred = list(adb.get(uniprot_id))
    except:
        pred = []
        no_results.append(uniprot_id)
    # print(i, len(pred))
    if len(pred) > 1:
        num_multiple += 1

print(f'num_multiple: {num_multiple}')
print(f'no_results: {no_results}')
