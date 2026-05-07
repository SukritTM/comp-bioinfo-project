# CS 690U Final Project run instructions

## Install Dependencies

Make a new conda environment: `conda create -n <env_name> python=3.10`
Activate the environment: `conda activate <env_name>`
Install foldseek: `conda install -c conda-forge -c bioconda foldseek`
Install Dependencies: `conda install numpy pandas scipy seaborn matplotlib tqdm`
Install DGEB: `pip install dgeb`

## Downloading Structures from AlphaFold Protein Structure Database

There are two scripts that download structures from the AlphaFold PSDB: `download-structures.py` and `download-structures-traintest.py`. `download-structures.py` downloads all structures to a single folder, and `download-structures-traintest.py` seperates the downloads by the query and corpus sets (test and train respectively). FoldSeek requires the structures to be seperated into query and corpus sets, and the rest of the scripts expect all structures in the same place. To download structures in either format, simply run the scripts.

## Running FoldSeek

Assume that your train and test seperated structures are stored in `path\to\traintest\structures\train` and `path\to\traintest\structures\test` respectively. Follow the steps to generate alignment scores from FoldSeek:

1) `foldseek createdb path\to\traintest\structures\train\ corpus_db\corpus`
2) `foldseek createdb path\to\traintest\structures\test\ query_db\query`
3) Depending on if you want all-vs-all or prefilter results:
    1) `foldseek search query_db/query corpus_db/corpus align_db/align -s 7.5 -a --exhaustive-search` for all-v-all
    2)  `foldseek search query_db/query corpus_db/corpus align_db/align -s 7.5 -a ` for prefiltered
4) `foldseek convertalis query_db\query corpus_db\corpus align_db\align alignment.tsv --format-mode 4 --format-output query,target,alntmscore,rmsd,prob,evalue,alnlen,qlen,tlen,qcov,tcov`
5) `python format-results.py -i alignment.tsv -o alignment-formatted.csv`

## Compute Metrics

Run `foldseek_compare.py --results alignment-formatted.csv --labels dgeb_euk_labels.json --output_dir results_foldseek`