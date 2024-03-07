#!/bin/bash
#SBATCH --mem=32g
#SBATCH -c 2
#SBATCH --output=parse_multiple_chains.out

source activate mlfold
python parse_multiple_chains.py --input_path='/projects/ml/struc2seq/data_for_complexes/training_scripts/2022/LigandMPNN/benchmarking/clean_pdb_only' --output_path='./parsed_pdbs.jsonl'
