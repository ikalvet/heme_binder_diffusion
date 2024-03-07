#!/bin/bash
#SBATCH -p gpu
#SBATCH --mem=32g
#SBATCH --gres=gpu:rtx2080:1
#SBATCH -c 3
#SBATCH --output=example_3.out

source activate mlfold

folder_with_pdbs="/home/justas/MPNN_tests/proteinmpnn/ligandMPNN/input_example"
path_for_parsed_chains="../output/parsed_design_1.jsonl"
path_for_designed_sequences="../output"


python ../helper_scripts/parse_multiple_chains.py --input_path=$folder_with_pdbs --output_path=$path_for_parsed_chains

python ../protein_mpnn_run.py \
        --model_name "v_32_010" \
        --jsonl_path $path_for_parsed_chains \
        --out_folder $path_for_designed_sequences \
        --num_seq_per_target 2 \
        --sampling_temp "0.1" \
        --pack_side_chains 1 \
        --batch_size 1
