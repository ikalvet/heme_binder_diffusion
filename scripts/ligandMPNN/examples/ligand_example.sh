#!/bin/bash
#SBATCH -p gpu
#SBATCH --mem=32g
#SBATCH --gres=gpu:rtx2080:1
#SBATCH -c 3
#SBATCH --output=example_3.out

source activate mlfold

path_to_PDB="/home/gyurie/mpnn_ligand/inf_example/srtRe1657.pdb"
path_to_PDB_params="/home/gyurie/mpnn_ligand/inf_example/SRO.params"
path_for_designed_sequences="../output"

python ../protein_mpnn_run.py \
        --model_name "v_32_010" \
        --pdb_path $path_to_PDB \
        --ligand_params_path $path_to_PDB_params \
        --out_folder $path_for_designed_sequences \
        --num_seq_per_target 2 \
        --sampling_temp "0.1" \
        --pack_side_chains 1 \
        --batch_size 1
