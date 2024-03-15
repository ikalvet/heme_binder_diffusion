# De novo heme binding protein design pipeline using RFdiffusionAA
![image](https://github.com/ikalvet/heme_binder_diffusion/assets/30599647/6aa36676-9cba-4a9d-940c-a313920935f3)
#### And other ligand binders too, I guess
Indrek Kalvet, PhD (Institute for Protein Design, University of Washington), ikalvet@uw.edu

As implemented in the publication <i>"Generalized Biomolecular Modeling and Design with RoseTTAFold All-Atom"</i>

Link:

The included notebook `pipeline.ipynb` illustrates the design of heme-binding proteins, starting from minimal information (heme + substrate + CYS motif). It should work with minor modifications also for any other ligand.

The pipeline consists of 7 steps:

0) The protein backbones are generated with RFdiffusionAA
1) Sequence is designed with proteinMPNN (without the ligand)
2) Structures are predicted with AlphaFold2
3) Ligand binding site is designed with LigandMPNN/FastRelax, or Rosetta FastDesign
4) Sequences surrounding the ligand pocket are diversified with LigandMPNN
5) Final designed sequences are predicted with AlphaFold2
6) Alphafold2-predicted models are relaxed with the ligand and analyzed

## Installation
### Dependencies

#### LigandMPNN and AlphaFold2
To download the LigandMPNN and AlphaFold2 (v2.3.2) repositories referenced in this pipeline run:
```
git submodule init
git submodule update
```

To download the model weight files for AlphaFold2 and proteinMPNN run this command:<br>
`bash get_af2_and_mpnn_model_params.sh`

If you already have downloaded the weights elsewhere on your system then please edit these scripts with appropriate paths:<br>
    proteinMPNN: `lib/LigandMPNN/mpnn_api.py` [lines 45-49]<br>
    AlphaFold2: `scripts/af2/AlphaFold2.py` [line 40]

#### RFdiffusionAA:
Download RFdiffusionAA from here: https://github.com/baker-laboratory/rf_diffusion_all_atom<br>
and follow its instructions.<br>
Make sure to provide a full path to the checkpoint file in this configuration file:<br>
`rf_diffusion_all_atom/config/inference/aa.yaml`

#### RFjoint inpainting (proteininpainting)
(Optional) Download RFjoint Inpainting here: https://github.com/RosettaCommons/RFDesign

Inpainting is used to further resample/diversify diffusion outputs, and it may also increase AF2 success rates.

### Python or Apptainer image
This pipeline consists of multiple different Python scripts using a different Python modules - most notably PyTorch, PyRosetta, Jax, Jaxlib, Tensorflow, Prody, OpenBabel.<br>
Separate conda environments for AlphaFold2 and RFdiffusionAA/ligandMPNN were used to test this pipeline, and the environment YML files are provided in `envs/`.


To create a conda environment capable of running RFdiffusionAA, LigandMPNN and PyRosetta, set it up as follows:<br>
`conda env create -f envs/diffusion.yml`


A minimal conda environment for AlphaFold2 is set up as follows:<br>
`conda env create -f envs/mlfold.yml`


### Executing the pipeline
Please adjust the the critical paths defined in the first couple of cells of the notebook based on your system configuration. Other than that, the pipeline is executed by running the cells and waiting for them to finish.

Certain tasks are configured to run as Slurm jobs on a compute cluster. The Slurm script setup is handled in `scripts/utils/utils.py` by the function `create_slurm_submit_script()`.
Please modify this script, and any references to it in the notebook, based on how your system accepts jobs.
