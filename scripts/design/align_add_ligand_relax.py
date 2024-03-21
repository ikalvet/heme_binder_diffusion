#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:47:06 2019

@author: ikalvet
"""


import pyrosetta as pyr
import pyrosetta.rosetta
import os
import sys
import numpy as np
import queue
import threading
import time
import random
import datetime
import pandas as pd
import json
import argparse
import pyrosetta.distributed.io
from pyrosetta.rosetta.protocols.grafting.simple_movers import DeleteRegionMover
from pyrosetta.rosetta.utility import vector1_numeric_xyzVector_double_t as xyzVector
from pyrosetta.rosetta.numeric import xyzMatrix_double_t
from pyrosetta.rosetta.numeric import xyzVector_double_t
from pyrosetta.rosetta.protocols.toolbox import superposition_transform
from pyrosetta.rosetta.protocols.toolbox import apply_superposition_transform
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.core.pose import append_subpose_to_pose
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
from pyrosetta.rosetta.core.select.residue_selector import ResidueIndexSelector
from pyrosetta.rosetta.core.select.residue_selector import NotResidueSelector
from pyrosetta.rosetta.core.select.residue_selector import AndResidueSelector


SCRIPT_PATH = os.path.dirname(__file__)
sys.path.append(f"{SCRIPT_PATH}/../utils")
import design_utils
import no_ligand_repack
import scoring_utils


def add_matcher_line_to_pose(pose, residues):
    _str = pyrosetta.distributed.io.to_pdbstring(pose)
    pdbff = _str.split("\n")
    ligand_name = pose.residue(pose.size()).name3()
    new_pdb = []
    if "ATOM" in pdbff[0]:
        for i, resno in enumerate(residues):
            new_pdb.append(f"REMARK 666 MATCH TEMPLATE X {ligand_name}    0 MATCH MOTIF A {pose.residue(resno).name3()}  {resno}  {i+1}  1               \n")
        for l in pdbff:
            new_pdb.append(l)
    else:
        for l in pdbff:
            if "HEADER" in l:
                new_pdb.append(l)
                for i, resno in enumerate(residues):
                    new_pdb.append(f"REMARK 666 MATCH TEMPLATE X {ligand_name}    0 MATCH MOTIF A {pose.residue(resno).name3()}  {resno}  {i+1}  1               \n")
            else:
                new_pdb.append(l)
    pose2 = pyrosetta.Pose()
    pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(pose2, "\n".join(new_pdb))
    return pose2



"""
Parsing user input
"""
# def main():
if __name__ == "__main__":

    arguments = sys.argv.copy()

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", type=str, required=True, help="Input PDB")
    parser.add_argument("--ref_pdb", type=str, required=True, help="Reference PDB from where the ligand is taken, and structure will be aligned to.")
    parser.add_argument("--cstfile", type=str, required=False, help="CST file used for matching?")
    parser.add_argument("--params", nargs="+", required=False, help="params files used for matching?")
    parser.add_argument("--ligand", type=str, required=True, help="Ligand name")
    parser.add_argument("--nstruct", type=int, default=1, help="How many relax iterations")
    parser.add_argument("--nproc", type=int, help="How many CPU cores")
    parser.add_argument("--suffix", type=str, help="Ligand name")
    parser.add_argument("--cartesian", action="store_true", default=False, help="Cartesian relax?")
    parser.add_argument("--outdir", type=str, default="./", help="Output directory")

    args = parser.parse_args()

    pdbfile = args.pdb

    suffix = ""
    if args.suffix is not None:
        suffix = args.suffix + "_"

    ligand = args.ligand

    cstfile = args.cstfile
    scorefilename = f"{args.outdir}/scorefile.txt"

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)


    if args.cstfile is not None:
        matched_residues = design_utils.get_matcher_residues(args.ref_pdb)
        his_seqpos = list(matched_residues.keys())[0]
    else:
        his_seqpos = None
        matched_residues = {}
    
    outname = f"{os.path.basename(pdbfile).replace('.pdb','')}_{ligand}_{suffix}rlx"

    """
    Getting PyRosetta started
    """
    extra_res_fa = '-extra_res_fa'
    for p in args.params:
        if os.path.exists(p):
            extra_res_fa += " {}".format(p)

    DAB = f"{SCRIPT_PATH}/../utils/DAlphaBall.gcc" # This binary was compiled on UW systems. It may or may not work correctly on yours
    assert os.path.exists(DAB), "Please compile DAlphaBall.gcc and manually provide a path to it in this script under the variable `DAB`\n"\
                            "For more info on DAlphaBall, visit: https://www.rosettacommons.org/docs/latest/scripting_documentation/RosettaScripts/Filters/HolesFilter"

    multithreading = ""
    if args.nproc is None:
        NPROC = None
        if "OMP_NUM_THREADS" in os.environ:
            NPROC = int(os.environ["OMP_NUM_THREADS"])
        if "SLURM_CPUS_ON_NODE" in os.environ:
            NPROC = int(os.environ["SLURM_CPUS_ON_NODE"])
        if NPROC is None:
            NPROC = os.cpu_count()
    else:
        NPROC = args.nproc

    print(f"Using {NPROC} threads")
    if NPROC > 1:
        multithreading = f"-multithreading true -multithreading:total_threads {NPROC} -multithreading:interaction_graph_threads {NPROC}"

    pyr.init(f"-dalphaball {DAB} -beta_nov16 {extra_res_fa} -run:preserve_header {multithreading}")


    """
    PyRosetta is now running
    """

    pyrosetta.rosetta.basic.options.set_integer_option("packing:ex1:level", 1)
    pyrosetta.rosetta.basic.options.set_integer_option("packing:ex2:level", 1)
    pyrosetta.rosetta.basic.options.set_integer_option("packing:ex3:level", 1)
    pyrosetta.rosetta.basic.options.set_integer_option("packing:ex4:level", 0)

    scorefxn = pyr.get_fa_scorefxn()

    pose = pyr.pose_from_file(pdbfile)
    ref_pose = pyr.pose_from_file(args.ref_pdb)

    pose2 = pose.clone()


    ## Aligning input PDB to reference PDB
    overlay_pos = pyrosetta.rosetta.utility.vector1_unsigned_long()
    for n in range(1, pose.size()+1):
        overlay_pos.append(n)

    rmsd = pyrosetta.rosetta.protocols.toolbox.pose_manipulation.superimpose_pose_on_subset_CA(pose2, ref_pose, overlay_pos, 0)

    print(f"Alignment RMSD = {rmsd:.3f}")


    ## Adding ligand to input pose
    pyrosetta.rosetta.core.pose.append_subpose_to_pose(pose2, ref_pose, ref_pose.size(), ref_pose.size(), new_chain=True)

    ### Adjusting the residue identities of catalytic residues
    ### In particular HIS tautomers
    if args.cstfile is not None:
        for r in matched_residues:
            if pose2.residue(r).name() != ref_pose.residue(r).name():
                mutres = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()
                mutres.set_res_name(ref_pose.residue(r).name())
                mutres.set_target(r)
                mutres.apply(pose2)


    # Creating a new pose with correct matcher info
    pose2 = add_matcher_line_to_pose(pose2, list(matched_residues.keys()))

    # Storing ligand info
    ligand_seqpos = pose2.size()
    ligand_name = pose2.residue(ligand_seqpos).name3()
    ligand_pose = pyrosetta.rosetta.core.pose.Pose()
    pyrosetta.rosetta.core.pose.append_subpose_to_pose(ligand_pose, pose2, pose2.size(), pose2.size(), new_chain=True)


    substrate_atom_names = {"HBA": [f"C{n}" for n in range(35, 48)] + ["O6"]}

    substrate_atomnos = None
    if ligand_name in substrate_atom_names.keys():
        substrate_atomnos = []
        for n in range(1, pose2.residue(ligand_seqpos).natoms()+1):
            if pose2.residue(ligand_seqpos).atom_name(n).strip() in substrate_atom_names[ligand_name]:
                substrate_atomnos.append(n)



    sfx = pyr.get_fa_scorefxn()
    sfx_cst = pyr.get_fa_scorefxn()

    if args.cstfile is not None:
        sfx_cst.set_weight(pyrosetta.rosetta.core.scoring.score_type_from_name("atom_pair_constraint"), 1.5)
        sfx_cst.set_weight(pyrosetta.rosetta.core.scoring.score_type_from_name("angle_constraint"), 1.0)
        sfx_cst.set_weight(pyrosetta.rosetta.core.scoring.score_type_from_name("dihedral_constraint"), 1.0)     
        cst_mover = design_utils.CSTs(args.cstfile, sfx_cst)


    ### Setting up FastRelax
    fastRelax = pyrosetta.rosetta.protocols.relax.FastRelax(sfx_cst, 1)
    fastRelax.constrain_relax_to_start_coords(True)
    
    if args.cartesian is True:
        fastRelax.cartesian(args.cartesian)
        sfx_cst.set_weight(pyrosetta.rosetta.core.scoring.score_type_from_name("cart_bonded"), 0.5)
        sfx_cst.set_weight(pyrosetta.rosetta.core.scoring.score_type_from_name("pro_close"), 0.0)
        fastRelax.set_scorefxn(sfx_cst)

    tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.IncludeCurrent())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.NoRepackDisulfides())
    e = pyrosetta.rosetta.core.pack.task.operation.ExtraRotamersGeneric()
    e.ex1(True)
    e.ex2(True)
    e.ex1aro(True)
    e.ex1_sample_level(pyrosetta.rosetta.core.pack.task.ExtraRotSample(1))
    e.ex2_sample_level(pyrosetta.rosetta.core.pack.task.ExtraRotSample(1))
    tf.push_back(e)
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.RestrictToRepacking())
    fastRelax.set_task_factory(tf)

    movemap = pyrosetta.rosetta.core.kinematics.MoveMap()
    movemap.set_jump(True)
    movemap.set_chi(True)
    movemap.set_bb(True)
    fastRelax.set_movemap(movemap)


    ### Performing FastRelax
    for N in range(args.nstruct):
        
        t0 = time.time()
        
        pose3 = pose2.clone()
        if args.cstfile is not None:
            cst_mover.add_cst(pose3)
        
        fastRelax.apply(pose3)


        try:
            df_scores = pd.DataFrame.from_records([pose3.scores])
        except:
            print("The protocol failed. See log.")
            sys.exit(1)

        output_pdb_iter = args.outdir + "/" + outname + f"_{N}.pdb"
        pose3.dump_pdb(output_pdb_iter)

        #========================================================================
        # Extra filters
        #========================================================================

        df_scores['description'] = outname + f"_{N}"

        df_scores['score_per_res'] = pose3.scores["total_score"] / pose3.size()

        if args.cstfile is not None:
            sfx_cst(pose3)
            df_scores['all_cst'] = sum([pose.scores[s] for s in pose.scores if "constraint" in s])


        if "contact_molecular_surface" in df_scores.keys():
            df_scores.at[0, "cms_per_atom"] = df_scores.at[0, "contact_molecular_surface"] / pose3.residue(ligand_seqpos).natoms()

        # Get the ligand ddg, without including serine-ligand repulsion
        from scoring_utils import calculate_ddg
        sf = pyr.get_fa_scorefxn()
        def fix_scorefxn(sfxn, allow_double_bb=False):
            opts = sfxn.energy_method_options()
            opts.hbond_options().decompose_bb_hb_into_pair_energies(True)
            opts.hbond_options().bb_donor_acceptor_check(not allow_double_bb)
            sfxn.set_energy_method_options(opts)
        fix_scorefxn(sf)

        df_scores['corrected_ddg'] = calculate_ddg(pose3, sf, his_seqpos)

        # Calculating relative ligand SASA
        # First figuring out what is the path to the ligand PDB file

        free_ligand_sasa = scoring_utils.getSASA(ligand_pose, resno=1)
        ligand_sasa = scoring_utils.getSASA(pose3, resno=ligand_seqpos)
        df_scores.at[0, 'L_SASA'] = ligand_sasa / free_ligand_sasa

        if substrate_atomnos is not None:
            df_scores.at[0, 'substrate_SASA'] = scoring_utils.getSASA(pose3, ligand_seqpos, substrate_atomnos)

        # Heme models only:
        if "FE" in [pose3.residue(ligand_seqpos).atom_type(n+1).element() for n in range(pose3.residue(ligand_seqpos).natoms())]:
            # Using a custom function to find HBond partners of the COO groups.
            # The XML implementation misses some interations it seems.
            for n in range(1, 5):
                df_scores.at[0, f"O{n}_hbond"] = scoring_utils.find_hbonds_to_residue_atom(pose3, ligand_seqpos, f"O{n}")


            # Checking if both COO groups on heme are hbonded
            if any([df_scores.at[0, x] > 0.0 for x in ['O1_hbond', 'O3_hbond']]) and any([df_scores.at[0, x] > 0.0 for x in ['O2_hbond', 'O4_hbond']]):
                df_scores.at[0, 'COO_hbond'] = 1.0
            else:
                df_scores.at[0, 'COO_hbond'] = 0.0

            cms = pyrosetta.rosetta.protocols.simple_filters.ContactMolecularSurfaceFilter()
            lig_sel = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(ligand_seqpos)
            protein_sel = pyrosetta.rosetta.core.select.residue_selector.ChainSelector("A")
            cms.use_rosetta_radii(True)
            cms.distance_weight(0.5)
            cms.selector1(protein_sel)
            cms.selector2(lig_sel)
            df_scores.at[0, "contact_molecular_surface"] =  cms.compute(pose3)  # ContactMolecularSurface
            df_scores.at[0, "cms_per_atom"] = df_scores.at[0, "contact_molecular_surface"] / pose3.residue(ligand_seqpos).natoms()


            ## Calculating shape complementarity
            sc = pyrosetta.rosetta.protocols.simple_filters.ShapeComplementarityFilter()
            sc.use_rosetta_radii(True)
            sc.selector1(protein_sel)
            sc.selector2(lig_sel)
            df_scores.at[0, "sc"] = sc.score(pose3)


            # Adding the constraints to the pose and using it to figure out
            # exactly which position belongs to which constraint
            cst_mover.add_cst(pose3)
            _pose3 = pose3.clone()
            target_resno = no_ligand_repack.get_target_residues_from_csts(_pose3)
            for i, resno in enumerate(target_resno):
                df_scores.at[0, f'SR{i+1}'] = resno

            # Calculating the no-ligand-repack as it's done in enzdes
            nlr_scores = no_ligand_repack.no_ligand_repack(_pose3, sfx,
                                                           target_residues=target_resno,
                                                           ligand_resno=ligand_seqpos)

            # Measuring the Heme-CYS/HIS angle because not all deviations
            # are reflected in the all_cst score
            heme_atoms = ["N1", "N2", "N3", "N4"]
            if pose3.residue(target_resno[0]).name3() == "CYX":
                target_atom = "SG"
            elif pose3.residue(target_resno[0]).name3() == "HIS":
                target_atom = "NE2"
            else:
                target_atom = None
    
            if target_atom is not None:
                angles = [scoring_utils.get_angle(pose3.residue(ligand_seqpos).xyz(nx),
                                                  pose3.residue(ligand_seqpos).xyz("FE1"),
                                                  pose3.residue(target_resno[0]).xyz(target_atom)) for nx in heme_atoms]
                df_scores.at[0, "heme_angle_wrst"] = min(angles)
    
            for k in nlr_scores.keys():
                df_scores.at[0, k] = nlr_scores.iloc[0][k]

        ## Calculating a bunch of rmsds
        pose4 = pose3.clone()
        rmsd2 = pyrosetta.rosetta.protocols.toolbox.pose_manipulation.superimpose_pose_on_subset_CA(pose4, ref_pose, overlay_pos, 0)
        
        df_scores.at[0, "rmsd_CA_in_ref"] = rmsd
        df_scores.at[0, "rmsd_CA_rlx_in"] = pyrosetta.rosetta.protocols.toolbox.pose_manipulation.superimpose_pose_on_subset_CA(pose4, pose2, overlay_pos, 0)
        df_scores.at[0, "rmsd_CA_rlx_ref"] = rmsd2

        print(df_scores.iloc[0])
        scoring_utils.dump_scorefile(df_scores, scorefilename)

        t1 = time.time()
        print(f"Relax iteration {N} took {(t1-t0):.3f} seconds.")
