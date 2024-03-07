#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Indrek Kalvet
ikalvet@uw.edu
"""
import sys
import pyrosetta as pyr
import pyrosetta.rosetta
from pyrosetta.rosetta.core.select.residue_selector import ResidueIndexSelector
from pyrosetta.rosetta.core.select.residue_selector import NotResidueSelector
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
from pyrosetta.rosetta.protocols.grafting.simple_movers import DeleteRegionMover
import numpy as np
import pandas as pd
import design_utils


def separate_protein_and_ligand(pose, resno=None):
    """
    Separates the ligand and the rest of the pose to 'infinity'
    (to 666 angstrom).
    Assumes ligand is the last residue in the pose.
    Arguments:
        pose (object, pyrosetta.rosetta.core.pose.Pose)
    """
    assert isinstance(pose, pyrosetta.rosetta.core.pose.Pose), "pose: Invalid input type"

    tmp_pose = pose.clone()
    if resno is None:
        lig_seqpos = tmp_pose.size()
    else:
        lig_seqpos = resno
    assert tmp_pose.residue(lig_seqpos).is_ligand(), f"Residue {lig_seqpos} is not ligand!?"

    lig_jump_no = tmp_pose.fold_tree().get_jump_that_builds_residue(lig_seqpos)
    rbt = pyr.rosetta.protocols.rigid.RigidBodyTransMover(tmp_pose, lig_jump_no)
    rbt.step_size(666)
    rbt.apply(tmp_pose)
    return tmp_pose


def perform_no_ligand_repack(pose, scrfxn, repack_residues_list=None, outfile=None):
    """
    Performs repacking of an input pose.
    If repack_residues_list is provided then it generates a ResidueSelector
    out of that list, allowing only these residues to be repacked.
    All other residues are nto touched.
    Arguments:
        pose (object, pyrosetta.rosetta.core.pose.Pose)
        scrfxn (obj, pyrosetta.rosetta.core.scoring.ScoreFunction)
        repack_residues_list (list) :: list of integers
        outfile (str) :: name of the dumped PDB file
    """
    nt = type(None)
    assert isinstance(pose, pyrosetta.rosetta.core.pose.Pose), "pose: Invalid input type"
    assert isinstance(scrfxn, pyrosetta.rosetta.core.scoring.ScoreFunction), "scrfxn: Invalid input type"
    assert isinstance(repack_residues_list, (list, nt)), "repack_residues_list: Invalid input type"
    assert isinstance(outfile, (str, nt)), "outfile: Invalid input type"

    # Building a ResidueSelector out of a list of residue numbers
    # that should be repacked
    if repack_residues_list is not None:
        repack_residues = ResidueIndexSelector()
        for r in repack_residues_list:
            repack_residues.append_index(r)
        do_not_repack = NotResidueSelector(repack_residues)
    else:
        # If no list of resno's is provided, exit!
        sys.exit("Not implemented yet!")

    tmp_pose = pose.clone()

    # Now lets see how to set up task operations
    # The task factory accepts all the task operations
    tf = pyrosetta.rosetta.core.pack.task.TaskFactory()

    # These three are pretty standard
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.IncludeCurrent())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.NoRepackDisulfides())

    # Setting up residue-level extra rotamers. Not sure if better than standard?
    erg_RLT = pyrosetta.rosetta.core.pack.task.operation.ExtraRotamersGenericRLT()
    erg_RLT.ex1(True)
    erg_RLT.ex2(True)
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(erg_RLT, repack_residues))

    # We are not designing, so this will also be used
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.RestrictToRepacking())

    # disable design on the repack-only part
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(
        pyrosetta.rosetta.core.pack.task.operation.RestrictToRepackingRLT(), repack_residues))

    # don't repack the rest of the protein
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(
        pyrosetta.rosetta.core.pack.task.operation.PreventRepackingRLT(), do_not_repack, False))

    # Have to convert the task factory into a PackerTask
    task = tf.create_task_and_apply_taskoperations(tmp_pose)

    pack_mover = PackRotamersMover(scrfxn, task)
    pack_mover.apply(tmp_pose)

    if outfile is not None:
        tmp_pose.dump_pdb(outfile)

    return tmp_pose


def rmsd_no_super(pose1, pose2, residues):
    """
    This function is supposed to mimic the corresponding function found in Rosetta.
    """
    sum2 = 0.0
    natoms = 0
    for res in residues:
        num_atoms = pose1.residue(res).natoms()
        for atomno in range(1, num_atoms+1):
            diff = pose1.residue(res).xyz(atomno) - pose2.residue(res).xyz(atomno)
            sum2 += diff.length_squared()
            natoms += 1
    return np.sqrt(sum2/natoms)


def get_target_residues_from_csts(pose):
    assert pose.constraint_set().has_constraints(), "pose does not have constraints"
    obs = pyrosetta.rosetta.protocols.toolbox.match_enzdes_util.get_enzdes_observer(pose)
    cc = obs.cst_cache()
    
    targets = []
    for cst_no in range(1, cc.ncsts()+1):
        for cst in cc.param_cache(cst_no).active_pose_constraints():
            for resno in cst.residues():
                if resno in targets:
                    continue
                elif pose.residue(resno).is_ligand() or pose.residue(resno).is_virtual_residue():
                    continue
                else:
                    targets.append(resno)
    return targets


def no_ligand_repack(pose, scorefxn, target_residues=None, ligand_resno=None, dump_pdb=False):
    """
    Master function for performing the task.
    Arguments:
        pose (obj, ...)
        scorefxn (obj, ...)
        scoring_df (obj, pandas.DataFrame) :: dataframe where the scores will be stored
        dump_pdb (bool) :: (default=False), is the no-ligand-repacked pose dumped as PDB
    """
    nlr_df = pd.DataFrame()
    pdbfile = pose.pdb_info().name()

    if target_residues is None:
        if pose.constraint_set().has_constraints():
            target_residues = get_target_residues_from_csts(pose)
        else:
            target_residues = []


    # Getting the names of ligand heavyatoms (non-H)
    heavyatoms = design_utils.get_ligand_heavyatoms(pose, ligand_resno)

    # Finding out what residues belong to what layer, based on the CA distance
    # from ligand heavyatoms. Most of it is technically useless for this script,
    # but it was easier to just copy-paste it.
    SEL_mutate_residues, SEL_repack_residues, SEL_do_not_repack, residues = design_utils.get_layer_selections(pose, [], [], ref_resno=ligand_resno, heavyatoms=heavyatoms)

    # Getting no-ligand-repack residues.
    # Basically a list of all residues that are close-ish to the ligand.
    nlr_repack_residues = residues[0] + residues[1] + residues[2] + residues[3]

    print("No ligand repack repacked residues:")
    print("+".join([str(x) for x in nlr_repack_residues]))

    """
    Getting started with the repack
    """
    # Shooting the ligand to space
    pose3 = separate_protein_and_ligand(pose, ligand_resno)

    # Scoring the original and separated pose
    scorefxn(pose)
    scorefxn(pose3)

    # Removing constraints
    if pose3.constraint_set().has_constraints():
        pose3.constraint_set().clear()

    # Running no-ligand-repack
    nlr_pose = perform_no_ligand_repack(pose3, scorefxn,
                                        repack_residues_list=nlr_repack_residues)

    if dump_pdb is True:
        # Dumping the PDB
        # But first, removing the ligand form the pose
        nlr_pose_clone = nlr_pose.clone()
        drm = DeleteRegionMover()
        drm.region(str(pose.size()), str(pose.size()))
        drm.apply(nlr_pose_clone)
        nlr_pose_clone.dump_pdb(pdbfile.replace('.pdb', '_nlr.pdb'))

    # Calculating the nlr_rms of the whole protein
    nlr_rms = rmsd_no_super(pose3, nlr_pose, nlr_repack_residues)

    nlr_df.at[0, 'nlr_dE'] = nlr_pose.scores['total_score'] - pose.scores['total_score']
    nlr_df.at[0, 'nlr_totrms'] = nlr_rms

    if len(target_residues) != 0:
        # Calculating the nlr-rms of each target residue
        for i, resno in enumerate(target_residues):
            nlr_df.at[0, f'nlr_SR{i+1}_rms'] = rmsd_no_super(pose3, nlr_pose, [resno])
    return nlr_df
