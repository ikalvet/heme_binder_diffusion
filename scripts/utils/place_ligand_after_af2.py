#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 20:11:21 2022

@author: indrek
"""

import pyrosetta as pyr
import pyrosetta.rosetta
import pyrosetta.distributed.io
import glob
import os, sys
import pandas as pd
import numpy as np
import queue, threading
import multiprocessing
import argparse
import design_utils


def add_matcher_line_to_pose(pose, ref_pose, tgt_residues, ref_residues):
    """
    Takes REMARK 666 lines from ref pose and adjusts them based on the new positions in tgt_residues
    """
    if len(tgt_residues) == 0:
        return pose

    _str_ref = open(ref_pose.pdb_info().name(), "r").readlines()
    _ref_remarks = [l for l in _str_ref if "REMARK 666" in l]

    _str = pyrosetta.distributed.io.to_pdbstring(pose)
    pdbff = _str.split("\n")

    new_pdb = []
    if "ATOM" in pdbff[0]:
        for lr in _ref_remarks:
            new_pdb.append(lr)
        for l in pdbff:
            new_pdb.append(l)
    else:
        for l in pdbff:
            if "HEADER" in l:
                new_pdb.append(l)
                for lr in _ref_remarks:
                    new_pdb.append(lr)
            elif "REMARK 666" in l:  # Skipping existing REMARK 666 lines
                continue
            else:
                new_pdb.append(l)
    pose2 = pyrosetta.Pose()
    pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(pose2, "\n".join(new_pdb))
    return pose2



parser = argparse.ArgumentParser()
parser.add_argument("--pdb", nargs="+", help="Input PDBs. If argument not given, all PDBs in working directory are taken.")
parser.add_argument("--pdblist", type=str, help="File containing list of PDB filenames")
parser.add_argument("--params", nargs="+", help="params files")
parser.add_argument("--align_start", type=int, help="Start position of the alignment region in the reference PDB")
parser.add_argument("--align_end", type=int,  help="End position of the alignment region in the reference PDB")
parser.add_argument("--ref", required=True, nargs="+", type=str, help="Reference PDB(s).")
parser.add_argument("--outdir", type=str, default="./", help="Output directory")
parser.add_argument("--clobber", action="store_true", default=False, help="Overwrite existing files?")
parser.add_argument("--fix_catres", action="store_true", default=False, help="Fix the rotamer of catalytic residue?")


args = parser.parse_args()

if args.pdb is not None:
    pdbfiles = args.pdb
elif args.pdblist is not None:
    pdbfiles = open(args.pdblist, 'r').readlines()
    extension = ".pdb"
    if pdbfiles[0][-4:] == extension:
        extension = ""
    pdbfiles = [f.rstrip() + extension for f in pdbfiles]
else:
    pdbfiles = glob.glob("*.pdb")


extra_res = " ".join(args.params)
pyr.init(f"-extra_res_fa {extra_res} -mute all -run:preserve_header")

if not os.path.exists(args.outdir):
    os.mkdir(args.outdir)


ref_poses = {}

ref_path = os.path.dirname(args.ref[0])

the_queue = multiprocessing.Queue()  # Queue stores the iterables
manager = multiprocessing.Manager()

ref_poses = manager.dict()

for i, pdbfile in enumerate(pdbfiles):
    the_queue.put((i, pdbfile))

def process(q):
    while True:
        p= q.get(block=True)
        if p is None:
            return
        i, pdbfile = p[0], p[1]

        pose = pyr.pose_from_file(pdbfile)

        ref_names = []
        for r in args.ref:
            if os.path.basename(r).replace(".pdb", "_native") in pdbfile or os.path.basename(r).replace(".pdb", "_T") in pdbfile:
                ref_names.append(r)

        if len(ref_names) != 1:
            print(f"Can't find ref structure for {pdbfile}: {ref_names}")
            continue

        ref_name = ref_names[0]

        if ref_name not in ref_poses.keys():
            ref_poses[ref_name] = pyr.pose_from_file(ref_name)

        overlay_pos = pyrosetta.rosetta.utility.vector1_unsigned_long()
        for n in range(1, pose.size()+1):
            overlay_pos.append(n)

        matched_residues = design_utils.get_matcher_residues(ref_name)

        offset = 0

        pose2 = pose.clone()

        rmsd = pyrosetta.rosetta.protocols.toolbox.pose_manipulation.superimpose_pose_on_subset_CA(pose2, ref_poses[ref_name], overlay_pos, offset)
        print(f"{pdbfile}: alignment RMSD = {rmsd:.3f}")

        pyrosetta.rosetta.core.pose.append_subpose_to_pose(pose2, ref_poses[ref_name], ref_poses[ref_name].size(), ref_poses[ref_name].size(), True)

        for j, catres_seqpos in enumerate(matched_residues):
            catres_AA = ref_poses[ref_name].residue(catres_seqpos).name()
            catres_AA3 = ref_poses[ref_name].residue(catres_seqpos).name3()

            # Fixing catalytic residue tautomer to be the same as in the reference
            print(f"{pdbfile}: fixing {catres_AA3}{catres_seqpos} with reference {catres_AA}")
            mutres = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()
            mutres.set_res_name(catres_AA)  # fixes HIS_D as well
            mutres.set_target(catres_seqpos)
            mutres.apply(pose2)

            # Fixing catalytic residue rotamer to be the same as in the reference
            if args.fix_catres is True:
                # Fixing catalytic residue rotamer
                for n in range(ref_poses[ref_name].residue(catres_seqpos).nchi()):
                    pose2.residue(catres_seqpos).set_chi(n+1, ref_poses[ref_name].residue(catres_seqpos).chi(n+1))


        pose2 = add_matcher_line_to_pose(pose2, ref_poses[ref_name], matched_residues, matched_residues)
        
        ligand_name = pose2.residue(pose2.size()).name3()
        
        save_name = f"{args.outdir}/{os.path.basename(pdbfile).replace('.pdb', f'_{ligand_name}.pdb')}"
        if os.path.exists(save_name) and args.clobber is False:
            print("Warning! file exists! Use --clobber to overwrite it.")
            continue
        pose2.dump_pdb(save_name)

pool = multiprocessing.Pool(os.cpu_count(), process, (the_queue, ))

# None to end each process
for _i in range(os.cpu_count()):
    the_queue.put(None)

# Closing the queue and the pool
the_queue.close()
the_queue.join_thread()
pool.close()
pool.join()

