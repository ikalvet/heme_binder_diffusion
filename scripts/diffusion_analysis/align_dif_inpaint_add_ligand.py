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
import os, sys, copy
import pandas as pd
import numpy as np
import queue, threading
import argparse
import multiprocessing


def get_matcher_residues(filename):
    pdbfile = open(filename, 'r').readlines()

    matches = {}
    for line in pdbfile:
        if "ATOM" in line:
            break
        if "REMARK 666" in line:
            lspl = line.split()
            resno = int(lspl[11])

            matches[resno] = {'target_name': lspl[5],
                              'target_chain': lspl[4],
                              'target_resno': int(lspl[6]),
                              'chain': lspl[9],
                              'name3': lspl[10],
                              'cst_no': int(lspl[12]),
                              'cst_no_var': int(lspl[13])}
    return matches


def add_matcher_line_to_pose(pose, ref_pose, tgt_residues, ref_residues):
    """
    Takes REMARK 666 lines from ref pose and adjusts them based on the new positions in tgt_residues
    """
    if len(tgt_residues) == 0:
        return pose

    _new_remarks = []

    for i, r in enumerate(tgt_residues):
        _new_remarks.append(f"REMARK 666 MATCH TEMPLATE {tgt_residues[r]['target_chain']} {tgt_residues[r]['target_name']}"
                            f"  {tgt_residues[r]['target_resno']:>3} MATCH MOTIF {tgt_residues[r]['chain']} "
                            f"{tgt_residues[r]['name3']}  {r:>3}  {tgt_residues[r]['cst_no']}  "
                            f"{tgt_residues[r]['cst_no_var']}               \n")

    _str = pyrosetta.distributed.io.to_pdbstring(pose)
    pdbff = _str.split("\n")

    new_pdb = []
    if "ATOM" in pdbff[0]:
        for lr in _new_remarks:
            new_pdb.append(lr)
        for l in pdbff:
            new_pdb.append(l)
    else:
        for l in pdbff:
            if "HEADER" in l:
                new_pdb.append(l)
                for lr in _new_remarks:
                    new_pdb.append(lr)
            elif "REMARK 666" in l:  # Skipping existing REMARK 666 lines
                continue
            else:
                new_pdb.append(l)
    pose2 = pyrosetta.Pose()
    pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(pose2, "\n".join(new_pdb))
    return pose2



def dump_scorefile(df, filename):
    widths = {}
    for k in df.keys():
        if k in ["SCORE:", "description", "name"]:
            widths[k] = 0
        if k == 'hbnet_residues':
            widths[k] = 40
        elif len(k) >= 12:
            widths[k] = len(k) + 1
        else:
            widths[k] = 12

    with open(filename, "w") as file:
        title = ""
        for k in df.keys():
            if k == "SCORE:":
                title += k
            elif k in ["description", "name"]:
                continue
            else:
                title += f"{k:>{widths[k]}}"
        if all([t not in df.keys() for t in ["description", "name"]]):
            title += f" {'description'}"
        elif "description" in df.keys():
            title += f" {'description'}"
        elif "name" in df.keys():
            title += f" {'name'}"
        file.write(title + "\n")

        for index, row in df.iterrows():
            line = ""
            for k in df.keys():
                if isinstance(row[k], (float, np.float16)):
                    val = f"{row[k]:.3f}"
                else:
                    val = row[k]
                if k == "SCORE:":
                    line += val
                elif k in ["description", "name"]:
                    continue
                else:
                    line += f"{val:>{widths[k]}}"

            if all([t not in df.keys() for t in ["description", "name"]]):
                line += f" {index}"
            elif "description" in df.keys():
                line += f" {row['description']}"
            elif "name" in df.keys():
                line += f" {row['name']}"
            file.write(line + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", nargs="+", help="Input PDBs. If argument not given, all PDBs in working directory are taken.")
    parser.add_argument("--pdblist", type=str, help="File containing list of PDB filenames")
    parser.add_argument("--params", nargs="+", help="params files")
    parser.add_argument("--ref", required=True, nargs="+", type=str, help="Reference PDB(s) from diffusion")
    parser.add_argument("--ref_trb", nargs="+", type=str, help="TRB file(s) of diffusion outputs. Used for finding catalytic residues if they can't be parsed from the pdb REMARK 666 line(s).")
    parser.add_argument("--outdir", type=str, default="./", help="Output directory")
    parser.add_argument("--clobber", action="store_true", default=False, help="Overwrite existing files?")
    parser.add_argument("--clash", action="store_true", default=False, help="Clashcheck and keep only non-clashing structures")
    parser.add_argument("--nproc", type=int, help="Number of processes")


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
        sys.exit(1)


    extra_res = " ".join(args.params)
    pyr.init(f"-extra_res_fa {extra_res} -run:preserve_header -mute all")


    ref_poses = {}


    the_queue = multiprocessing.Queue()  # Queue stores the iterables

    manager = multiprocessing.Manager() 
    ref_poses = manager.dict()  # Need a special dictionary to store outputs from multiple processes


    # q = queue.Queue()
    for i, pdbfile in enumerate(pdbfiles):
        the_queue.put((i, pdbfile))


    def process(q, ref_poses):
        while True:
            p = q.get(block=True)
            if p is None:
                return
            i, pdbfile = p[0], p[1]

            print(f"{pdbfile}: processing")
            if os.path.exists(f"{args.outdir}/{os.path.basename(pdbfile)}") and args.clobber is False:
                print("Warning! file exists! Use --clobber to overwrite it.")
                continue

            pose = pyr.pose_from_file(pdbfile)

            # print(f"{pdbfile}: loading trb")
            trb = np.load(pdbfile.replace(".pdb", ".trb"), allow_pickle=True)

            __rs = [r for r in args.ref if os.path.basename(r).replace(".pdb", "_") in os.path.basename(pdbfile)]
            if "traj" in pdbfile:
                __rs = [r for r in __rs if "traj" in r]

            if len(__rs) != 1 and len(args.ref) > 1:
                print(f"{pdbfile}: can't find ref structure: {__rs}")
                sys.exit(1)
            elif len(__rs) == 1:
                ref_pdb = __rs[0]
            elif len(args.ref) == 1 and len(__rs) == 0:
                ref_pdb = args.ref[0]
            else:
                print("Unexpected situation with figuring out reference structure.")
                sys.exit(1)

            if ref_pdb not in ref_poses.keys():
                ref_poses[ref_pdb] = pyr.pose_from_file(ref_pdb)
            ref_pose = ref_poses[ref_pdb].clone()

            matched_residues = get_matcher_residues(ref_pdb)
            if len(matched_residues) == 0:
                print(f"{pdbfile}: did not find catalytic residue information form REMARK 666, trying to use diffusion TRB instead")
                try:
                    ref_trb = np.load(ref_pdb.replace(".pdb", ".trb"), allow_pickle=True)
                except FileNotFoundError:
                    assert args.ref_trb is not None, f"{pdbfile}: could not find catalytic/fixed residues from REMARK 666 and no diffusion trb has been provided."
                    __ref_trbs = [x for x in args.ref_trb if os.path.basename(ref_pdb).replace(".pdb", ".trb") in x]
                    assert len(__ref_trbs) == 1, f"{pdbfile}: Unable to find correct reference TRB file: {__ref_trbs}"
                    ref_trb = np.load(__ref_trbs[0], allow_pickle=True)

                matched_residues = {}
                # print(ref_trb["con_hal_pdb_idx"])
                for r in ref_trb["con_hal_pdb_idx"]:
                    if ref_trb["inpaint_seq"][r[1]-1] == True:
                        matched_residues[r[1]] = {"name3": ref_pose.residue(r[1]).name3(),
                                                  "chain": r[0]}
                # print(f"{pdbfile}: found catalytic/fixed residues: {matched_residues}")

            ### Aligning regions that were fixed during inpainting ###
            ### Code snippet from Kiera Sumida ###
            align_map = pyrosetta.rosetta.std.map_core_id_AtomID_core_id_AtomID()
            aln_atoms = ['N', 'CA', 'C', 'O']
            
            align_positions = []
            if "con_ref_idx0" in trb.keys() and "con_hal_idx0" in trb.keys():
                bb_inpaint = True
                for template_i, target_i in zip(trb["con_ref_idx0"], trb["con_hal_idx0"]):
                    align_positions.append((template_i, target_i))
            else:
                bb_inpaint = False
                for n in range(pose.size()):
                    if pose.residue(n+1).is_ligand():
                        continue
                    align_positions.append((n, n))

            for (template_i, target_i) in align_positions:
                if bb_inpaint is True and trb["inpaint_seq"][target_i] == False:
                    continue
                res_template_i = ref_pose.residue(template_i+1)
                res_target_i = pose.residue(target_i+1)
                for n in aln_atoms:
                    template_atom_idx = res_template_i.atom_index(n)
                    atom_id_template = pyrosetta.rosetta.core.id.AtomID(template_atom_idx, template_i+1)
                    target_atom_idx = res_target_i.atom_index(n)
                    atom_id_target = pyrosetta.rosetta.core.id.AtomID(target_atom_idx, target_i+1)
                    align_map[atom_id_target] = atom_id_template

            pose2 = pose.clone()

            try:
                rmsd = pyrosetta.rosetta.core.scoring.superimpose_pose(pose2, ref_pose, align_map)
            except RuntimeError:
                print(align_map)
                print(trb["con_ref_idx0"])
                print(trb["con_hal_idx0"])
                sys.exit("Issue with aligning")

            print(f"{pdbfile}: aligned to {ref_pdb} with RMSD = {rmsd:.3f}")
            
            if rmsd > 1.5:
                print("Alignment RMSD > 1.5 - attempting to align by excluding positions that do not align well")
                align_positions2 = []
                for (r1, r2) in align_positions:
                    if (pose2.residue(r2).xyz("CA") - ref_pose.residue(r1).xyz("CA")).norm() < 3.0:
                        align_positions2.append((r1, r2))

                align_map = pyrosetta.rosetta.std.map_core_id_AtomID_core_id_AtomID()
                for (template_i, target_i) in align_positions2:
                    if bb_inpaint is True and trb["inpaint_seq"][target_i] == False:
                        continue
                    res_template_i = ref_pose.residue(template_i+1)
                    res_target_i = pose.residue(target_i+1)
                    for n in aln_atoms:
                        template_atom_idx = res_template_i.atom_index(n)
                        atom_id_template = pyrosetta.rosetta.core.id.AtomID(template_atom_idx, template_i+1)
                        target_atom_idx = res_target_i.atom_index(n)
                        atom_id_target = pyrosetta.rosetta.core.id.AtomID(target_atom_idx, target_i+1)
                        align_map[atom_id_target] = atom_id_template

                rmsd = pyrosetta.rosetta.core.scoring.superimpose_pose(pose2, ref_pose, align_map)

                print(f"{pdbfile}: aligned again to {ref_pdb} with RMSD = {rmsd:.3f}")
            
            matched_residues_in_design = {}
            for r in matched_residues:
                if bb_inpaint is True:
                    for i, res in enumerate(trb["con_ref_pdb_idx"]):    
                        if res == (matched_residues[r]["chain"], np.int64(r)):
                            resno_in_design = trb["con_hal_pdb_idx"][i][1]
                            matched_residues_in_design[resno_in_design] = copy.deepcopy(matched_residues[r])
                            matched_residues_in_design[resno_in_design]["chain"] = trb["con_hal_pdb_idx"][i][0]
                            # Adjusting target residue number if it's not ligand. In case of an upstream match
                            tgt_resno_orig = matched_residues_in_design[resno_in_design]["target_resno"]
                            if tgt_resno_orig != 0 and pose2.residue(tgt_resno_orig).is_protein()\
                                and (matched_residues_in_design[resno_in_design]["target_chain"], np.int64(tgt_resno_orig)) not in trb["con_hal_pdb_idx"]:
                                    (_ch, _rn) = trb["con_hal_pdb_idx"][trb["con_ref_pdb_idx"].index((matched_residues[r]["target_chain"],
                                                                                                      np.int64(matched_residues[r]["target_resno"])))]
                                    matched_residues_in_design[resno_in_design]["target_chain"] = _ch
                                    matched_residues_in_design[resno_in_design]["target_resno"] = _rn
                            break
                else:
                    matched_residues_in_design[r] = copy.deepcopy(matched_residues[r])



            ### Fixing catalytic residues
            # print(f"{pdbfile}: fixing catalytic residues")
            for i, ref_catres in enumerate(matched_residues.keys()):
                if bb_inpaint is True:
                    tgt_catres = trb["con_hal_idx0"][trb["con_ref_idx0"].tolist().index(ref_catres-1)]+1
                else:
                    tgt_catres = ref_catres

                # print(f"{pdbfile}: adjusting catalytic residue {pose2.residue(tgt_catres).name3()}{tgt_catres} based on reference residue {ref_pose.residue(ref_catres).name3()}{ref_catres}")
                mutres = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()
                mutres.set_res_name(ref_pose.residue(ref_catres).name())
                mutres.set_target(tgt_catres)
                mutres.apply(pose2)
        
                for n in range(ref_pose.residue(ref_catres).nchi()):
                    pose2.residue(tgt_catres).set_chi(n+1, ref_pose.residue(ref_catres).chi(n+1))
        
        
            # print(f"{pdbfile}: adding ligand to pose")
            ### Adding ligand to pose ###
            pyrosetta.rosetta.core.pose.append_subpose_to_pose(pose2, ref_pose, ref_pose.size(), ref_pose.size(), True)
        
            ligand_seqpos = pose2.size()
            assert pose2.residue(ligand_seqpos).is_ligand()
            ligand = pose2.residue(ligand_seqpos)

            ### Performing clashcheck between the ligand and the protein
            if args.clash is True:
                ignore_HAs = []  # list of heavyatom names that will be ignored during clashcheck
                heavyatoms = design_utils.get_ligand_heavyatoms(pose2, pose2.size())

                heavyatoms = []
                for n in range(1, ligand.natoms() + 1):
                    if ligand.atom_type(n).element() != 'H':
                        heavyatoms.append(ligand.atom_name(n).lstrip().rstrip())

                clash = False
                for res in pose2.residues:
                    if res.is_ligand():
                        continue
                    if (res.xyz("CA") - ligand.nbr_atom_xyz()).norm() > 15.0:
                        continue
                    for ha in heavyatoms:
                        if ha in ignore_HAs:
                            continue
                        for bbatm in res.all_bb_atoms():
                            if (ligand.xyz(ha) - res.xyz(bbatm)).norm() < 3.0:
                                clash = True
                                break
                        if clash is True:
                            break
                    if clash is True:
                        break
                if clash is True:
                    print(f"{pdbfile}: CLASH!")
                    continue
        
            pose2 = add_matcher_line_to_pose(pose2, ref_pose, matched_residues_in_design, matched_residues)
        
            if os.path.exists(f"{args.outdir}/{os.path.basename(pdbfile)}") and args.clobber is False:
                print("Warning! file exists! Use --clobber to overwrite it.")
                pass
            pose2.dump_pdb(f"{args.outdir}/{os.path.basename(pdbfile)}")


    if args.nproc is None:
        NPROC = os.cpu_count()
    else:
        NPROC = args.nproc

    pool = multiprocessing.Pool(processes=NPROC,
                                initializer=process,
                                initargs=(the_queue, ref_poses, ))

    # None to end each process
    for _i in range(NPROC):
        the_queue.put(None)

    # Closing the queue and the pool
    the_queue.close()
    the_queue.join_thread()
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()
