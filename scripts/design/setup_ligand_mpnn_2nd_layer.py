#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 15:22:02 2022

@author: ikalvet
"""

import os, sys
import argparse
import json
import glob
import numpy as np
import multiprocessing
import pyrosetta as pyr
import pyrosetta.rosetta

SCRIPT_PATH = os.path.dirname(__file__)
sys.path.append(f"{SCRIPT_PATH}/../utils")
import design_utils
from pssm_tools import make_bias_pssm


ALPHABET = "ABCDEFGHIJKLMOPQRSTUVW"


def split_residues_to_chains(pose, ignore_residues):
    # TODO: currently not putting ligand in this list!!
    design_res_chains = {}
    for res in pose.residues:
        if res.seqpos() in ignore_residues:
            continue
        if res.is_ligand() is True:
            continue
        ch = pose.pdb_info().chain(res.seqpos())
        if ch not in design_res_chains.keys():
            design_res_chains[ch] = []
        design_res_chains[ch].append(res.seqpos())
    return design_res_chains


def get_residues_with_close_sc(pose, ref_atoms, residues=None, exclude_residues=None, cutoff=4.5):
    """
    """
    if residues is None:
        residues = [x for x in range(1, pose.size()+1)]
    if exclude_residues is None:
        exclude_residues = []

    close_ones = []
    for resno in residues:
        if resno in exclude_residues:
            continue
        if pose.residue(resno).is_ligand():
            continue
        res = pose.residue(resno)
        close_enough = False
        for atomno in range(1, res.natoms()):
            if res.atom_type(atomno).is_heavyatom():
                for ha in ref_atoms:
                    if (res.xyz(atomno) - pose.residue(pose.size()).xyz(ha)).norm() < cutoff:
                        close_enough = True
                        close_ones.append(resno)
                        break
                    if close_enough is True:
                        break
                if close_enough is True:
                    break
    return close_ones


def main(args):
    """
    For each input PDB, will find positions close to ligand and enable them for MPNN-design
    """
    if args.pdb is not None:
        pdbfiles = args.pdb
    elif args.pdblist is not None:
        pdbfiles = open(args.pdblist, 'r').readlines()
        extension = ".pdb"
        if pdbfiles[0].rstrip()[-4:] == extension:
            extension = ""
        pdbfiles = [f.rstrip() + extension for f in pdbfiles]
    else:
        pdbfiles = glob.glob("*.pdb")
    
    if len(pdbfiles) == 0:
        sys.exit("No PDB files")

    ligand_polar_bias_atoms = None
    if args.ligand_bias_atoms is not None:
        ligand_polar_bias_atoms = args.ligand_bias_atoms

    if args.bias_AAs is not None and ligand_polar_bias_atoms is None:
        print("Will NOT generate an AA bias JSON file. Please provide target ligand atoms with --ligand_bias_atoms")

    extra_res_fa = ""
    if args.params is not None:
        extra_res_fa = f'-extra_res_fa {" ".join(args.params)}'

    pyr.init(f"{extra_res_fa} -run:preserve_header -mute all")

    rotamer_jsonl = None
    if args.rotamer_jsonl is not None:
        rotamer_jsonl = open(args.rotamer_jsonl, "r").readlines()
        pass

    the_queue = multiprocessing.Queue()  # Queue stores the iterables

    manager = multiprocessing.Manager()
    masked_pos = manager.dict()
    pssm_dict_designs = manager.dict()
    for pdbfile in pdbfiles:
        the_queue.put(pdbfile)
        masked_pos[os.path.basename(pdbfile).replace(".pdb", "")] = {}
        if ligand_polar_bias_atoms is not None:
            pssm_dict_designs[os.path.basename(pdbfile).replace(".pdb", "")] = {}


    def process(q):
        while True:
            pdbfile = q.get(block=True)
            if pdbfile is None:
                return
            pdbname = os.path.basename(pdbfile)

            pose = pyr.pose_from_file(pdbfile)

            matched_residues = design_utils.get_matcher_residues(pdbfile)

            ligands = {}
            if args.ligand is None:
                for r in pose.residues:
                    if r.is_ligand() is True and r.is_virtual_residue() is False:
                        ligands[r.seqpos()] = r.name3()
            else:
                for r in pose.residues:
                    if r.name3() in args.ligand and r.is_ligand():
                        ligands[r.seqpos()] = r.name3()

            pocket_residues = list(matched_residues.keys()) + args.fix

            for lig_resno in ligands:
                heavyatoms = design_utils.get_ligand_heavyatoms(pose, lig_resno)

                SEL_mutate_residues, SEL_repack_residues, SEL_do_not_repack, residues\
                    = design_utils.get_layer_selections(pose, repack_only_pos=pocket_residues,
                                                        design_pos=[], ref_resno=lig_resno, heavyatoms=heavyatoms,
                                                        cuts=[args.dist_bb, args.dist_bb+2.0, args.dist_bb+4.0, args.dist_bb+6.0], design_GP=True)

                # Need to somehow pick pocket residues that have SC atoms close to the ligand.
                # Exclude from design: residues[0] and those that have SC atoms very close.
                close_ones = get_residues_with_close_sc(pose, heavyatoms, residues[1]+residues[2], exclude_residues=pocket_residues, cutoff=args.dist_sc)
                pocket_residues += residues[0] + close_ones

            pocket_residues = list(set(pocket_residues))

            if args.design_full is True:
                design_residues = [res.seqpos() for res in pose.residues if res.seqpos() not in pocket_residues and not res.is_ligand()]
            else:
                design_residues = [x for x in residues[0]+residues[1]+residues[2]+residues[3] if x not in pocket_residues]
                
                # Also including all Alanines that are not in the pocket.
                ala_positons = [res.seqpos() for res in pose.residues if res.seqpos() not in pocket_residues+design_residues and res.name3() == "ALA"]
                print("ALA positions", '+'.join([str(x) for x in ala_positons]))
                design_residues += ala_positons


            # Creating a PSSM if some positions need to be biased for more polar residues
            if ligand_polar_bias_atoms is not None:
                assert all([x in heavyatoms for x in ligand_polar_bias_atoms]), f"Some provided polar atoms ({ligand_polar_bias_atoms}) not found in ligand: {heavyatoms}"
                # polars = "GRKEDQNSTYWH"


                SEL_mutate_residues, SEL_repack_residues, SEL_do_not_repack, residues\
                    = design_utils.get_layer_selections(pose, repack_only_pos=list(matched_residues.keys()),
                                                        design_pos=[], ref_resno=lig_resno, heavyatoms=ligand_polar_bias_atoms,
                                                        cuts=[args.dist_bb, args.dist_bb+2.0, args.dist_bb+4.0, args.dist_bb+6.0], design_GP=True)


                polar_bias_residues = residues[0]
                close_ones = get_residues_with_close_sc(pose, ligand_polar_bias_atoms, residues[1]+residues[2], exclude_residues=polar_bias_residues, cutoff=args.dist_sc)
                polar_bias_residues += close_ones

                polar_res = '+'.join([str(x) for x in polar_bias_residues])
                print(f"{pdbname.replace('.pdb', '')}: {len(polar_bias_residues)} designable positions around target atom(s) {ligand_polar_bias_atoms}:\n   {polar_res}")

                design_residues = list(set(design_residues+polar_bias_residues+[x for x in residues[0]+residues[1] if pose.residue(x).name1() in args.bias_AAs]))

                if args.omit is False:
                    pssm_dict_designs[pdbname.replace(".pdb", "")] = make_bias_pssm(pose, polar_bias_residues, args.bias_AAs, args.bias_low, args.bias_high)
                else:
                    design_residues_ch = split_residues_to_chains(pose, [res.seqpos() for res in pose.residues if res.seqpos() not in polar_bias_residues and not res.is_ligand()])
                    pssm_dict_designs[pdbname.replace(".pdb", "")] = {ch: [[design_residues_ch[ch], args.bias_AAs]] for ch in design_residues_ch}


            design_res = '+'.join([str(x) for x in design_residues])
            pocket_res = '+'.join([str(x) for x in pocket_residues])
            print(f"{pdbname.replace('.pdb', '')}: {len(design_residues)} designable positions:\n"
                  f"   {design_res}\nPocket positions:\n   {pocket_res}")

            fixed_res_chains = split_residues_to_chains(pose, design_residues)
            
            if args.old_format is True:
                masked_pos[pdbname.replace(".pdb", "")] = fixed_res_chains
            else:
                masked_pos[pdbname.replace(".pdb", "")] = ' '.join([' '.join([f"{ch}{r}" for r in fixed_res_chains[ch]]) for ch in fixed_res_chains])

            # Adding fixed list and PSSM for each rotamer, if provided
            if rotamer_jsonl is not None:
                n_rot = 0
                for line in rotamer_jsonl:
                    parsed_rotamer = json.loads(line)
                    _rot_parent_name = "_".join(parsed_rotamer["name"].split("_")[:-1])
                    if pdbname.replace(".pdb", "") == _rot_parent_name:
                        n_rot += 1
                        masked_pos[parsed_rotamer["name"]] = fixed_res_chains

                        if ligand_polar_bias_atoms is not None:
                            pssm_dict_designs[parsed_rotamer["name"]] = pssm_dict_designs[pdbname.replace(".pdb", "")]
                print(f"Added fixed positions also for {n_rot} rotamers.")


    N_PROCESSES = os.cpu_count()-1

    pool = multiprocessing.Pool(processes=N_PROCESSES,
                                initializer=process,
                                initargs=(the_queue, ))

    # None to end each process
    for _i in range(N_PROCESSES):
        the_queue.put(None)

    # Closing the queue and the pool
    the_queue.close()
    the_queue.join_thread()
    pool.close()
    pool.join()
    
    masked_pos = dict(masked_pos)
    pssm_dict_designs = dict(pssm_dict_designs)


    print("Creating a JSON file specifing fixed positions: masked_pos.jsonl")
    with open(f"{args.output_path}", 'w') as f:
        f.write(json.dumps(masked_pos) + '\n')

    if ligand_polar_bias_atoms is not None:

        # Write PSSM output to:
        if args.omit is False:
            pssm_dict_fn = 'pssm_dict.jsonl'
        else:
            pssm_dict_fn = 'omit_AA_dict.jsonl'
        print(f"Creating a JSON file specifing fixed PSSM for polar bias: {pssm_dict_fn}")
        with open(pssm_dict_fn, 'w') as f:
            f.write(json.dumps(pssm_dict_designs) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", nargs="+", help="Input PDBs. If argument not given, all PDBs in working directory are taken.")
    parser.add_argument("--pdblist", type=str, help="File containing list of PDB filenames")
    parser.add_argument("--params", nargs="+", help="params files")
    parser.add_argument("--design_full", action="store_true", default=False, help="If used then all positions that are not immediately next to the ligand will be redesigned.\n"
                        "By default only the second layer is designed.")
    parser.add_argument("--fix", type=int, nargs="+", default=[], help="Positions that must be fixed. Same positions will be applied to all!")
    parser.add_argument("--ligand", nargs="+", help="Ligand name(s) around which positions will not be designed. By default all ligand will be considered. Use if you want to consider only a subset of ligands.")
    parser.add_argument("--dist_bb", type=float, default=5.5, help="CA - ligand heavyatom distance cutoff below which a residue will be fixed.")
    parser.add_argument("--dist_sc", type=float, default=4.5, help="sidechain heavyatom - ligand heavyatom distance cutoff below which a residue will be fixed.")
    parser.add_argument("--ligand_bias_atoms", nargs="+", help="Ligand atom names that require polar contacts. Heavyatoms only.")
    parser.add_argument("--bias_AAs", type=str, default="GRKEDQNSTYWH", help="AA1 that the designs will be biased for. Default = GRKEDQNSTYWH")
    parser.add_argument("--bias_low", type=float, default=-1.0, help="Log_odd low limit for bias. default = -1.0")
    parser.add_argument("--bias_high", type=float, default=1.39, help="Log_odd high limit for bias. default = 1.39")
    parser.add_argument("--rotamer_jsonl", type=str, help="JSONL file generate with parsed rotamers. Names in the JSONL must be <pdfile_r{n}>")
    parser.add_argument("--output_path", type=str, help="JSONL file generate with parsed rotamers. Names in the JSONL must be <pdfile_r{n}>")
    parser.add_argument("--omit", action="store_true", default=False, help="Should the bias AA's receive pssm bias or be omitted?")
    parser.add_argument("--old_format", action="store_true", default=False, help="The JSON file will be produced in the old MPNN style format with a per-chain dictionary.\n"
                                                                                 "{'pdbname': {'A': [resno1, resno2]}}")
    args = parser.parse_args()
    main(args)
