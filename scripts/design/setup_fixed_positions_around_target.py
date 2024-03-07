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
import queue, threading
import pyrosetta as pyr
import pyrosetta.rosetta


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


def pose_numbering_to_pdb(pose, residues):
    """
    Takes a per-chain dictionary of residue numbers
    {"A": [k, l, m], "B": [n, o ,p]}
    and returns the same dictonary with PDB numbering
    """
    # TODO: currently not putting ligand in this list!!
    design_res_chains = {}
    for ch in residues:
        design_res_chains[ch] = []
        for resno in residues[ch]:
            for n in range(1, len(pose.chain_sequence(pose.chain(resno)))):
                _r = pose.pdb_rsd((ch, n))
                if _r.seqpos() == resno and _r.name3() == pose.residue(resno).name3():
                    design_res_chains[ch].append(n)
                    break
    return design_res_chains


def get_residues_with_close_sc(pose, target_resno, target_atoms, residues=None, cutoff_sc=None, exclude_residues=None):
    """
    """
    if residues is None:
        residues = [x for x in range(1, pose.size()+1)]
    if exclude_residues is None:
        exclude_residues = []

    if cutoff_sc is None:
        cutoff_sc = 4.5

    target_res = pose.residue(target_resno)
    close_ones = []
    for resno in residues:
        if resno in exclude_residues:
            continue
        if pose.residue(resno).is_ligand():
            continue
        if (pose.residue(resno).nbr_atom_xyz() - target_res.nbr_atom_xyz()).norm() > 20.0:
            continue
        res = pose.residue(resno)
        close_enough = False
        for atomno in range(1, res.natoms()):
            if not res.atom_type(atomno).is_heavyatom():
                continue
            for ha in target_atoms:
                if (res.xyz(atomno) - target_res.xyz(ha)).norm() < cutoff_sc:
                    close_enough = True
                    close_ones.append(resno)
                    break
                if close_enough is True:
                    break
            if close_enough is True:
                break
    return close_ones


def find_target_heavyatoms(pose, target_resno):
    tgt_res = pose.residue(target_resno)
    print(f"Finding target heavyatoms for residue {target_resno}")
    target_heavyatoms = []
    for n in range(1, tgt_res.natoms()+1):
        if tgt_res.atom_type(n).is_heavyatom():
            target_heavyatoms.append(n)
    return target_heavyatoms


def get_packer_layers(pose, ref_resno, cuts, target_atoms, do_not_design=None, allow_design=None, design_GP=False):
    """
    Finds residues that are within certain distances from target atoms, defined though <cuts>.
    Returns a list of lists where each embedded list contains residue numbers belongin to that layer.
    Last list contains all other residue numbers that were left over.
    get_packer_layers(pose, target_atoms, do_not_design, cuts) -> list
    Arguments:
        pose (object, pyrosetta.rosetta.core.pose.Pose)
        target_atoms (list) :: list of integers.
                               Atom numbers of a residue or a ligand that are used to calculate residue distances.
        do_not_design (list) :: list of integers. Residue numbers that are not allowed to be designed.
                                Used to override layering to not design matched residues.
        cuts (list) :: list of floats.
        design_GP (bool) :: if True, allows redesign of GLY and PRO
    """
    assert len(cuts) > 3, f"Not enough layer cut distances defined {cuts}"

    if do_not_design is None:
        do_not_design = []
    if allow_design is None:
        allow_design = []

    KEEP_RES = ["GLY", "PRO"]
    if design_GP is True:
        KEEP_RES = []

    residues = []
    ligand_atoms = []
    if isinstance(target_atoms, list):
        for a in target_atoms:
            ligand_atoms.append(pose.residue(ref_resno).xyz(a))
    if isinstance(target_atoms, dict):
        for k in target_atoms:
            for a in target_atoms[k]:
                ligand_atoms.append(pose.residue(k).xyz(a))

    residues = [[] for i in cuts] + [[]]

    for resno in range(1, pose.size()):
        if pose.residue(resno).is_ligand():
            continue
        if pose.residue(resno).is_virtual_residue():
            continue
        if resno in do_not_design:
            # do_not_design residues are added to repack layer
            residues[2].append(resno)
            continue
        if resno in allow_design:
            # allow_design residues are added to design layer
            residues[0].append(resno)
            continue
        resname = pose.residue(resno).name3()
        CA = pose.residue(resno).xyz('CA')
        CA_distances = []

        if 'GLY' not in resname:
            CB = pose.residue(resno).xyz('CB')
            CB_distances = []

        for a in ligand_atoms:
            CA_distances.append((a - CA).norm())
            if 'GLY' not in resname:
                CB_distances.append((a - CB).norm())
        CA_mindist = min(CA_distances)

        if 'GLY' not in resname:
            CB_mindist = min(CB_distances)

        # Figuring out into which cut that residue belongs to,
        # based on the smallest CA distance and whether the CA is further away from the ligand than CB or not.
        # PRO and GLY are disallowed form the first two cuts that would allow design,
        # they can only be repacked.

        if CA_mindist <= cuts[0] and resname not in KEEP_RES:
            residues[0].append(resno)
        elif CA_mindist <= cuts[1] and resname not in KEEP_RES:
            if resname == "GLY":
                residues[1].append(resno)
            elif CB_mindist < CA_mindist:
                residues[1].append(resno)
            elif CA_mindist < cuts[1]-1.0 and CB_mindist < cuts[1]-1.0:
                # Special case when the residue is close, but CB is pointing slightly away.
                residues[1].append(resno)
            else:
                residues[2].append(resno)
        elif CA_mindist <= cuts[2]:
            residues[2].append(resno)
        elif CA_mindist <= cuts[3] and resname not in KEEP_RES:
            if resname == "GLY":
                residues[3].append(resno)
            elif CB_mindist < CA_mindist:
                residues[3].append(resno)
            else:
                residues[-1].append(resno)
        else:
            residues[-1].append(resno)

    return residues


def find_ligand_resnos(pose):
    tgt_residues = []
    for r in pose.residues:
        if r.is_ligand() is True and r.is_virtual_residue() is False:
            tgt_residues.append(r.seqpos())
    return tgt_residues


def get_fixed_positions(pdbfile=None, pose=None, target_resno=None, cutoff_CA=None, cutoff_sc=None, return_as_list=False):
    """
    Figures out which positions around a target residue should be fixed in MPNN design

    Parameters
    ----------
    pdbfile : str, optional
        Path to PDB file
    pose : pyrosetta.rosetta.core.pose.Pose, optional
        Rosetta pose
    target_resno : int or list, optional
        DESCRIPTION. The default is None.
    cutoff_CA : float, optional
        Cutoff for residue CA and any target heavyatom distance. If below cutoff then will be fixed.
        default = 5.5
    cutoff_sc : float, optional
        Cutoff for any residue sc-atom and any target heavyatom distance. If below cutoff then will be fixed.
        default = 4.5

    Returns
    -------
    fixed_res_chains : dict
        Dictionary of fixed reside numbers, split into chains.
    """
    assert isinstance(pdbfile, (type(None), str))
    assert isinstance(target_resno, (type(None), int, list))
    assert not all([x is None for x in [pdbfile, pose]]), "Must provide either path to pdbfile, or a pose object as input"

    if cutoff_CA is None:
        cutoff_CA = 5.5

    cuts = [cutoff_CA, cutoff_CA+2.5, cutoff_CA+4.5, cutoff_CA+6.5]

    if isinstance(target_resno, int):
        target_resno = [target_resno]

    pdbname = "pose"
    if pdbfile is not None:
        pdbname = os.path.basename(pdbfile)
    else:
        if pose.pdb_info() is not None:
            pdbname = os.path.basename(pose.pdb_info().name())

    if pose is None:
        pose = pyr.pose_from_file(pdbfile)

    tgt_residues = []
    if target_resno is None:
        for r in pose.residues:
            if r.is_ligand() is True and r.is_virtual_residue() is False:
                tgt_residues.append(r.seqpos())
    else:
        for r in pose.residues:
            if r.seqpos() in target_resno:
                if not r.is_ligand():
                    print(f"Warning! residue {r.name3()}-{r.seqpos()} is not a ligand! I hope this is intentional.")
                tgt_residues.append(r.seqpos())

    pocket_residues = []
    for tgt_resno in tgt_residues:
        heavyatoms = find_target_heavyatoms(pose, tgt_resno)

        residues = get_packer_layers(pose, tgt_resno, cuts=cuts, target_atoms=heavyatoms, design_GP=True)

        # Need to somehow pick pocket residues that have SC atoms close to the ligand.
        # Exclude from design: residues[0] and those that have SC atoms very close.
        close_ones = get_residues_with_close_sc(pose, tgt_resno, heavyatoms,
                                                cutoff_sc=cutoff_sc, exclude_residues=[])
        pocket_residues += residues[0] + close_ones

    pocket_residues = list(set(pocket_residues))

    design_residues = [res.seqpos() for res in pose.residues if res.seqpos() not in pocket_residues and not res.is_ligand()]

    design_res = '+'.join([str(x) for x in design_residues])
    pocket_res = '+'.join([str(x) for x in pocket_residues])
    print(f"{pdbname.replace('.pdb', '')}: {len(design_residues)} Non-pocket positions:\n"
          f"   {design_res}\nPocket positions:\n   {pocket_res}")

    if return_as_list is False:
        fixed_res_chains = split_residues_to_chains(pose, design_residues)
        return fixed_res_chains
    else:
        return design_residues


def write_json(dictionary, filename):
    """
    Writes dictionary into a JSON file
    """
    with open(filename, 'w') as f:
        f.write(json.dumps(dictionary) + '\n')


def main(args):
    """
    For each input PDB, will find positions close to ligand and enable them for MPNN-design
    """
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
    
    if len(pdbfiles) == 0:
        sys.exit("No PDB files")

    ligand_polar_bias_atoms = None
    if args.ligand_bias_atoms is not None:
        ligand_polar_bias_atoms = args.ligand_bias_atoms

    extra_res_fa = ""
    if args.params is not None:
        extra_res_fa = '-extra_res_fa'
        for p in args.params:
            if os.path.exists(p):
                extra_res_fa += " {}".format(p)


    pyr.init(f"{extra_res_fa} -run:preserve_header")

    masked_pos = {}

    # If called from commandline then running multithreaded
    print("Running multithreaded PDB processing")
    q = queue.Queue()
    for pdbfile in pdbfiles:
        q.put(pdbfile)
        masked_pos[os.path.basename(pdbfile).replace(".pdb", "")] = {}

    def process():
        while True:
            pdbfile = q.get(block=True)
            if pdbfile is None:
                return

            pdbname = os.path.basename(pdbfile)

            pose = pyr.pose_from_file(pdbfile)

            target_resno = None
            if args.ligand is not None:
                target_resno = []
                for r in pose.residues:
                    if r.name3() in args.ligand and r.is_ligand():
                        target_resno.append(r.seqpos())
                        
            fixed_res_chains = get_fixed_positions(pdbfile, pose, target_resno, cutoff_CA=5.5, cutoff_sc=4.5)

            masked_pos[pdbname.replace(".pdb", "")] = fixed_res_chains

    threads = [threading.Thread(target=process) for _i in range(os.cpu_count())]

    for thread in threads:
        thread.start()
        q.put(None)  # one EOF marker for each thread

    for thread in threads:
        thread.join()

    print("Creating a JSON file specifing fixed positions: masked_pos.jsonl")
    write_json(masked_pos, args.output_path)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", nargs="+", help="Input PDBs. If argument not given, all PDBs in working directory are taken.")
    parser.add_argument("--pdblist", type=str, help="File containing list of PDB filenames")
    parser.add_argument("--params", nargs="+", help="params files")
    parser.add_argument("--design_full", action="store_true", default=False, help="If used then all positions that are not immediately next to the ligand will be redesigned.\n"
                        "By default only the second layer is designed.")
    parser.add_argument("--ligand", nargs="+", help="Ligand name(s) around which positions will not be designed. By default all ligand will be considered. Use if you want to consider only a subset of ligands.")
    parser.add_argument("--output_path", type=str, help="JSONL file generate with parsed rotamers. Names in the JSONL must be <pdfile_r{n}>")

    args = parser.parse_args()
    main(args)
