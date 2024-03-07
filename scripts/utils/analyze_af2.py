#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 2022

@author: ikalvet
"""
import sys, os
sys.path = [x for x in sys.path if sys.base_exec_prefix in x] + [x for x in sys.path if sys.base_exec_prefix not in x]
import pyrosetta as pyr
import pyrosetta.rosetta
import re
import pandas as pd
import numpy as np
import argparse
import scipy.spatial
import queue
import threading
import multiprocessing
import json


def reorder_df_columns(df):
    namekeys = ["description", "name", "Name", "Output_PDB"]
    # Reordering hte columns to make sure that the name column is last
    cols = df.columns.tolist()
    cols = [x for x in cols if x not in namekeys] + [x for x in cols if x in namekeys]
    df = df[cols]
    return df


def dump_scorefile(df, filename):
    widths = {}
    namekeys = ["description", "name", "Output_PDB", "Name"]

    for k in df.keys():
        if k in ["SCORE:"] + namekeys:
            widths[k] = 0
        if len(k) >= 12:
            widths[k] = len(k) + 1
        else:
            widths[k] = 12

    with open(filename, "w") as file:
        title = ""
        for k in df.keys():
            if k == "SCORE:":
                title += k
            elif k in namekeys:
                title += f" {k}"
            else:
                title += f"{k:>{widths[k]}}"
        if all([t not in df.keys() for t in namekeys]):
            title += f" {'description'}"
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
                elif k in namekeys:
                    line += f" {val}"
                else:
                    line += f"{val:>{widths[k]}}"
            if all([t not in df.keys() for t in namekeys]):
                line += f" {index}"
            file.write(line + "\n")


def rmsd(geom, target):
    return np.sqrt(((geom - target) ** 2).mean())


def get_per_res_rmsd(design: pyrosetta.rosetta.core.pose.Pose, prediction: pyrosetta.rosetta.core.pose.Pose, rmsd_type: str, N_term_offset=0) -> list:
    """calculate per residue rmsd (Ca or sc) of prediction to design"""
    # per_res_rmsd = pyrosetta.rosetta.core.simple_metrics.per_residue_metrics.PerResidueRMSDMetric()
    
    result = {}
    for resno in range(1, prediction.size()+1):
        resp = prediction.residue(resno)
        resd = design.residue(resno+N_term_offset)
        if resp.name3() != resd.name3():
            result[resno-1] = 0.0
        else:
            atoms = [resp.atom_name(n).strip() for n in range(1, resp.natoms()+1) if not resp.atom_is_hydrogen(n) and not resp.is_virtual(n)]
            ref_coords = [resd.xyz(a) for a in atoms]
            mdl_coords = [resp.xyz(a) for a in atoms]
            result[resno-1] = np.sqrt(sum([(np.linalg.norm(c1-c2))**2 for c1, c2 in zip(ref_coords, mdl_coords)])/len(atoms))
    return result


def get_matcher_residues(filename):
    pdbfile = open(filename, 'r').readlines()

    matches = {}
    for l in pdbfile:
        if "ATOM" in l:
            break
        if "REMARK 666" in l:
            lspl = l.split()
            chain = lspl[9]
            res3 = lspl[10]
            resno = int(lspl[11])
            
            matches[resno] = {'name3': res3,
                              'chain': chain}
    return matches


def get_residues_with_close_sc_or_bb(pose, ref_resno, residues=None, exclude_residues=None):
    """
    """
    if residues is None:
        residues = [x for x in range(1, pose.size()+1)]
    if exclude_residues is None:
        exclude_residues = []

    ref_residue = pose.residue(ref_resno)
    heavyatoms = [ref_residue.atom_name(n).strip() for n in range(1, ref_residue.natoms()+1) if ref_residue.atom_type(n).is_heavyatom()]

    close_ones = []
    for resno in residues:
        if resno in exclude_residues:
            continue
        if pose.residue(resno).is_ligand():
            continue
        if (pose.residue(resno).nbr_atom_xyz() - ref_residue.nbr_atom_xyz()).norm() > 14.0:
            continue
        res = pose.residue(resno)
        close_enough = False

        for ha in heavyatoms:
            # If the CA of residue is really close then keep it, no questions asked
            if (res.xyz("CA") - ref_residue.xyz(ha)).norm() < 5.5:
                close_enough = True
                close_ones.append(resno)
                break
            else:
                # If the CA is further then check if any sidechain atoms is really close
                for atomno in range(1, res.natoms()):
                    if res.atom_type(atomno).is_heavyatom():
                        if (res.xyz(atomno) - ref_residue.xyz(ha)).norm() < 4.5:
                            close_enough = True
                            close_ones.append(resno)
                            break
                if close_enough is True:
                    break
            if close_enough is True:
                break
    return close_ones


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--params', type=str, nargs='+', help="Rosetta params files needed to load any PDBs, if the reference PDB's have ligands")
    parser.add_argument('--ref_path', type=str, required=True, help='path where reference PDB files are found for rmsd calculations')
    parser.add_argument('--scorefile', type=str, required=True, help='AlphaFold2 output scorefile with lDDT, Output_PDB and Name headers')
    parser.add_argument('--lddt', type=float, default=0.0, help='Minimum lddt cutoff above which structures are analyzed.')
    parser.add_argument('--mpnn', action="store_true", default=False, help='Finds reference structures based on a hardcoded MPNN naming convention that exists in this pipeline.')
    parser.add_argument('--pocket', action="store_true", default=False, help='Calculates sidechain rmsds of pocket residues, defined by proximity to a ligand in the reference structure.')
    parser.add_argument('--posdict', type=str, help='A JSON file that defines per-chain residue numbers for pocket sidechain rmsd calculation for each reference structure.')
    parser.add_argument('--no_align', action="store_true", default=False, help='Calculates rmsd by Rosetta CA superimposition. Alternative (False) calculates rmsd by distance matrix difference')
    parser.add_argument('--out', type=str, default="scores.sc", help='Name of output scorefile')

    args = parser.parse_args()

    params = args.params
    ref_path = args.ref_path
    scorefile = args.scorefile
    no_align = args.no_align
    mpnn_naming = args.mpnn
    lddt_cutoff = args.lddt
    pocket_rmsd = args.pocket

    pocket_df = pd.DataFrame()

    posdict = None
    if args.posdict is not None:
        with open(args.posdict, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            posdict = json.loads(json_str)


    if os.path.exists(args.out):
        print(f"Skipping {scorefile}")
        sys.exit(0)

    scores = pd.read_csv(scorefile, header=0)

    print(f"Only calculating RMSD for predictions with lDDT >= {lddt_cutoff:.1f}")
    if len(scores.loc[scores.lDDT > lddt_cutoff]) == 0:
        sys.exit(f"No designs with lDDT above {lddt_cutoff}. Nothing to analyze")
    else:
        print(f"{len(scores.loc[scores.lDDT > lddt_cutoff])} models with lDDT >= {lddt_cutoff} to analyze.")

    if params is None:
        pyr.init('-mute all -beta_nov16')
    else:
        pyr.init('-extra_res_fa {} -mute all'.format(" ".join(params)))


    the_queue = multiprocessing.Queue()  # Queue stores the iterables

    manager = multiprocessing.Manager() 
    ref_poses = manager.dict()  # Need a special dictionary to store outputs from multiple processes
    results = manager.dict()
    
    
    if pocket_rmsd is True:
        pocket_results = manager.dict()

    for idx, row in scores.iterrows():
        results[idx] = manager.dict()
        if pocket_rmsd is True:
            pocket_results[idx] = manager.dict()
        the_queue.put(idx)

    ref_pockets = manager.dict()

    def process(q):
        while True:
            idx = q.get(block=True)
            if idx is None:
                return
            row = scores.iloc[idx]

            results[idx]["Output_PDB"] = row["Output_PDB"]
            results[idx]['rmsd'] = np.nan

            if row.lDDT < lddt_cutoff:
                continue

            if "Output_PDB" in scores.keys():
                model_name = f"{row['Output_PDB']}"
                if ".pdb" not in model_name:
                    model_name += ".pdb"
            else:
                model_name = f"{row.ID}_{row['Model/Tag']}"

            if mpnn_naming is True:
                if "_native" in row['Name'][-7:]:
                    # replacing last instance of 'native'
                    ref_name = "".join(row['Name'].rsplit("_native", 1))
                elif bool(re.search("_T[0-9].[0-9]_0_[0-9]", row['Name'])):
                    ref_name = "_".join(row['Name'].split("_")[:-3])
                else:
                    ref_name = f"{row['Name']}"
            else:
                ref_name = f"{row['Name']}"


            ref_pose = pyr.pose_from_file(f"{ref_path}/{ref_name}.pdb")
            if "Output_PDB" in scores.keys():
                model_pdb = f"{row['Output_PDB']}.pdb"
                if ".pdb" in row['Output_PDB']:
                    model_pdb = f"{row['Output_PDB']}"
                model_pose = pyr.pose_from_file(model_pdb)
            else:
                model_pose = pyr.pose_from_file(f"{row.ID}_{row['Model/Tag']}.pdb")

            reslist = pyrosetta.rosetta.std.list_unsigned_long_t()
            for n in range(1, model_pose.size()+1):
                reslist.append(n)

            matches = get_matcher_residues(f"{ref_path}/{ref_name}.pdb")

            # Finding how the sequence should be aligned
            # It doesn't do non-continuous sequences yet - TODO!
            if len(ref_pose.sequence()) < len(model_pose.sequence()):
                shorter_seq = ref_pose.sequence()
                longer_seq = model_pose.sequence()
                _sp = ref_pose
                _lp = model_pose
            else:
                shorter_seq = model_pose.sequence()
                longer_seq = ref_pose.sequence()
                _lp = ref_pose
                _sp = model_pose

            N_term_offset = longer_seq.find(shorter_seq)
            if mpnn_naming is True:
                N_term_offset = 0
            C_term_offset = len(longer_seq[N_term_offset:]) - len(shorter_seq)

            if N_term_offset == -1:
                print(f"{row.Output_PDB}: Sequences not alignable?\n{shorter_seq}\n{longer_seq}")
                continue

            if no_align is False:
                
                overlay_pos = pyrosetta.rosetta.utility.vector1_unsigned_long()
                for n in range(1, _sp.size()+1):
                    overlay_pos.append(n)
                rmse = pyrosetta.rosetta.protocols.toolbox.pose_manipulation.superimpose_pose_on_subset_CA(_lp, _sp, overlay_pos, N_term_offset)
                results[idx]['rmsd'] = rmse

            elif no_align is True:

                # Finding the distance matrices of CA atoms in both poses
                shorter_distmat = scipy.spatial.distance.pdist([np.array(_sp.residue(n+1).xyz("CA")) for n in range(len(shorter_seq))], 'euclidean')
                longer_distmat = scipy.spatial.distance.pdist([np.array(_lp.residue(n+1).xyz("CA")) for n in range(N_term_offset, len(longer_seq)-C_term_offset)], 'euclidean')

                assert len(shorter_distmat) == len(longer_distmat), f"{row.Output_PDB}: distmats not equal length"

                # Calculating CA RMSD
                rmse = rmsd(shorter_distmat, longer_distmat)
                results[idx]["rmsd"] = rmse

            print(f"{row.Output_PDB}: length = {min([ref_pose.size(), model_pose.size()])}, "
                  f"lDDT = {row.lDDT:.2f}, rmsd = {rmse:.3f}, ")

            # Calculating sidechain RMSD of each residue
            res_rmsd = get_per_res_rmsd(ref_pose, model_pose, 'sc', N_term_offset=N_term_offset)

            # Finding RMSD's of matched residues
            for i, resno in enumerate(matches):
                results[idx][f'rmsd_SR{i+1}'] = res_rmsd[resno-1]
                if np.isnan(res_rmsd[resno-1]):
                    results[idx][f'rmsd_SR{i+1}'] = 0.0

            # If requested, calculating SC rmsd-s of pocket residues
            if pocket_rmsd is True:
                for _k in ["Name", "Output_PDB"]:
                    pocket_results[idx][_k] = row[_k]

                if posdict is None:
                    if ref_name not in ref_pockets.keys():
                        close_ones = get_residues_with_close_sc_or_bb(ref_pose, ref_pose.size())
                        ref_pockets[ref_name] = [x for x in close_ones]
                    else:
                        close_ones = [x for x in ref_pockets[ref_name]]
                else:
                    close_ones = []
                    for ch in posdict[ref_name]:
                        close_ones += posdict[ref_name][ch]
                    # print(close_ones)
                pocket_rmsds = [res_rmsd[resno-1] for resno in close_ones]
                results[idx]["rmsd_pocket"] = np.average(pocket_rmsds)
                results[idx]["rmsd_pocket_std"] = np.std(pocket_rmsds)
                for resno, rms in zip(close_ones, pocket_rmsds):
                    pocket_results[idx][f"rms_{resno}"] = rms
                pocket_results[idx]["ref_name"] = ref_name




    if "OMP_NUM_THREADS" in os.environ:
        N_PROCESSES = int(os.environ["OMP_NUM_THREADS"])
        print(f"Using {N_PROCESSES} processes")
    else:
        N_PROCESSES = os.cpu_count() - 1


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


    for i in results.keys():
        assert scores.at[i, "Output_PDB"] == results[i]["Output_PDB"], f"Bad match between scores dataframe and results dict? {results[i].items(), scores.iloc[i]}"
        for _k in results[i].keys():
            scores.at[i, _k] = results[i][_k]


    scores = reorder_df_columns(scores.drop(columns=["Sequence"]))

    _title = f"{('ID'):>3}{('lDDT'):>8} {('rmsd'):>8}"
    for k in scores.keys():
        if "SR" in k:
            _title += f" {k:>8}"
        if "rmsd_pocket" in k:
            _title += f"{('rmsd_pocket'):>12}"
    _title += f" {('Name'):<}"
    print(_title)

    for idx, row in scores.iterrows():
        if row.hasnans:
            continue
        _line = f"{row['ID']:>3}{row['lDDT']:>8.3f} {row['rmsd']:>8.3f}"
        for k in scores.keys():
            if "SR" in k:
                _line += f" {row[k]:>8.3f}"
        if pocket_rmsd is True:
            _line += f"{row['rmsd_pocket']:>12.3f}"
        _line += f" {row['Name']:<}"
        print(_line)

    if pocket_rmsd is True:
        for i in pocket_results.keys():
            assert pocket_df.at[i, "Output_PDB"] == pocket_results[i]["Output_PDB"], f"Bad match between scores dataframe and pocket results dict? {results[i].items(), scores.iloc[i]}"
            for _k in pocket_results[i].keys():
                pocket_df.at[i, _k] = pocket_results[i][_k]
        pocket_df = reorder_df_columns(pocket_df)
        if os.path.dirname(args.out) == "":
            pocket_scorefile = "pocket_" + os.path.basename(args.out)
        else:
            pocket_scorefile = os.path.dirname(args.out) + "/pocket_" + os.path.basename(args.out)
        print(f"Writing scorefile with per-residue sc-rmsd values: {pocket_scorefile}")
        dump_scorefile(pocket_df, pocket_scorefile)

    dump_scorefile(scores, args.out)

if __name__ == "__main__":
    main()
