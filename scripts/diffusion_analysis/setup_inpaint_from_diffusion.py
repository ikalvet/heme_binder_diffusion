#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Indrek Kalvet
ikalvet@uw.edu
"""

import os
import sys
import numpy as np
import pandas as pd
import time
import queue
import threading
import argparse
from shutil import copy2
import pyrosetta as pyr
import pyrosetta.rosetta
import json


aa3to1 = {
    "ALA":'A', "ARG":'R', "ASN":'N', "ASP":'D', "CYS":'C',
    "GLN":'Q', "GLU":'E', "GLY":'G', "HIS":'H', "ILE":'I',
    "LEU":'L', "LYS":'K', "MET":'M', "PHE":'F', "PRO":'P',
    "SER":'S', "THR":'T', "TRP":'W', "TYR":'Y', "VAL":'V' }

aa1to3 = {val: k for k, val in aa3to1.items()}


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--pdb", nargs="+", type=str, required=True, help="Input PDBs from aa_diffusion")
    parser.add_argument("--trb", nargs="+", type=str, help="TRB files from aa_diffusion")
    parser.add_argument("--params", nargs="+", type=str, help="Params files of ligands and noncanonicals")
    parser.add_argument("--group", type=int, help="How many designs will be in each JSON file")
    parser.add_argument("--var", action="store_true", default=False, help="Inpaint with variable contig lengths?")
    parser.add_argument("--only_seq", action="store_true", default=False, help="Do not generate backbone, just design sequence")
    parser.add_argument("--design_full", action="store_true", default=False, help="Design all non-catalytic (defined with --ref_catres) residues.")
    parser.add_argument("--ref_catres", type=str, nargs="+", help="Format: 'A15 Bxx'. Residue that is defined as catalytic in the diffusion input. This will be parsed from the trb con_ref_pdb_idx")

    args = parser.parse_args()
    
    pdbfiles = args.pdb
    params = args.params
    

    extra_res_fa = ""
    if len(params) > 0:
        extra_res_fa = "-extra_res_fa"
        for p in params:
            extra_res_fa += " " + p

    pyr.init(f"{extra_res_fa} -mute all -run:preserve_header")


    inpaint_dict = {"pdb": None,
        "task": "hal",
        "dump_all": True,
        "inf_method": "multi_shot",
        "num_designs": 1,
        "tmpl_conf": "1.0",
        "exclude_aa": "C",
        "inpaint_seq": None,
        "contigs": None,
        "out": None}


    inpaint_dict_list = []

    start = time.time()

    q = queue.Queue()

    for i, pdbfile in enumerate(pdbfiles):
        q.put((i, pdbfile))

    def process():
        while True:
            p = q.get(block=True)
            if p is None:
                return
            i = p[0]
            pdbfile = p[1]
            
            if args.trb is None:
                trbfile = pdbfile.replace(".pdb", ".trb")
                if not os.path.exists(trbfile) and "traj" in pdbfile.split("_")[-1]:
                    trbfile = "_".join(pdbfile.split("_")[:-1]) + ".trb"

            else:
                __trbfs = [x for x in args.trb if os.path.basename(pdbfile).replace(".pdb", "") in x]
                assert len(__trbfs) == 1, f"Bad number of trbs for {pdbfile}: {__trbfs}"
                trbfile = __trbfs[0]


            pose = pyr.pose_from_file(pdbfile)
    
            dssp = pyrosetta.rosetta.core.scoring.dssp.Dssp(pose)
            if pose.residue(pose.size()).is_ligand():
                secstruct = dssp.get_dssp_secstruct()[:-1]  # excluding the ligand
            else:
                secstruct = dssp.get_dssp_secstruct()



            ### Loading trb file and figuring out fixed positions between ref and hal
            trb = np.load(trbfile, allow_pickle=True)
            fixed_pos_in_hal0 = trb["con_hal_idx0"]
            fixed_pos_in_hal = [x+1 for x in fixed_pos_in_hal0]
            fixed_pos_in_ref = trb["con_ref_pdb_idx"]

            # Diffusion fixed positions will not be touched
            forbidden_positions = [x for x in fixed_pos_in_hal]
            catalytic_positions = []
            if args.ref_catres is not None:
                for _nr, _or in zip(fixed_pos_in_hal, fixed_pos_in_ref):
                    if f"{_or[0]}{_or[1]}" in args.ref_catres:
                        catalytic_positions.append(_nr)

            if args.only_seq is False:
                # Positions 5 res upstream and downstream of fixed or catalytic positions will not be touched
                for x in fixed_pos_in_hal:
                    if x in catalytic_positions or args.ref_catres is None:
                        for n in range(x-4, x+4):
                            if n > 0 and n not in forbidden_positions and n < pose.size():
                                forbidden_positions.append(n)
                    else:
                        forbidden_positions.append(x)

                # Finding which positions are close to ligand and also not allowing design on these
                if pose.residue(pose.size()).is_ligand():
                    ligand = pose.residue(pose.size())
                    lig_heavyatoms = [atomno for atomno in range(1, ligand.natoms()+1) if ligand.atom_type(atomno).element() != "H"]
                    for res in pose.residues:
                        if res.is_ligand() or res.is_virtual_residue():
                            continue
                        for lig_ha in lig_heavyatoms:
                            if (ligand.xyz(lig_ha) - res.xyz("CA")).norm() < 5.0:
                                forbidden_positions.append(res.seqpos())
                                break
        
                forbidden_positions = sorted(list(set(forbidden_positions)))
        
        
                #### BELOW IS A VERY PRIMITIVE WAY TO FIGURE OUT WHAT POSITIONS TO INPAINT ###
                # TODO: split it into functions when things work reliably
        
                # Finding loops in the structure
                loop_start = None
                loops = []
                for i, l in enumerate(secstruct):
                    if l == "L" and (i == 0 or secstruct[i-1] != "L"):
                        loop_start = i
                    if l == "L" and (i == len(secstruct)-1 or secstruct[i+1] != "L"):
                        loops.append((loop_start+1, i+1))
                        loop_start = None
        
                # Expand positions around loops to up to 3 residues
                expanded_loops = []
                for j, loop in enumerate(loops):
                    if loop[1] - loop[0] + 1 == 1:  # Dropping loops of length 1
                        continue
                    ns = [n for n in range(loop[0], loop[1]+1)]
                    if any([n in forbidden_positions for n in ns]):  # Dropping loops that contain any forbidden residues
                        continue
                    # Expanding the loops
                    expanded_loop = [n for n in range(loop[0], loop[1]+1)]
                    # C-term
                    for _ in range(3):
                        if expanded_loop[-1] + 1 not in forbidden_positions and expanded_loop[-1] + 1 in range(1, pose.size()):
                            expanded_loop.append(expanded_loop[-1] + 1)
                        else:
                            break
                    # N-term
                    for _ in range(3):
                        if expanded_loop[0] - 1 not in forbidden_positions and expanded_loop[0] - 1 in range(1, pose.size()):
                            if j > 0 and loops[j-1][-1] == expanded_loop[0] - 2:  # Not adding the last bit if it makes two regions run into eachother
                                break
                            expanded_loop = [expanded_loop[0] - 1] + expanded_loop
                        else:
                            break
                    expanded_loops.append(expanded_loop)
        
                # Adding terminal chunks to inpaint regions
                if not any([x < 6 for x in forbidden_positions]):
                    if 1 < expanded_loops[0][0] <= 6:  # There's already a loop that's close to N-term
                        expanded_loops[0] = [x for x in range(1, expanded_loops[0][0])] + expanded_loops[0]
                    elif expanded_loops[0][0] > 6:  # No loop that includes any of the first 6 residues
                        expanded_loops = [[x for x in range(1, 7)]] + expanded_loops
                if not any([x > pose.size() - 7 for x in forbidden_positions]):
                    if pose.size() > expanded_loops[-1][-1] >= pose.size()-7:  # There's already a loop that's close to C-term
                        expanded_loops[-1] = expanded_loops[-1] + [x for x in range(expanded_loops[-1][-1], pose.size())]
                    elif expanded_loops[-1][-1] < pose.size()-7:  # No loop that includes any of the last 6 residues
                        expanded_loops.append([x for x in range(pose.size()-7, pose.size())])

                # Checking if loop expansion caused some of them to overlap
                # Combining overlapping regions
                _tmp_exp_loops = []
                for j, loop in enumerate(expanded_loops):
                    if j > 0 and any([x in _tmp_exp_loops[-1] for x in loop]):
                        continue
                    _combined_loops = [x for x in loop]
                    jjj = j+1
                    jj = j
                    if jjj <= len(expanded_loops)-1:
                        while any([x in expanded_loops[jjj] for x in expanded_loops[jj]]) or expanded_loops[jj][-1]+1 == expanded_loops[jjj][0]:
                            _combined_loops += expanded_loops[jjj]
                            jjj += 1
                            jj += 1
                            if jjj > len(expanded_loops)-1:
                                break
                    _tmp_exp_loops.append(sorted(list(set(_combined_loops))))
                expanded_loops = [x for x in _tmp_exp_loops]

                n_des = 1
                contig = ""
                for j, loop in enumerate(expanded_loops):
                    loop_length = len(loop)
                    if args.var is True:
                        if 10 <= loop_length < 15:
                            loop_length = f"{loop_length}-{loop_length+1}"
                            n_des = 2
                        elif 15 <= loop_length < 20:
                            loop_length = f"{loop_length-1}-{loop_length+2}"
                            n_des = 3
                        elif 20 <= loop_length <= 25:
                            loop_length = f"{loop_length-2}-{loop_length+3}"
                            n_des = 5
                        elif loop_length > 25:
                            loop_length = f"{loop_length-int(0.1*loop_length)}-{loop_length+int(0.1*loop_length)}"
                            n_des = 5

                    if j == 0 and loop[0] == 1:
                        contig += f"{loop_length},A{loop[-1]+1}-"
                    elif j == 0:
                        contig += f"A1-{loop[0]-1},{loop_length},A{loop[-1]+1}-"
                    elif j == len(expanded_loops)-1 and loop[-1] == pose.size()-1:
                        contig += f"{loop[0]-1},{loop_length}"
                    elif j == len(expanded_loops)-1:
                        contig += f"{loop[0]-1},{loop_length},A{loop[-1]+1}-{pose.size()-1}"
                    else:
                        contig += f"{loop[0]-1},{loop_length},A{loop[-1]+1}-"


                keep_regions = [x for x in contig.split(",") if "A" in x]
                try:
                    keep_regions_expanded = [[n for n in range(int(x.split("-")[0][1:]), int(x.split("-")[1])+1)] for x in keep_regions]
                except ValueError:
                    print(pdbfile)
                    print(expanded_loops)
                    print(contig)
                    print(keep_regions)
                    sys.exit(1)

                # print(fixed_pos_in_hal)

                keep_regions_expanded_no_catres = []
                
                if args.design_full is True:
                    _positions_to_not_design = catalytic_positions
                else:
                    _positions_to_not_design = fixed_pos_in_hal
                
                for rgn in keep_regions_expanded:
                    if any([n in _positions_to_not_design for n in rgn]):
                        _tmp_rgn = []
                        for resno in rgn:
                            if resno not in _positions_to_not_design:
                                _tmp_rgn.append(resno)
                                if resno == rgn[-1]:
                                    keep_regions_expanded_no_catres.append(_tmp_rgn)
                            # elif resno in fixed_pos_in_hal and trb["inpaint_seq"][resno-1] == False:  # TODO: workaround for a TRB bug
                                # inpaint_seq positions are designable
                                # _tmp_rgn.append(resno)
                                # print(resno)
                                # print(resno in fixed_pos_in_hal)
                                # print(trb["inpaint_seq"][resno-1])
                                # print(trb["inpaint_seq"])
                                # print(trb["inpaint_seq"][0])
                                # print([n for n,t in enumerate(trb["inpaint_seq"]) if t == True])

                            else:
                                if len(_tmp_rgn) != 0:
                                    keep_regions_expanded_no_catres.append(_tmp_rgn)
                                    _tmp_rgn = []
                    else:
                        keep_regions_expanded_no_catres.append(rgn)
                    
            elif args.only_seq is True:
                keep_regions_expanded_no_catres = [[]]
                if args.design_full is False:
                    forbidden_positions = [x for x in forbidden_positions if trb["inpaint_seq"][x-1] == True]
                else:
                    forbidden_positions = [x for x in catalytic_positions]
                for n in range(1, len(secstruct)+1):
                    if n not in forbidden_positions:
                        keep_regions_expanded_no_catres[-1].append(n)
                    else:
                        keep_regions_expanded_no_catres.append([])
                keep_regions_expanded_no_catres = [x for x in keep_regions_expanded_no_catres if len(x) != 0]
                contig = f"A1-{len(secstruct)}"
                n_des = 1

                
            inpaint_seq = ",".join([f"A{x[0]}-{x[-1]}" for x in keep_regions_expanded_no_catres])
            print(f"{os.path.basename(pdbfile)} {secstruct}\n"
                  f"{os.path.basename(pdbfile)}: contigs = {contig}\n"
                  f"{os.path.basename(pdbfile)}: inpaint_seq = {inpaint_seq}")

            ### Setting up the commands dictionary for inpainting
            _dict = {k: val for k, val in inpaint_dict.items()}
            _dict["pdb"] = os.path.realpath(pdbfile)
            _dict["out"] = os.path.basename(pdbfile).replace(".pdb", "_inp")
            _dict["contigs"] = [contig]
            _dict["inpaint_seq"] = [inpaint_seq]

            if "EEE" in secstruct and args.var is False and args.only_seq is False:
                _dict["num_designs"] = 2
            elif args.var is True:
                _dict["num_designs"] = n_des
            else:
                _dict["num_designs"] = n_des


            inpaint_dict_list.append(_dict)


    threads = [threading.Thread(target=process) for _i in range(os.cpu_count())]

    for thread in threads:
        thread.start()
        q.put(None)  # one EOF marker for each thread

    for thread in threads:
        thread.join()

    end = time.time()
    if args.group is None:
        with open("cmds.json", "w") as file:
            json.dump(inpaint_dict_list, file, separators=(",\n", ":"))
    else:
        for j, i in enumerate(range(0, len(inpaint_dict_list), args.group)):
            _tmp = inpaint_dict_list[i:i+args.group]
            with open(f"cmds_{j}.json", "w") as file:
                json.dump(_tmp, file, separators=(",\n", ":"))

    print("Creating inpainting inputs took {:.3f} seconds.".format(end - start))




if __name__ == "__main__":
    main()
