#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 14:52:16 2021

@author: ikalvet, northja
"""

import numpy as np
import argparse
import json
import sys
import os


if __name__ == "__main__":
# def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="masked_pos.jsonl", help="Output file")
    parser.add_argument("--trb", nargs="+", dest="trbfiles", help="trb files from inpainting or hallucination")
    parser.add_argument("--trblist", type=str, dest="trblist", help="File with a list trb filenames from inpainting or hallucination")
    parser.add_argument("--essential_motif_res", type=int, nargs='+', dest='essential_motif_res', required=False, default=None, help="Space-separated list of positions of catalytic residues in chain B")
    parser.add_argument("--keep_native", nargs='+', required=False, default=None, help="Space-separated list of positions of conserved residues, optionally with chain letters.")
    parser.add_argument("--old_format", action="store_true", default=False, help="The JSON file will be produced in the old MPNN style format with a per-chain dictionary.\n"
                                                                                 "{'pdbname': {'A': [resno1, resno2]}}")

    args = parser.parse_args()
    maskdict = {}

    if args.trblist is not None:
        _trbfiles = open(args.trblist, "r").readlines()
        _trbfiles = [f.rstrip() for f in _trbfiles if ".trb" in f]
        setattr(args, "trbfiles", _trbfiles)

    for trbfile in args.trbfiles:
        _trb = np.load(trbfile, allow_pickle=True)
        
        idx_catres = []
        if "con_hal_idx0" in _trb.keys():  # trb file from hallucination or inpainting
            seqpos = _trb["con_hal_idx0"]
            
            mask_seqpos = None
            if args.keep_native is not None:
                poslist = _trb["con_hal_idx0"]
                ref_pdb = _trb["con_ref_pdb_idx"]

                native_positions = []
                accepted = []  # list of tuples with fixed positions [(A, ##),]
                for p in args.keep_native:
                    if p.isnumeric():
                        native_positions.append(p)
                        accepted.append(("A", int(p)))
                    elif not p.isnumeric() and "-" not in p:
                        accepted.append((p[0], int(p[1:])))
                    elif "-" in p:
                        if p[0].isnumeric():
                            _ch = "A"
                            _rng = (int(p.split("-")[0]), int(p.split("-")[-1])+1)
                        else:
                            _ch = p[0]
                            _rng = (int(p.split("-")[0][1:]), int(p.split("-")[-1])+1)
                        for _n in range(_rng[0], _rng[1]):
                            native_positions.append(_n)
                            accepted.append((_ch, _n))
                    else:
                        print(f"Invalid value for -keep_native: {p}")

                # accepted = [('A', p) for p in native_positions]
                acc = [i for i,p in enumerate(ref_pdb) if p in accepted]  # List of fixed position id's in the reference PDB list
                mask_seqpos = [int(poslist[x])+1 for x in acc]  # Residue numbers of fixed positions in inpaint output
                # mask_seqpos = [i for i in mask_seqpos if _trb["inpaint_seq"][i-1] == True and _trb["inpaint_str"][i-1] == True]
                

            elif args.essential_motif_res is None: 
                mask_seqpos = [int(i+1) for i in _trb["con_hal_idx0"]]
                mask_seqpos = [i for i in mask_seqpos if _trb["inpaint_seq"][i-1] == True and _trb["inpaint_str"][i-1] == True]
            else:
                # get a list that's basically _trb["con_hal_idx0"] but doesn't have any B residues masked in it
                poslist = _trb["con_hal_idx0"]
                ref_pdb = _trb["con_ref_pdb_idx"]

                # remove all B residues from masked
                accepted = [('B', p) for p in args.essential_motif_res]
                acc = [i for i,p in enumerate(ref_pdb) if p in accepted]
                is_B = [i for i, p in enumerate(ref_pdb) if p[0] == 'B']
                rem = list(set(is_B) - set(acc))
                
                idx_catres = [poslist[i] for i in acc]

                for i in sorted(rem, reverse=True): 
                    del poslist[i]

                mask_seqpos = [int(i+1) for i in poslist]
                # mask_seqpos = [i for i in mask_seqpos if _trb["inpaint_seq"][i-1] == True and _trb["inpaint_str"][i-1] == True]

            
        else:
            mask_seqpos = [i+1 for i, val in enumerate(_trb["inpaint_seq"]) if val == True]
            print(_trb["flags"].inpaint_seq, mask_seqpos)
            # mask_seqpos = [i for i in mask_seqpos if _trb["inpaint_seq"][i-1] == True and _trb["inpaint_str"][i-1] == True]
            # print(f"Bad trb file: {trbfile}")
            # continue
        if args.old_format is True:
            maskdict[os.path.basename(trbfile).replace(".trb", "")] = {"A": mask_seqpos}
        else:
            maskdict[os.path.basename(trbfile).replace(".trb", "")] = ' '.join([f"A{r}" for r in mask_seqpos])

    with open(args.out, 'w') as f:
        f.write(json.dumps(maskdict) + '\n')

# if __name__ == "__main__":
#     main()
