#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 19:09:02 2021

@author: indrek
"""
import os, sys, signal
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import AlphaFold2
import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--fasta', metavar='FILENAME', nargs='+', help='FASTA file name to analyze with one or more sequences')
parser.add_argument('--af-models', metavar='MODEL', nargs='+', default="4", help='AlphaFold models to run (1-5)')
parser.add_argument('--af-nrecycles', type=int, default=3, help='Number of recycling iterations for AlphaFold')
parser.add_argument('--nstruct', type=int, default=1, help='Number of structures per input sequence, with different random seed')
parser.add_argument('--scorefile', type=str, default="scores.csv", help='Scorefile name. (default = scores.csv)')
parser.add_argument("--npy", action="store_true", default=False, help="Should the npz files be dumped?")

args = parser.parse_args()
# pid = start_nvidia_smi()


if args.fasta is not None:
    sequences = []
    for fname in args.fasta:
        fastafile = open(fname, "r").readlines()
        for i, line in enumerate(fastafile):
            if line[0] == ">":
                seq = fastafile[i+1].rstrip()
                descr = line.replace(">", "").rstrip()
                sequences.append([seq, descr])
    # Sorting the sequences by length to speed up the predictions
    # The model is compiled once per each sequence length
    sequences = sorted(sequences, key=lambda x: len(x[0]))
else:
    sys.exit(1)

with open(args.scorefile, "a") as file:
    file.write("ID,Name,Sequence,Model/Tag,Output_PDB,lDDT,Time\n")


predictions = AlphaFold2.predict_sequences(sequences, args.af_models, args.af_nrecycles, args.scorefile, nstruct=args.nstruct, npy=args.npy)
