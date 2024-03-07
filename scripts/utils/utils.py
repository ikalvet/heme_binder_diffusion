#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 10:56:48 2024

@author: indrek
"""

import os, sys
import matplotlib.pyplot as plt
import numpy as np


def parse_fasta_files(fasta_files):
    """

    Parameters
    ----------
    fasta_files : list
        list of FASTA filenames.

    Returns
    -------
    fasta : dict
        dictionary where the contents of all of the fasta files are combined.
        keys are sequences names, and values are sequences

    """
    fasta = {}
    for ff in fasta_files:
        ffo = open(ff, "r").readlines()
        for i,l in enumerate(ffo):
            if ">" in l:
                fasta[l] = ffo[i+1]
    return fasta

def split_fasta_based_on_length(fasta_dict, count=None, write_files=False):
    """
    Splits an input FASTA dict into groups based on sequence length, and
    splits the groups based on <count> number.

    Parameters
    ----------
    fasta_dict : dict
        dictionary with design names as keys and sequences as values.
    count : int
        how many sequences in each group.

    Returns
    -------
    sorted_seqs : dict
        dictionary where keys are sequence lengths.
        Each item in sub-dict is a FASTA dictionary of length <count>

    """
    seqs = fasta_dict
    unique_seqs = {}
    # Removing duplicates
    for k, seq in seqs.items():
        if seq not in unique_seqs.values():
            unique_seqs[k] = seq
        else:
            print(f"Duplicate sequence: {k}")
    _len = len(unique_seqs)

    print(f"{len(seqs)-_len} duplicate sequences removed.")

    sorted_seqs = {}
    for k in unique_seqs:
        if len(unique_seqs[k]) not in sorted_seqs.keys():
            sorted_seqs[len(unique_seqs[k])] = [[]]
        if count is not None and len(sorted_seqs[len(unique_seqs[k])][-1]) == count:
            sorted_seqs[len(unique_seqs[k])].append([])
        sorted_seqs[len(unique_seqs[k])][-1].append(k)


    for n in sorted_seqs:
        if len(sorted_seqs[n]) == 1:
            continue
        if count > 64 and len(sorted_seqs[n][-1]) <= 16:
            print(f"Regrouping {n}_{len(sorted_seqs[n])}, {len(sorted_seqs[n][-1])} sequences")
            sorted_seqs[n][-2] += sorted_seqs[n][-1]
            sorted_seqs[n][-1] = None
    
    if write_files is True:
        for n in sorted_seqs:
            for j, seqset in enumerate(sorted_seqs[n]):
                if seqset is None:
                    continue
                print(f"{len(seqset)} sequences of {n} length.")
                with open(f"{n}aa_{j}.fasta", "w") as file:
                    for k in seqset:
                        file.write(f"{k}\n{seqs[k]}\n")

    sorted_seqs = {k: [{kk: seqs[kk] for kk in seqset} for seqset in v if seqset is not None] for k,v in sorted_seqs.items() if v is not None}
    return sorted_seqs


def plot_score_pairs(df, score1, score2, score1_line=None, score2_line=None, filename=None):
    
    print(f"Plotting {score1} vs {score2}.")
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.015
    

    min_max = {"rmsd": [0.0, None],
               "lDDT": [None, 100.0]}

    # for k in [score1, score2]:
    #     if k == "rmsd":
    #         min_max[k] = [0.0, None]
    #     if k == "lDDT":
    #         min_max[k] = [min(df[k]), 100.0]

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]
    
    # start with a rectangular Figure
    plt.figure(figsize=(8, 8))
    
    x = df[score1]
    y = df[score2]
    
    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=True)
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)

    # the scatter plot:
    ax_scatter.scatter(x, y, alpha=0.5, linewidth=1)
    ax_scatter.set_xlabel(score1, size=14)
    ax_scatter.set_ylabel(score2, size=14)
    if score1 in min_max.keys():
        ax_scatter.set_xlim(min_max[score1])
    if score2 in min_max.keys():
        ax_scatter.set_ylim(min_max[score2])
    
    ax_scatter.plot([score1_line, score1_line], ax_scatter.get_ylim(), linestyle="--", color="gray", linewidth=1.5)  # X-axis line
    ax_scatter.plot(ax_scatter.get_xlim(), [score2_line, score2_line], linestyle="--", color="gray", linewidth=1.5)  # Y-axis line


    # now determine nice limits by hand:
    binwidth = 0.25
    lim = np.ceil(np.abs([x, y]).max() / binwidth) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=21)
    ax_histy.hist(y, bins=21, orientation='horizontal')

    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())
    
    if filename is not None:
        plt.savefig(filename, dpi=300)
    plt.show()


comparisons = {'<=': '__le__',
               '<': '__lt__',
               '>': '__gt__',
               '>=': '__ge__',
               '=': '__eq__'}


def filter_scores(scores, filters):
    """

    Parameters
    ----------
    scores : pandas.DataFrame
        DESCRIPTION.
    filters : dict
        filter conditions defined as {'key': [cutoff, sign]}
        where sign is one of: '>=', '<=', '=', '>', '<'

    Returns
    -------
    filtered_scores : pandas.DataFrame
        DataFrame filtered based on the filter values

    """
    filtered_scores = scores.copy()

    for s in filters.keys():
        if filters[s] is not None and s in scores.keys():
            val = filters[s][0]
            sign = comparisons[filters[s][1]]
            filtered_scores =\
              filtered_scores.loc[(filtered_scores[s].__getattribute__(sign)(val))]
            n_passed = len(scores.loc[(scores[s].__getattribute__(sign)(val))])
            print(f"{s:<24} {filters[s][1]:<2} {val:>7.3f}: {len(filtered_scores)} "
                  f"items left. {n_passed} pass ({(n_passed/len(scores))*100:.0f}%).")
    return filtered_scores


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


def create_slurm_submit_script(filename, gpu=False, gres=None, time=None, mem="2g", N_nodes=1, N_cores=1, name=None, array=None, array_commandfile=None, group=None, email=None, command=None, outfile_name="output"):
    """
    Arguments:
        time (str) :: time in 'D-HH:MM:SS'
    """
    
    if gpu is True:
        assert gres is not None, "Need to specify resources when asking for a GPU"
    
    cbo = "{"
    cbc = "}"
    submit_txt = \
    f'''#!/bin/bash
#SBATCH --job-name={name}
#SBATCH -t {time}
#SBATCH -N {N_nodes}
#SBATCH -n {N_cores}
#SBATCH --mem={mem}
#SBATCH -o {outfile_name}.log
#SBATCH -e {outfile_name}.err
'''
    if gpu is True:
        submit_txt += f"""#SBATCH -p gpu
#SBATCH --gres={gres}\n"""
    else:
        submit_txt += "#SBATCH -p cpu\n"
    
    if email is not None:
        assert "@" in email, "invalid email address provided"
        submit_txt += f"""#SBATCH --mail-type=END
#SBATCH --mail-user={email}\n"""
    
    if array is not None:
        if group is None:
            submit_txt += f"#SBATCH -a 1-{array}\n"
            submit_txt += f'sed -n "${cbo}SLURM_ARRAY_TASK_ID{cbc}p" {array_commandfile} | bash\n'
        else:
            N_tasks = array
            if N_tasks % group == 0:
                N_tasks = int(N_tasks / group)
            else:
                N_tasks = int(N_tasks // group) + 1
            submit_txt += f'#SBATCH -a 1-{N_tasks}\n'
            submit_txt += f"GROUP_SIZE={group}\n"
         
            submit_txt += "LINES=$(seq -s 'p;' $((($SLURM_ARRAY_TASK_ID-1)*$GROUP_SIZE+1)) $(($SLURM_ARRAY_TASK_ID*$GROUP_SIZE)))\n"
            submit_txt += f'sed -n "${cbo}LINES{cbc}p" ' + f"{array_commandfile} | bash -x\n"
    else:
        submit_txt += f"\n{command}\n"
    
    with open(filename, "w") as file:
        for l in submit_txt:
            file.write(l)

