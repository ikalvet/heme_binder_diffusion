"""
@author: Indrek Kalvet
"""

import numpy as np
import pyrosetta as pyr
import pyrosetta.rosetta
from pyrosetta.distributed.packed_pose.core import PackedPose


init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L',
                       'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X',
                       'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j',
                       'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']


def softmax(x, T):
    return np.exp(x/T)/np.sum(np.exp(x/T), -1, keepdims=True)


def make_bias_pssm(pose, design_positions, bias_AAs, bias_low=-1.0, bias_high=1.39):
    """
    Parameters
    ----------
    pose : pyrosetta.rosetta.core.pose.Pose
        Input Rosetta pose
    design_positions : list
        list of positions for which PSSM will be generated
    bias_AAs : str
        AA1's which will be biased towards or against with the PSSM
    bias_low : float
        log_odd for unwanted residue types (default = -1.0)
    bias_high : float
        log_odd for favored residue types (default = 1.39)

    Returns
    -------
    pssm_dict : dict
        dictionary with pssm_bias, pssm_log_odds and pssm_coefs values for each residue.
        Residues are stored in separate dictionaries based on the chain letter
    """
    assert isinstance(pose, (pyrosetta.rosetta.core.pose.Pose,
                             pyrosetta.distributed.packed_pose.core.PackedPose)), "Bad input type for `pose`"
    assert isinstance(design_positions, list), "Bad input type for `design_positions`"
    assert isinstance(bias_AAs, str), "Bad input type for `bias_AAs`"
    assert isinstance(bias_low, (float, np.float64)), "Bad input type for `bias_low`"
    assert isinstance(bias_high, (float, np.float64)), "Bad input type for `bias_high`"

    if isinstance(pose, pyrosetta.rosetta.core.pose.Pose):
        _pose = pose
    elif isinstance(pose, pyrosetta.distributed.packed_pose.core.PackedPose):
        _pose = pose.pose

    mpnn_alphabet = 'ACDEFGHIKLMNPQRSTVWYX'

    pssm_dict = {}

    pssm_log_odds = np.ndarray((_pose.size()-1, 21))
    pssm_coefs = []  # a number between 0.0 and 1.0 specifying how much attention put to PSSM, can be adjusted later as a flag

    for res in _pose.residues:
        if res.is_ligand() is True:
            continue
        
        if res.seqpos() not in design_positions:
            pssm_coefs.append(0.0)
            pssm_log_odds[res.seqpos()-1] = np.zeros(21)
        else:
            pssm_coefs.append(1.0)
            _pos_log_odds = np.ndarray((21, ))
            for i, r in enumerate(mpnn_alphabet):
                if r in bias_AAs:
                    _pos_log_odds[i] = np.float64(bias_high)
                else:
                    _pos_log_odds[i] = np.float64(bias_low)
            _pos_log_odds[-1] = np.float64(0.0)
            pssm_log_odds[res.seqpos()-1] = _pos_log_odds

    X_mask = np.concatenate([np.zeros([1,20]), np.ones([1,1])], -1)

    pssm_bias = (softmax(np.array(pssm_log_odds)-X_mask*1e8, 1.0)).tolist()  # PSSM like, [length, 21] such that sum over the last dimension adds up to 1.0
    pssm_log_odds_list = pssm_log_odds.tolist()

    # Clumsy way of making the dictionary chain-specific
    for res in _pose.residues:
        # Not adding ligand chain to the PSSM, is that ok?
        if res.is_ligand() is True:
            continue

        ch = _pose.pdb_info().chain(res.seqpos())
        if ch not in pssm_dict.keys():
            pssm_dict[ch] = {"pssm_coef": [],
                             "pssm_bias": [],
                             "pssm_log_odds": []}
        
        pssm_dict[ch]["pssm_coef"].append(pssm_coefs[res.seqpos()-1])
        pssm_dict[ch]["pssm_bias"].append(pssm_bias[res.seqpos()-1])
        pssm_dict[ch]["pssm_log_odds"].append(pssm_log_odds_list[res.seqpos()-1])
    return pssm_dict

