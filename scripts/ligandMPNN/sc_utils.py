from __future__ import print_function
import json, time, os, sys, glob
import shutil
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
import torch.distributions as D


import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import itertools


def build_sc_model(checkpoint_path):
    import json, time, os, sys, glob
    import shutil
    import numpy as np
    import torch
    from torch import optim
    import random
    from dateutil import parser
    import csv
    import copy
    import math
    from typing import Dict
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.utils
    import torch.utils.checkpoint
    import queue
    import torch.distributions as D

    sys.path.append("/home/justas/openfold")

    from openfold.np import protein
    import openfold.np.residue_constants as rc
    from openfold.utils.rigid_utils import Rotation, Rigid
    from openfold.utils.tensor_utils import (
             batched_gather,
             one_hot,
             tree_map,
             tensor_tree_map,
         )
    from openfold.utils import feats
    from openfold.data.data_transforms import make_atom14_masks, atom37_to_torsion_angles
    from openfold.np.residue_constants import (
            restype_rigid_group_default_frame,
            restype_atom14_to_rigid_group,
            restype_atom14_mask,
            restype_atom14_rigid_group_positions,
            restype_atom37_to_rigid_group
        )
    from openfold.np.protein import Protein, to_pdb, from_pdb_string


    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = Packer(num_mix=3,
                   node_features=128,
                   edge_features=128,
                   hidden_dim=128,
                   num_encoder_layers=3,
                   num_decoder_layers=3,
                   k_neighbors=32,
                   augment_eps=0.00,
                   device=device)

    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def pack_side_chains(num_packs, model, S, xyz4, X_m, Y, Y_m, Z, Z_m, Z_t, mask, name_, base_folder, residue_idx, chain_encoding_all):
    import json, time, os, sys, glob
    import shutil
    import numpy as np
    import torch
    from torch import optim
    import random
    from dateutil import parser 
    import csv
    import copy
    import math
    from typing import Dict
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.utils
    import torch.utils.checkpoint
    import queue
    import torch.distributions as D

    sys.path.append("/home/justas/openfold")

    from openfold.np import protein
    import openfold.np.residue_constants as rc
    from openfold.utils.rigid_utils import Rotation, Rigid
    from openfold.utils.tensor_utils import (
	     batched_gather,
	     one_hot,
	     tree_map,
	     tensor_tree_map,
	 )
    from openfold.utils import feats
    from openfold.data.data_transforms import make_atom14_masks, atom37_to_torsion_angles
    from openfold.np.residue_constants import (
	    restype_rigid_group_default_frame,
	    restype_atom14_to_rigid_group,
	    restype_atom14_mask,
	    restype_atom14_rigid_group_positions,
            restype_atom37_to_rigid_group
	)
    from openfold.np.protein import Protein, to_pdb, from_pdb_string
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    map_mpnn_to_af2_seq = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], device=device)

    chi_pi_periodic = [
        [0.0, 0.0, 0.0, 0.0],  # ALA
        [0.0, 0.0, 0.0, 0.0],  # ARG
        [0.0, 0.0, 0.0, 0.0],  # ASN
        [0.0, 1.0, 0.0, 0.0],  # ASP
        [0.0, 0.0, 0.0, 0.0],  # CYS
        [0.0, 0.0, 0.0, 0.0],  # GLN
        [0.0, 0.0, 1.0, 0.0],  # GLU
        [0.0, 0.0, 0.0, 0.0],  # GLY
        [0.0, 0.0, 0.0, 0.0],  # HIS
        [0.0, 0.0, 0.0, 0.0],  # ILE
        [0.0, 0.0, 0.0, 0.0],  # LEU
        [0.0, 0.0, 0.0, 0.0],  # LYS
        [0.0, 0.0, 0.0, 0.0],  # MET
        [0.0, 1.0, 0.0, 0.0],  # PHE
        [0.0, 0.0, 0.0, 0.0],  # PRO
        [0.0, 0.0, 0.0, 0.0],  # SER
        [0.0, 0.0, 0.0, 0.0],  # THR
        [0.0, 0.0, 0.0, 0.0],  # TRP
        [0.0, 1.0, 0.0, 0.0],  # TYR
        [0.0, 0.0, 0.0, 0.0],  # VAL
        [0.0, 0.0, 0.0, 0.0],  # UNK
    ]
    chi_pi_periodic=1.0-2*torch.tensor(chi_pi_periodic, device=device)

    predictions = {}
    sss = num_packs
    num_samples = np.maximum(2*num_packs,4)
    S_af2 = torch.argmax(torch.nn.functional.one_hot(S, 21).float() @ map_mpnn_to_af2_seq.float(), -1)
    with torch.no_grad():
        S_af2 = S_af2.repeat(num_samples,1)
        S = S.repeat(num_samples,1)
        xyz4 = xyz4.repeat(num_samples,1,1,1)
        mask = mask.repeat(num_samples,1)
        residue_idx = residue_idx.repeat(num_samples,1)
        chain_encoding_all = chain_encoding_all.repeat(num_samples,1)
        masks14_37 = make_atom14_masks({"aatype": S_af2})

        xyz37 = torch.zeros([S.shape[0], S.shape[1], 37, 3], device=device)
        xyz37[:,:,:3,:] = xyz4[:,:,:3,:] #N, Ca, C
        xyz37[:,:,4,:] = xyz4[:,:,3,:] #O
       
        temp_dict = {"aatype": S_af2,
                     "all_atom_positions": xyz37,
                     "all_atom_mask": masks14_37['atom37_atom_exists']}
        torsion_dict = atom37_to_torsion_angles("")(temp_dict)

        torsions_mask_ = torsion_dict["torsion_angles_mask"][:,:,3:]
        make_alt_torsions = chi_pi_periodic[S_af2,:][...,None]
        mean, concentration, mix_logits, predicted_real_values, xi_log_probs_out = model(make_alt_torsions, torsions_mask_, xyz37, X_m, Y, Y_m, Z, Z_m, Z_t,  S, mask, residue_idx, chain_encoding_all)
        xi_log_probs_mean = torch.sum(xi_log_probs_out*torsions_mask_,-1)/(torch.sum(torsions_mask_,-1)+1e-6) #[B, L]
        xi_value, xi_idx = torch.sort(xi_log_probs_mean, dim=0, descending=True) 
            
        mean = torch.gather(mean, dim=0, index=xi_idx[:sss,][...,None,None].repeat(1,1,4,3))
        concentration = torch.gather(concentration, dim=0, index=xi_idx[:sss,][...,None,None].repeat(1,1,4,3))
        mix_logits = torch.gather(mix_logits, dim=0, index=xi_idx[:sss,][...,None,None].repeat(1,1,4,3))
        predicted_real_values = torch.gather(predicted_real_values, dim=0, index=xi_idx[:sss,][...,None].repeat(1,1,4))

        mix = D.Categorical(logits=mix_logits)
        comp = D.VonMises(mean, concentration)
        pred_dist = D.MixtureSameFamily(mix, comp)

        torsion_pred_unit = torch.cat([torch.sin(predicted_real_values[:,:,:,None]), torch.cos(predicted_real_values[:,:,:,None])], -1)

        Lmax_p = pred_dist.log_prob(predicted_real_values) #[sss,L,4]
            
        tmp_types = torch.tensor(restype_atom37_to_rigid_group, device=device)[S_af2[:sss,]]
        mask_for_37_apply = (torch.clone(tmp_types)==0) #[B,L,37]
        tmp_types[tmp_types<4]=4
        tmp_types -= 4
        atom_types_for_b_factor = torch.nn.functional.one_hot(tmp_types,4) #[B, L, 37, 4] 
                                                        
        torsions_mask_sum = (torsion_dict["torsion_angles_mask"][:sss,:,3:].sum(-1)==0).float()

        uncertainty = Lmax_p[:,:,None,:]*atom_types_for_b_factor #[B,L,37,4]
        b_factor_pred = uncertainty.sum(-1) #[B, L, 37]
        b_factor_pred = (1-torsions_mask_sum[:,:,None])*b_factor_pred + torsions_mask_sum[:,:,None]*1.0
        b_factor_pred = torch.clip(b_factor_pred, -1.0, 2.0)

        rigids = Rigid.make_transform_from_reference(
                             n_xyz=xyz4[:sss,:,0,:],
                             ca_xyz=xyz4[:sss,:,1,:],
                             c_xyz=xyz4[:sss,:,2,:],
                             eps=1e-9)
    
        torsions_update = torch.clone(torsion_dict['torsion_angles_sin_cos'])[:sss,]
        torsions_update[:,:,3:] = torsion_pred_unit
        pred_frames = feats.torsion_angles_to_frames(rigids,
                                                            torsions_update,
                                                            S_af2[:sss,],
                                                            torch.tensor(restype_rigid_group_default_frame, device=device))
    
    
        atom14_pred = feats.frames_and_literature_positions_to_atom14_pos(pred_frames,
                                                                                 S_af2[:sss],
                                                                                 torch.tensor(restype_rigid_group_default_frame, device=device),
                                                                                 torch.tensor(restype_atom14_to_rigid_group, device=device),
                                                                                 torch.tensor(restype_atom14_mask, device=device),
                                                                                 torch.tensor(restype_atom14_rigid_group_positions, device=device))
        masks14_37_sss = make_atom14_masks({"aatype": S_af2[:sss,]}) 
        xyz37_pred = feats.atom14_to_atom37(atom14_pred, masks14_37_sss)
        for k in range(sss):
            protein_pred = Protein(
                                    atom_positions=np.array(xyz37_pred[k].cpu().data.numpy()),
                                    atom_mask=np.array(masks14_37['atom37_atom_exists'][k].cpu().data.numpy()),  #all_res_mask or X_sc_37 instead??
                                    aatype=np.array(S_af2[k].cpu().data.numpy()),
                                    residue_index=residue_idx[k].detach().cpu().numpy(),
                                    chain_index=chain_encoding_all.detach().cpu().numpy()[k,],
                                    b_factors=b_factor_pred[k]
                                )   
        
            idx = np.argwhere(mask[k].detach().cpu().numpy()==1)
            if base_folder is not None and name_ is not None:
                with open(f'{base_folder}/{name_}_{k}.pdb', 'w') as fp:
                    fp.write(to_pdb(protein_pred))
            else:
                predictions[sss] = protein_pred
    if base_folder is None and name_ is None:
        return predictions


def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features

def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features

def gather_nodes_t(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        #h_EV = self.norm1(h_EV)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        #dh = self.norm2(h_V)
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)

        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        #h_EV = self.norm3(h_EV)

        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))

        return h_V, h_E

class MPNNLayerJ(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(MPNNLayerJ, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,-1, h_E.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V


class MPNNLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(MPNNLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))
        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()
    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h

class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2*max_relative_feature+1+1, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(offset + self.max_relative_feature, 0, 2*self.max_relative_feature)*mask + (1-mask)*(2*self.max_relative_feature+1)
        d_onehot = torch.nn.functional.one_hot(d, 2*self.max_relative_feature+1+1)
        E = self.linear(d_onehot.float())
        return E
    




class ProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
        num_rbf=16, num_rbf_sc=8,top_k=30, augment_eps=0., Y_eps=0., num_chain_embeddings=16, device=None, side_residue_num=32, atom_context_num=25):
        """ Extract protein features """
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.Y_eps = Y_eps
        self.num_rbf = num_rbf
        self.num_rbf_side = num_rbf_sc
        self.atom_context_num = atom_context_num
        self.num_positional_embeddings = num_positional_embeddings
        self.side_residue_num = side_residue_num
        # Positional encoding
        self.embeddings = PositionalEncodings(num_positional_embeddings)
        # Normalization and embedding
        node_in, edge_in = 6, num_positional_embeddings + num_rbf*25
        self.node_project_down = nn.Linear(5*num_rbf+105, node_features, bias=False)
        #self.node_embedding = nn.Linear(128, node_features, bias=False) #NOT USED
        self.j_nodes = nn.Linear(105, node_features, bias=False)
        self.j_edges = nn.Linear(num_rbf, node_features, bias=False)


        self.norm_j_edges = nn.LayerNorm(node_features)
        self.norm_j_nodes = nn.LayerNorm(node_features)


        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_nodes = nn.LayerNorm(node_features)
        self.norm_edges = nn.LayerNorm(edge_features)
        self.edge_embedding_s = nn.Linear(num_rbf_sc*31*5, edge_features, bias=False)

        #element_dict = {"C": 0, "N": 1, "O": 2, "P": 3, "S": 4}
        #dna_rna_atom_types = ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "N2", "N3", "C4", "C5", "C6", "N7", "C8", "N9", "O4", "O2", "N4", "C7"]
        #atom_list = [
        #'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
        #'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
        #'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
        #'CZ3', 'NZ', 'OXT']
        self.DNA_RNA_types = torch.tensor([3, 2, 2, 2, 0, 0, 2, 0, 2, 0, 2, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 2, 2, 1, 0], device=device)
        self.side_chain_atom_types = torch.tensor([1, 0, 0, 2, 0, 0, 0, 0, 2, 2, 4, 0, 0, 0, 1, 1, 2, 2, 4, 0, 0, 0, 0, 1, 1, 1, 2, 2, 0, 1, 1, 2, 0, 0, 0, 1, 2], device=device)

    def _dist(self, X, mask, eps=1E-6):
        """ Pairwise euclidean distances """
        # Convolutional network on NCHW
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)

        # Identify k nearest neighbors (including self)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        sampled_top_k = self.top_k
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
        mask_neighbors = gather_edges(mask_2D.unsqueeze(-1), E_idx)
        return D_neighbors, E_idx, mask_neighbors

    def _rbf_side(self, D):
        # Distance radial basis function
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf_side
        D_mu = torch.linspace(D_min, D_max, D_count).to(device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    def _rbf(self, D):
        # Distance radial basis function
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count).to(device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6) #[B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0] #[B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B



    def _get_rbf_side(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6) #[B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0] #[B,L,K]
        RBF_A_B = self._rbf_side(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, Z, Z_m, Z_t, X, Y, Y_m, mask, atom_mask, residue_idx, chain_labels):
        """ Featurize coordinates as an attributed graph """
        device = X.device
        #if self.training and self.augment_eps > 0:
        #    X = X + self.augment_eps * torch.randn_like(X)
        #    Y = Y + self.Y_eps * torch.randn_like(Y)
        #    Z = Z + self.Y_eps * torch.randn_like(Z)

        b = X[:,:,1,:] - X[:,:,0,:]
        c = X[:,:,2,:] - X[:,:,1,:]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + X[:,:,1,:]
        Ca = X[:,:,1,:]
        N = X[:,:,0,:]
        C = X[:,:,2,:]
        O = X[:,:,4,:]

        #N, Ca, C, O, Cb - are five atoms representing a residue

        #Get neighbors
        D_neighbors, E_idx, mask_neighbors = self._dist(Ca, mask)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors)) #Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx)) #N-N
        RBF_all.append(self._get_rbf(C, C, E_idx)) #C-C
        RBF_all.append(self._get_rbf(O, O, E_idx)) #O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx)) #Cb-Cb

        RBF_all.append(self._get_rbf(Ca, N, E_idx)) #Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx)) #Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx)) #Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx)) #Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx)) #N-C
        RBF_all.append(self._get_rbf(N, O, E_idx)) #N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx)) #N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx)) #Cb-C
        RBF_all.append(self._get_rbf(Cb, O, E_idx)) #Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx)) #O-C

        RBF_all.append(self._get_rbf(N, Ca, E_idx)) #N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx)) #C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx)) #O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx)) #Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx)) #C-N
        RBF_all.append(self._get_rbf(O, N, E_idx)) #O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx)) #Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx)) #C-Cb


        RBF_all.append(self._get_rbf(O, Cb, E_idx)) #O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx)) #C-O

        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        offset = residue_idx[:,:,None]-residue_idx[:,None,:]
        offset = gather_edges(offset[:,:,:,None], E_idx)[:,:,:,0] #[B, L, K]

        d_chains = 1+0*((chain_labels[:, :, None] - chain_labels[:,None,:])==0).long()
        E_chains = gather_edges(d_chains[:,:,:,None], E_idx)[:,:,:,0]
        E_positional = self.embeddings(offset, E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)



        E_idx_sub = E_idx[:,:,:self.side_residue_num] #[B, L, 15]
        #RBF_sidechain = []
        #R_m = gather_nodes(atom_mask[:,:,5:], E_idx_sub) #[B, L, K, 31]
        #X_sidechain = X[:,:,5:,:].view(X.shape[0], X.shape[1], -1)
        #R = gather_nodes(X_sidechain, E_idx_sub).view(X.shape[0], X.shape[1], self.side_residue_num, -1, 3) #[B, L, 15, 9, 3]

        Y_t = self.DNA_RNA_types[None,None,None,:].repeat(Y.shape[0], Y.shape[1], Y.shape[2], 1) #[B, L, 10, 26]
        #R_t = self.side_chain_atom_types[None,None,None,5:].repeat(X.shape[0], X.shape[1], self.side_residue_num, 1) #[B, L, 25, 46]
        #R - [B, L, 15, 31, 3]
        #R_m - [B, L, 15, 31]
        #R_t - [B, L, 15, 31]

        #Y - [B, L, 10, 26, 3]
        #Y_m  - [B, L, 10, 26]
        #Y_t - [B, L, 10, 26]

        #R = R.view(X.shape[0], X.shape[1], -1, 3)
        #R_m = R_m.view(X.shape[0], X.shape[1], -1)
        #R_t = R_t.view(X.shape[0], X.shape[1], -1)

        Y = Y.view(X.shape[0], X.shape[1], -1, 3)
        Y_m = Y_m.view(X.shape[0], X.shape[1], -1)
        Y_t = Y_t.view(X.shape[0], X.shape[1], -1)

        J = torch.cat([Y, Z[:,None,:,:].repeat(1,Y.shape[1], 1, 1)], -2)
        J_m = torch.cat([Y_m, Z_m[:,None,:].repeat(1, Y.shape[1], 1)], -1)
        J_t = torch.cat([Y_t, Z_t[:,None,:].repeat(1, Y.shape[1], 1)], -1)

        #J = torch.cat((R, Y), 2) #[B, L, atoms, 3]
        #J_m = torch.cat((R_m, Y_m), 2) #[B, L, atoms]
        #J_t = torch.cat((R_t, Y_t), 2) #[B, L, atoms]

        Cb_J_distances = torch.sqrt(torch.sum((Cb[:,:,None,:] - J)**2,-1) + 1e-6) #[B, L, num_atoms]
        mask_J = mask[:,:,None]*J_m
        Cb_J_distances_adjusted = Cb_J_distances*mask_J+(1. - mask_J)*10000.0
        D_J, E_idx_J = torch.topk(Cb_J_distances_adjusted, self.atom_context_num, dim=-1, largest=False) #pick 25 closest atoms

        mask_far_atoms = (D_J < 20.0).float() #[B, L, K]

        J_picked_ = torch.gather(J, 2, E_idx_J[:,:,:,None].repeat(1,1,1,3)) #[B, L, 50, 3]
        num_atoms_batch = J_picked_.shape[2]
        J_t_picked_ = torch.gather(J_t, 2, E_idx_J) #[B, L, 50]
        J_m_picked_ = torch.gather(mask_J, 2, E_idx_J) #[B, L, 50]
        J_t_1hot_ = torch.nn.functional.one_hot(J_t_picked_, 105) #N, C, O, P, S #[B, L, 50, 4]

        J_picked = torch.zeros([X.shape[0], X.shape[1], self.atom_context_num, 3], device=device)
        J_m_picked = torch.zeros([X.shape[0], X.shape[1], self.atom_context_num], device=device)
        J_t_1hot = torch.zeros([X.shape[0], X.shape[1], self.atom_context_num, 105], device=device)

        J_picked[:,:,:num_atoms_batch,:] = J_picked_
        J_m_picked[:,:,:num_atoms_batch] = J_m_picked_
        J_t_1hot[:,:,:num_atoms_batch,:] = J_t_1hot_

        intermediate = torch.sqrt(torch.sum((J_picked[:,:,:,None,:] - J_picked[:,:,None,:,:])**2,-1) + 1e-5)
        J_edges = self._rbf(torch.sqrt(torch.sum((J_picked[:,:,:,None,:] - J_picked[:,:,None,:,:])**2,-1) + 1e-6)) #[B, L, 50, 50, num_bins]

        RBF_DNA = []

        D_N_J = self._rbf(torch.sqrt(torch.sum((N[:,:,None,:] - J_picked)**2,-1) + 1e-6)) #[B, L, 50, num_bins]
        D_Ca_J = self._rbf(torch.sqrt(torch.sum((Ca[:,:,None,:] - J_picked)**2,-1) + 1e-6))
        D_C_J = self._rbf(torch.sqrt(torch.sum((C[:,:,None,:] - J_picked)**2,-1) + 1e-6))
        D_O_J = self._rbf(torch.sqrt(torch.sum((O[:,:,None,:] - J_picked)**2,-1) + 1e-6))
        D_Cb_J = self._rbf(torch.sqrt(torch.sum((Cb[:,:,None,:] - J_picked)**2,-1) + 1e-6))

        D_all = torch.cat((D_N_J, D_Ca_J, D_C_J, D_O_J, D_Cb_J, J_t_1hot), dim=-1) #[B,L,25,5*num_bins+5]
        D_all = D_all*J_m_picked[:,:,:,None]*mask_far_atoms[:,:,:,None]

        V = self.node_project_down(D_all) #[B, L, 50, 32]
        V = self.norm_nodes(V)

        J_node_mask = J_m_picked*mask_far_atoms


        J_edges =  self.j_edges(J_edges)
        J_nodes = self.j_nodes(J_t_1hot.float())

        J_edges = self.norm_j_edges(J_edges)
        J_nodes = self.norm_j_nodes(J_nodes)


        return V, E, E_idx, J_node_mask, J_nodes, J_edges


class Packer(nn.Module):
    def __init__(self, node_features=128, edge_features=128,
        hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3, atom_context_num = 25,
        k_neighbors=32, augment_eps=0.00, dropout=0.0, vocab=21, num_mix=3, device=None):
        super(Packer, self).__init__()

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim


        self.V_C = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V_C_norm = nn.LayerNorm(hidden_dim)

        self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        self.W_c = nn.Linear(hidden_dim, hidden_dim, bias=True)


        self.W_nodes_j = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_edges_j = nn.Linear(hidden_dim, hidden_dim, bias=True)


        # Featurization layers
        self.features = ProteinFeatures(node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps, Y_eps=0.0, device=device, atom_context_num=atom_context_num)
        self.softmax = nn.LogSoftmax(-1)
        self.softplus = nn.Softplus(beta=1, threshold=20)
        # Embedding layers
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)
        self.W_torsions = nn.Linear(hidden_dim, 4*3*num_mix, bias=True) #out=W@x+b
        self.num_mix = num_mix

        self.W_xi1 = nn.Linear(hidden_dim, 3*num_mix, bias=True)
        self.W_xi2 = nn.Linear(hidden_dim, 3*num_mix, bias=True)
        self.W_xi3 = nn.Linear(hidden_dim, 3*num_mix, bias=True)
        self.W_xi4 = nn.Linear(hidden_dim, 3*num_mix, bias=True)

        self.LN_xi1 = nn.LayerNorm(hidden_dim)
        self.LN_xi2 = nn.LayerNorm(hidden_dim)
        self.LN_xi3 = nn.LayerNorm(hidden_dim)


        self.W_xi1p = nn.Linear(4, hidden_dim, bias=True)
        self.W_xi2p = nn.Linear(4, hidden_dim, bias=True)
        self.W_xi3p = nn.Linear(4, hidden_dim, bias=True)

        self.W_vxi1 = nn.Linear(hidden_dim*2, hidden_dim, bias=True)
        self.W_vxi2 = nn.Linear(hidden_dim*2, hidden_dim, bias=True)
        self.W_vxi3 = nn.Linear(hidden_dim*2, hidden_dim, bias=True)

        self.context_encoder_layers = nn.ModuleList([
            MPNNLayer(hidden_dim, hidden_dim*2, dropout=dropout)
            for _ in range(2)
        ])


        self.j_context_encoder_layers = nn.ModuleList([
            MPNNLayerJ(hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(2)
        ])



        #[B, L, 4, 36] - [nn.Linear(hidden_dim, 4*36, bias=True)]
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim*2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            MPNNLayer(hidden_dim, hidden_dim*2, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])

        #self.W_out = nn.Linear(hidden_dim, 21, bias=True) #NOT USED
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, make_alt_torsions, torsion_mask, X, X_m, Y, Y_m, Z, Z_m, Z_t, S, mask, residue_idx, chain_encoding_all, input_torsion_sin_cos=None, use_input_angles=0):
        """ Graph-conditioned sequence model """
        
        mask = mask[:1]
        #E, E_idx = self.features(X[:1,], mask[:1,], residue_idx[:1,], chain_encoding_all[:1,])
        V,  E, E_idx, J_m, J_nodes, J_edges = self.features(Z[:1], Z_m[:1], Z_t[:1], X[:1], Y[:1], Y_m[:1], mask, X_m[:1]*0, residue_idx[:1], chain_encoding_all[:1]) #sc atom mask is ZEROS
        h_V = self.W_s(S[:1,])
        h_E = self.W_e(E)
         
        h_E_context = self.W_v(V)
        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask[:1,].unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask[:1,].unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask[:1,], mask_attend)

      
        h_V_C = self.W_c(h_V)
        J_m_edges = J_m[:,:,:,None]*J_m[:,:,None,:]
        J_nodes = self.W_nodes_j(J_nodes)
        J_edges = self.W_edges_j(J_edges)
        for i in range(len(self.context_encoder_layers)):
            J_nodes = self.j_context_encoder_layers[i](J_nodes, J_edges, J_m, J_m_edges)
            h_E_context_cat = torch.cat([h_E_context, J_nodes], -1)
            h_V_C = self.context_encoder_layers[i](h_V_C, h_E_context_cat, mask, J_m)


        h_V_C = self.V_C(h_V_C)
        h_V = h_V + self.V_C_norm(h_V_C)




         
        h_V = h_V.repeat(X.shape[0],1,1)
        h_E = h_E.repeat(X.shape[0],1,1,1)
        E_idx = E_idx.repeat(X.shape[0],1,1)
#Xi1---------------------
        h_V_copy = torch.clone(h_V)
        for layer in self.decoder_layers:
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V = layer(h_V, h_EV, mask)


        xi1 = self.W_xi1(h_V)
        xi1 = xi1.reshape(h_V.shape[0],h_V.shape[1], self.num_mix, 3)
        m_xi1 = xi1[:,:,:,0].float()
        c_xi1 = 0.1 + self.softplus(xi1[:,:,:,1]).float()
        mix_xi1 = xi1[:,:,:,2].float()

        mix1 = D.Categorical(logits=mix_xi1)
        comp1 = D.VonMises(m_xi1, c_xi1)
        pred_dist1 = D.MixtureSameFamily(mix1, comp1)

        xi1_sampled = pred_dist1.sample([1])
        log_probs_xi1 = pred_dist1.log_prob(xi1_sampled)

        xi1_log_probs_vals, xi1_log_probs_idx = torch.sort(log_probs_xi1,0)

        xi1_output = torch.gather(xi1_sampled, dim=0, index=xi1_log_probs_idx[-1:,])[0,]
        xi1_log_probs_vals_out = xi1_log_probs_vals[-1,] #[B, L]
        
        xi1_sin_cos = torch.cat([torch.sin(xi1_output[:,:,None]), torch.cos(xi1_output[:,:,None])], -1)

        if use_input_angles:
            xi1_sin_cos = input_torsion_sin_cos[:,:,0,:]

        xi1_sin_cos = torch.cat([xi1_sin_cos, xi1_sin_cos*make_alt_torsions[:,:,0]], -1) #[B, L, 4, 4]
        xi1_sin_cos = xi1_sin_cos*torsion_mask[:,:,0,None]
        
        h_xi1 = self.W_xi1p(xi1_sin_cos)
        h_Vxi1 = torch.cat([h_V, h_xi1], -1)
        h_V = self.W_vxi1(h_Vxi1)
#Xi2---------------------
        h_V = h_V_copy + self.LN_xi1(h_V)
        h_V_copy = torch.clone(h_V)
        for layer in self.decoder_layers:
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V = layer(h_V, h_EV, mask)

        xi2 = self.W_xi2(h_V)
        xi2 = xi2.reshape(h_V.shape[0],h_V.shape[1], self.num_mix, 3)
        m_xi2 = xi2[:,:,:,0].float()
        c_xi2 = 0.1 + self.softplus(xi2[:,:,:,1]).float()
        mix_xi2 = xi2[:,:,:,2].float()

        mix2 = D.Categorical(logits=mix_xi2)
        comp2 = D.VonMises(m_xi2, c_xi2)
        pred_dist2 = D.MixtureSameFamily(mix2, comp2)

        xi2_sampled = pred_dist2.sample([1])
        log_probs_xi2 = pred_dist2.log_prob(xi2_sampled)

        xi2_log_probs_vals, xi2_log_probs_idx = torch.sort(log_probs_xi2,0)
        
        xi2_output = torch.gather(xi2_sampled, dim=0, index=xi2_log_probs_idx[-1:,])[0,]
        xi2_log_probs_vals_out = xi2_log_probs_vals[-1,] #[B, L]

        xi2_sin_cos = torch.cat([torch.sin(xi2_output[:,:,None]), torch.cos(xi2_output[:,:,None])], -1)


        if use_input_angles:
            xi2_sin_cos = input_torsion_sin_cos[:,:,1,:]


        xi2_sin_cos = torch.cat([xi2_sin_cos, xi2_sin_cos*make_alt_torsions[:,:,1]], -1) #[B, L, 4, 4]
        xi2_sin_cos = xi2_sin_cos*torsion_mask[:,:,1,None]

        h_xi2 = self.W_xi2p(xi2_sin_cos)
        h_Vxi2 = torch.cat([h_V, h_xi2], -1)
        h_V = self.W_vxi2(h_Vxi2)

#Xi3---------------------
        h_V = h_V_copy + self.LN_xi2(h_V)
        h_V_copy = torch.clone(h_V)
        for layer in self.decoder_layers:
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V = layer(h_V, h_EV, mask)


        xi3 = self.W_xi3(h_V)
        xi3 = xi3.reshape(h_V.shape[0],h_V.shape[1], self.num_mix, 3)
        m_xi3 = xi3[:,:,:,0].float()
        c_xi3 = 0.1 + self.softplus(xi3[:,:,:,1]).float()
        mix_xi3 = xi3[:,:,:,2].float()

        mix3 = D.Categorical(logits=mix_xi3)
        comp3 = D.VonMises(m_xi3, c_xi3)
        pred_dist3 = D.MixtureSameFamily(mix3, comp3)

        xi3_sampled = pred_dist3.sample([1])
        log_probs_xi3 = pred_dist3.log_prob(xi3_sampled)

        xi3_log_probs_vals, xi3_log_probs_idx = torch.sort(log_probs_xi3,0)
        
        xi3_output = torch.gather(xi3_sampled, dim=0, index=xi3_log_probs_idx[-1:,])[0,]
        xi3_log_probs_vals_out = xi3_log_probs_vals[-1,] #[B, L]

        xi3_sin_cos = torch.cat([torch.sin(xi3_output[:,:,None]), torch.cos(xi3_output[:,:,None])], -1)


        if use_input_angles:
            xi3_sin_cos = input_torsion_sin_cos[:,:,2,:]

        xi3_sin_cos = torch.cat([xi3_sin_cos, xi3_sin_cos*make_alt_torsions[:,:,2]], -1) #[B, L, 4, 4]
        xi3_sin_cos = xi3_sin_cos*torsion_mask[:,:,2,None]

        h_xi3 = self.W_xi3p(xi3_sin_cos)
        h_Vxi3 = torch.cat([h_V, h_xi3], -1)
        h_V = self.W_vxi3(h_Vxi3)

#Xi4---------------------
#Xi4---------------------
        h_V = h_V_copy + self.LN_xi3(h_V)
        h_V_copy = torch.clone(h_V)
        for layer in self.decoder_layers:
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V = layer(h_V, h_EV, mask)


        xi4 = self.W_xi4(h_V)
        xi4 = xi4.reshape(h_V.shape[0],h_V.shape[1], self.num_mix, 3)
        m_xi4 = xi4[:,:,:,0].float()
        c_xi4 = 0.1 + self.softplus(xi4[:,:,:,1]).float()
        mix_xi4 = xi4[:,:,:,2].float()

        mix4 = D.Categorical(logits=mix_xi4)
        comp4 = D.VonMises(m_xi4, c_xi4)
        pred_dist4 = D.MixtureSameFamily(mix4, comp4)

        xi4_sampled = pred_dist4.sample([1])
        log_probs_xi4 = pred_dist4.log_prob(xi4_sampled)

        xi4_log_probs_vals, xi4_log_probs_idx = torch.sort(log_probs_xi4,0)
        
        xi4_output = torch.gather(xi4_sampled, dim=0, index=xi4_log_probs_idx[-1:,])[0,]
        xi4_log_probs_vals_out = xi4_log_probs_vals[-1,] #[B, L]


        mean = torch.cat([m_xi1[:,:,None,:], m_xi2[:,:,None,:], m_xi3[:,:,None,:], m_xi4[:,:,None,:]], -2)
        concentration = torch.cat([c_xi1[:,:,None,:], c_xi2[:,:,None,:], c_xi3[:,:,None,:], c_xi4[:,:,None,:]], -2)
        mix_logits = torch.cat([mix_xi1[:,:,None,:], mix_xi2[:,:,None,:], mix_xi3[:,:,None,:], mix_xi4[:,:,None,:]], -2)

        sample = torch.cat([xi1_output[:,:,None],  xi2_output[:,:,None], xi3_output[:,:,None], xi4_output[:,:,None]], -1)

        xi_log_probs_out = torch.cat([xi1_log_probs_vals_out[:,:,None], xi2_log_probs_vals_out[:,:,None], xi3_log_probs_vals_out[:,:,None], xi4_log_probs_vals_out[:,:,None]], -1)
        return mean, concentration, mix_logits, sample, xi_log_probs_out
