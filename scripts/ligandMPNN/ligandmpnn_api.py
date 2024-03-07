import argparse
import os.path
import json, time, os, sys, glob
import shutil
import warnings
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import os.path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from protein_mpnn_utils import loss_nll, loss_smoothed, gather_edges, gather_nodes, gather_nodes_t, cat_neighbors_nodes, _scores, _S_to_seq, tied_featurize, parse_PDB
from protein_mpnn_utils import StructureDataset, StructureDatasetPDB, ProteinMPNN
from sc_utils import pack_side_chains, build_sc_model
if os.path.exists("/net/software"):
    sys.path.append("/net/software/lab/openfold")
else:
    sys.path.append("/home/indrek/UW_Digs/net/software/lab/openfold")
from openfold.np import protein
from pyrosetta_tools import parser_tools


init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']


class MPNNRunner(object):
    def __init__(self, checkpoint_path=None, backbone_noise=0.0):
        DEFAULT_CHECKPOINT = os.path.dirname(os.path.realpath(__file__)) + "/model_weights/v_32_020.pt"

        self.__checkpoint = checkpoint_path
        if checkpoint_path is None:
            self.__checkpoint = DEFAULT_CHECKPOINT
        print(f"Using model from checkpoint: {self.__checkpoint}")

        # Currently hardcoded sc-packing model path
        if os.path.exists("/databases/mpnn"):
            self.__checkpoint_path_sc = "/databases/mpnn/ligand_model_weights/sc_packing/v_32_005.pt"
        else:
            self.__checkpoint_path_sc = "/home/indrek/software/proteinmpnn/ligandMPNN/model_weights/sc_packing/v_32_005.pt"
            if not os.path.exists(self.__checkpoint_path_sc):
                sys.exit("Cannot find an existing sc-packing checkpoint file")

        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

        hidden_dim = 128
        num_layers = 3
        num_edges = 48

        checkpoint = torch.load(self.__checkpoint, map_location=self.device) 

        # noise_level_print = checkpoint['noise_level']
        # print(f'Training noise level: {noise_level_print}A')

        self.model = ProteinMPNN(num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim,
                                 num_encoder_layers=num_layers, num_decoder_layers=num_layers, augment_eps=backbone_noise,
                                 k_neighbors=num_edges, device=self.device)

        self.model.to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.sc_model = None
        pass

    def checkpoint(self):
        return self.__checkpoint


    class MPNN_Input(object):
        def __init__(self):
            self.structure = None
            self.chain_id = None
            self.fixed_positions = None
            self.omit_AA = None
            self.tied_positions = None
            self.pssm = None
            self.bias_by_res = None
            self.omit_AAs_list = []
            self.bias_AA = None
            self.temperatures = None

            self.designed_chains = None
            self.fixed_chains = None
            self.name = None
            self.max_length = 20000
            pass

        
        def validate(self) -> bool:
            """
            Checks the input object for consistency
            Returns
            -------
            bool
                True if all good. False is something is wrong.

            """
            if self.structure is None:
                print("No structure defined")
                return False
            # if not any(["all_atom" in k for k in self.structure.keys()]):
            #     print("Not all-atom input")
            #     return False
            if self.name is None:
                self.name = self.structure["name"]
            if self.designed_chains is None and self.fixed_chains is None and self.chain_id is None:
                all_chain_list = [item[-1:] for item in self.structure.keys() if item[:9]=='seq_chain'] #['A','B', 'C',...]
                designed_chain_list = all_chain_list
                self.designed_chains = [_ for _ in designed_chain_list]
                self.fixed_chain_list = [letter for letter in all_chain_list if letter not in designed_chain_list]

            dataset_valid = StructureDatasetPDB([self.structure], truncate=None, max_length=self.max_length)
            if len(dataset_valid) == 0:
                return False
            else:
                self.structure = dataset_valid[0]
            return True

        def __str__(self):
            report = ""
            for k in self.__dir__():
                if "__" not in k and k not in ["validate", "structure"]:
                    report += f"{k}: {self.__getattribute__(k)}\n"
            return report


    class MPNNOutput(object):
        def __init__(self, runner):
            self.__checkpoint = runner.checkpoint()
            self.__seqs = None
            self.__df = None
            self.__temperatures = None
            self.__mpnn = "FA"
            self.__packed = None


        def _create(self, mpnn_input, seqs):
            self.__seqs = seqs
            self.__temperatures = list(self.seqs.keys())
            self.__df = pd.DataFrame()
            n = 0
            for temp in seqs:
                for j in seqs[temp]:
                    for b_ix, item in enumerate(seqs[temp][j]):
                        for k, val in item.items():
                            if k == "packed":
                                continue
                            self.__df.at[n, k] = val

                        if item["native"] is True:
                            name_ = mpnn_input.name + "_native"
                        else:
                            name_ = mpnn_input.name + f"_T{temp}_s{b_ix}_{j}"
                        self.__df.at[n, "name"] = name_
                        self.__df.at[n, "iter"] = j
                        self.__df.at[n, "temp"] = temp
                        if item["packed"] is not None:
                            if self.__packed is None:
                                self.__packed = {}
                            self.__packed[n] = item["packed"]

                        n += 1



        def __len__(self):
            return len(self.df)

        @property
        def df(self):
            return self.__df
        
        @property
        def seqs(self):
            return self.__seqs

        @property
        def temperatures(self):
            return self.__temperatures
        
        @property
        def mpnn(self):
            return self.__mpnn

        @property
        def packed(self):
            return self.__packed

        @property
        def checkpoint(self):
            return self.__checkpoint

        def to_pdb(self, packed_protein: protein.Protein):
            assert isinstance(packed_protein, protein.Protein), "Bad input type for pdb dumping"
            pdb_lines = protein.to_pdb(packed_protein)
            return pdb_lines

        def write_fasta(self, filename) -> None:
            with open(filename, "w") as file:
                for idx, row in self.df.iterrows():
                    file.write(">" + row["name"] + "\n")
                    file.write(row["seq"] + "\n")
        
        def __repr__(self):
            return self.df.__repr__()


    def run(self, input_obj, NUM_BATCHES, BATCH_COPIES=1, pssm_threshold=0.0,
            pssm_log_odds_flag=False, pssm_bias_flag=False, pssm_multi=0.0,
            use_sc=True, use_DNA_RNA=True, use_ligand=True, pack_sc=False, num_packs=1, **kwargs):

        if pack_sc is True and self.sc_model is None:
            self.sc_model = build_sc_model(self.__checkpoint_path_sc)

        alphabet = 'ACDEFGHIKLMNPQRSTVWYX'

        if not input_obj.validate():
            print("Input validation failed")
            return None

        # batch_clones = input_obj.batch_clones
        # device = input_obj.device
        name_ = input_obj.name
        protein = input_obj.structure
        temperatures = input_obj.temperatures

        omit_AA_dict = None
        fixed_positions_dict = None
        tied_positions_dict = None
        pssm_dict = None
        bias_by_res_dict = None
        bias_AA_dict = None

        if input_obj.fixed_positions is not None:
            fixed_positions_dict = {name_: input_obj.fixed_positions}
        if input_obj.omit_AA is not None:
            omit_AA_dict = {name_: input_obj.omit_AA}
        if input_obj.tied_positions is not None:
            tied_positions_dict = {name_: input_obj.tied_positions}
        if input_obj.pssm is not None:
            pssm_dict = {name_: input_obj.pssm}
        if input_obj.bias_by_res is not None:
            bias_by_res_dict = {name_: input_obj.bias_by_res}
        if input_obj.bias_AA is not None:
            bias_AA_dict = {name_: input_obj.bias_AA}

        batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]

        if input_obj.chain_id is None:
            chain_id_dict = {name_: (input_obj.designed_chains, input_obj.fixed_chains)}
        else:
            chain_id_dict = {name_: input_obj.chain_id}

        omit_AAs_np = np.array([AA in input_obj.omit_AAs_list for AA in alphabet]).astype(np.float32)

        bias_AAs_np = np.zeros(len(alphabet))
        if bias_AA_dict:
            for n, AA in enumerate(alphabet):
                if AA in list(bias_AA_dict.keys()):
                    bias_AAs_np[n] = bias_AA_dict[AA]

        print(f'Generating sequences for: {name_}')
        t0 = time.time()


        with torch.no_grad():
            test_sum, test_weights = 0., 0.

            Z, Z_m, Z_t, X, X_m, Y, Y_m, S, mask, lengths, chain_M, chain_encoding_all,\
             chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list,\
             chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list,\
             pssm_coef, pssm_bias, pssm_log_odds_all, tied_beta = tied_featurize(batch_clones, self.device, chain_id_dict, fixed_positions_dict, omit_AA_dict, tied_positions_dict, pssm_dict)

            Z_cloned = torch.clone(Z)
            if not use_sc:
                X_m = X_m * 0
            if not use_DNA_RNA:
                Y_m = Y_m * 0
            if not use_ligand:
                Z_m = Z_m * 0
            # if args.mask_hydrogen:
            #     mask_hydrogen = ~(Z_t == 40)  #1 for not hydrogen, 0 for hydrogen
            #     Z_m = Z_m*mask_hydrogen

            # if args.random_ligand_rotation > 0.01:
            #     R_for_ligand = torch.tensor(make_random_rotation(args.random_ligand_rotation), device=self.device, dtype=torch.float32)
            #     Z_mean = torch.sum(Z*Z_m[:,:,None],1)/torch.sum(Z_m[:,:,None],1)
            #     Z = torch.einsum('ij, blj -> bli', R_for_ligand, Z-Z_mean[:,None,:]) + Z_mean[:,None,:]
            
            # if args.random_ligand_translation > 0.01: 
            #     Z_random = args.random_ligand_translation*torch.rand([3], device=self.device)
            #     Z = Z + Z_random[None,None,:]
 
            RMSD = torch.sqrt(torch.sum(torch.sum((Z_cloned-Z)**2,-1)*Z_m, 1)/(torch.sum(Z_m,1)+1e-6)+1e-6)

            pssm_log_odds_mask = (pssm_log_odds_all > pssm_threshold).float() #1.0 for true, 0.0 for false

            # if pack_sc is True:
            #      pack_side_chains(num_packs, self.sc_model, S[0][None], X[:1,:,:4], X_m[:1], Y[:1], Y_m[:1], Z[:1], Z_m[:1], Z_t[:1], mask[0][None], None, None, residue_idx[0][None], chain_encoding_all[0][None])

            randn_1 = torch.randn(chain_M.shape, device=X.device)
            log_probs = self.model(X, X_m, Y, Y_m, Z, Z_m, Z_t, randn_1, S, chain_M*chain_M_pos, chain_encoding_all, residue_idx, mask)
            mask_for_loss = mask*chain_M*chain_M_pos
            scores = _scores(S, log_probs, mask_for_loss)
            native_score = scores.cpu().data.numpy()
            
            seqs = {}
            score_list = []
            all_probs_list = []
            all_log_probs_list = []
            S_sample_list = []

            # Generate some sequences
            for temp_c, temp in enumerate(temperatures):
                seqs[temp] = {}
                for j in range(NUM_BATCHES):
                    seqs[temp][j] = []
                    randn_2 = torch.randn(chain_M.shape, device=X.device)
                    if tied_positions_dict == None:
                        sample_dict = self.model.sample(X, X_m, Y, Y_m, Z, Z_m, Z_t, randn_2, S,
                                                        chain_M, chain_encoding_all, residue_idx, mask=mask,
                                                        temperature=temp, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np,
                                                        chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef,
                                                        pssm_bias=pssm_bias, pssm_multi=pssm_multi,
                                                        pssm_log_odds_flag=pssm_log_odds_flag,
                                                        pssm_log_odds_mask=pssm_log_odds_mask,
                                                        pssm_bias_flag=pssm_bias_flag)
                        S_sample = sample_dict["S"]
                    else:
                        sample_dict = self.model.tied_sample(X, X_m, Y, Y_m, Z, Z_m, Z_t, randn_2, S, chain_M, chain_encoding_all,
                                                             residue_idx, mask=mask, temperature=temp, omit_AAs_np=omit_AAs_np,
                                                             bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask,
                                                             pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=pssm_multi,
                                                             pssm_log_odds_flag=pssm_log_odds_flag, pssm_log_odds_mask=pssm_log_odds_mask,
                                                             pssm_bias_flag=pssm_bias_flag, tied_pos=tied_pos_list_of_lists_list[0],
                                                             tied_beta=tied_beta)
                    # Compute scores
                        S_sample = sample_dict["S"]
                    log_probs = self.model(X, X_m, Y, Y_m, Z, Z_m, Z_t, randn_2, S_sample, chain_M*chain_M_pos, chain_encoding_all, residue_idx, mask)
                    mask_for_loss = mask*chain_M*chain_M_pos
                    scores = _scores(S_sample, log_probs, mask_for_loss)
                    scores = scores.cpu().data.numpy()
                    all_probs_list.append(sample_dict["probs"].cpu().data.numpy())
                    all_log_probs_list.append(log_probs.cpu().data.numpy())
                    S_sample_list.append(S_sample.cpu().data.numpy())
                    for b_ix in range(BATCH_COPIES):
                        masked_chain_length_list = masked_chain_length_list_list[b_ix]
                        masked_list = masked_list_list[b_ix]
                        seq_recovery_rate = torch.sum(torch.sum(torch.nn.functional.one_hot(S[b_ix], 21)*torch.nn.functional.one_hot(S_sample[b_ix], 21),axis=-1)*mask_for_loss[b_ix])/torch.sum(mask_for_loss[b_ix])
                        seq = _S_to_seq(S_sample[b_ix], chain_M[b_ix])
                        score = scores[b_ix]
                        score_list.append(score)
                        native_seq = _S_to_seq(S[b_ix], chain_M[b_ix])
                        if b_ix == 0 and j==0 and temp==temperatures[0]:
                            start = 0
                            end = 0
                            list_of_AAs = []
                            for mask_l in masked_chain_length_list:
                                end += mask_l
                                list_of_AAs.append(native_seq[start:end])
                                start = end
                            native_seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
                            l0 = 0
                            for mc_length in list(np.array(masked_chain_length_list)[np.argsort(masked_list)])[:-1]:
                                l0 += mc_length
                                native_seq = native_seq[:l0] + '/' + native_seq[l0:]
                                l0 += 1
                            sorted_masked_chain_letters = np.argsort(masked_list_list[0])
                            # print_masked_chains = [masked_list_list[0][i] for i in sorted_masked_chain_letters]
                            sorted_visible_chain_letters = np.argsort(visible_list_list[0])
                            # print_visible_chains = [visible_list_list[0][i] for i in sorted_visible_chain_letters]
                            # native_score_print = np.format_float_positional(np.float32(native_score.mean()), unique=False, precision=4)
                            # ligand_rmsd = np.format_float_positional(np.float32(RMSD[0].cpu().detach()), unique=False, precision=3)
                            seqs[temp][j].append({"seq": native_seq,
                                                  "score": float(native_score[b_ix]),
                                                  "seq_recovery": 1.0,
                                                  "batch": b_ix,
                                                  "native": True,
                                                  "packed": None})

                        start = 0
                        end = 0
                        list_of_AAs = []
                        for mask_l in masked_chain_length_list:
                            end += mask_l
                            list_of_AAs.append(seq[start:end])
                            start = end

                        seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
                        l0 = 0
                        for mc_length in list(np.array(masked_chain_length_list)[np.argsort(masked_list)])[:-1]:
                            l0 += mc_length
                            seq = seq[:l0] + '/' + seq[l0:]
                            l0 += 1
                        # score_print = np.format_float_positional(np.float32(score), unique=False, precision=4)
                        # seq_rec_print = np.format_float_positional(np.float32(seq_recovery_rate.detach().cpu().numpy()), unique=False, precision=4)
                        # global_b_ix = b_ix + j*BATCH_COPIES + temp_c*BATCH_COPIES*NUM_BATCHES
                        packed_structs = None
                        if pack_sc is True:
                            print(f"Packing sequence T{temp}-{j}-{b_ix}")
                            packed_structs = pack_side_chains(num_packs, self.sc_model, S_sample[b_ix][None], X[b_ix,:,:4][None], X_m[b_ix][None], Y[b_ix][None], Y_m[b_ix][None], Z[b_ix][None], Z_m[b_ix][None], Z_t[b_ix][None], mask[b_ix][None], None, None, residue_idx[b_ix][None]+1, chain_encoding_all[b_ix][None]-1)
                        seqs[temp][j].append({"seq": seq,
                                              "score": float(score),
                                              "seq_recovery": float(seq_recovery_rate.detach().cpu().numpy()),
                                              "batch": b_ix,
                                              "native": False,
                                              "packed": packed_structs})


        t1 = time.time()
        dt = round(float(t1-t0), 4)
        num_seqs = len(temperatures)*NUM_BATCHES*BATCH_COPIES
        total_length = X.shape[1]
        print(f'{num_seqs} sequences of length {total_length} generated in {dt} seconds')
        torch.cuda.empty_cache()

        out = self.MPNNOutput(self)
        out._create(input_obj, seqs)

        return out
