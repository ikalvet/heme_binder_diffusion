import argparse

def main(args):

    from dateutil import parser
    import numpy as np
    import os, time, gzip, json
    import glob 
    
    folder_with_pdbs_path = args.input_path
    save_path = args.output_path
    
    #alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
    #states = len(alpha_1)
    #alpha_3 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
    #           'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','GAP']


    alpha_1 = list("ARNDCQEGHILKMFPSTWYV-atcgdryu")
    states = len(alpha_1)
    alpha_3 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
             'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','GAP',
              ' DA', ' DT', ' DC', ' DG',  '  A',  '  U',  '  C',  '  G']


    
    aa_1_N = {a:n for n,a in enumerate(alpha_1)}
    aa_3_N = {a:n for n,a in enumerate(alpha_3)}
    aa_N_1 = {n:a for n,a in enumerate(alpha_1)}
    aa_1_3 = {a:b for a,b in zip(alpha_1,alpha_3)}
    aa_3_1 = {b:a for a,b in zip(alpha_1,alpha_3)}
    
    def AA_to_N(x):
      # ["ARND"] -> [[0,1,2,3]]
      x = np.array(x);
      if x.ndim == 0: x = x[None]
      return [[aa_1_N.get(a, states-1) for a in y] for y in x]
    
    def N_to_AA(x):
      # [[0,1,2,3]] -> ["ARND"]
      x = np.array(x);
      if x.ndim == 1: x = x[None]
      return ["".join([aa_N_1.get(a,"-") for a in y]) for y in x]


    def parse_extra_res_fa_param(param_fn):
        atmName_to_atmType = {}
        with open(param_fn) as fp:
            for line in fp:
                x = line.strip().split()
                if line.startswith('NAME'):
                    lig_name = x[1]
                elif line.startswith('ATOM'):
                    atmName_to_atmType[x[1]] = x[2]
        ligName_to_atms = {lig_name:atmName_to_atmType}
        return ligName_to_atms
    
    
    def parse_PDB_biounits(x, atoms=['N','CA','C'], chain=None, lig_params=[]):
      '''
      input:  x = PDB filename
              atoms = atoms to extract (optional)
      output: (length, atoms, coords=(x,y,z)), sequence
      '''
    
      alpha_1 = list("ARNDCQEGHILKMFPSTWYV-atcgdryuJ")
      states = len(alpha_1)
      alpha_3 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
                 'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','GAP',
                  ' DA', ' DT', ' DC', ' DG',  '  A',  '  U',  '  C',  '  G',
                 'LIG']
      
      aa_1_N = {a:n for n,a in enumerate(alpha_1)}
      aa_3_N = {a:n for n,a in enumerate(alpha_3)}
      aa_N_1 = {n:a for n,a in enumerate(alpha_1)}
      aa_1_3 = {a:b for a,b in zip(alpha_1,alpha_3)}
      aa_3_1 = {b:a for a,b in zip(alpha_1,alpha_3)}
    
      lig_to_atms = {}
      if len(lig_params) > 0:
        for lig_param in lig_params:
            lig_to_atms.update(parse_extra_res_fa_param(lig_param))
      lig_names = list(lig_to_atms.keys())      
      
      def AA_to_N(x):
        # ["ARND"] -> [[0,1,2,3]]
        x = np.array(x);
        if x.ndim == 0: x = x[None]
        return [[aa_1_N.get(a, states-1) for a in y] for y in x]
      
      def N_to_AA(x):
        # [[0,1,2,3]] -> ["ARND"]
        x = np.array(x);
        if x.ndim == 1: x = x[None]
        return ["".join([aa_N_1.get(a,"-") for a in y]) for y in x]
    
      xyz,seq,min_resn,max_resn = {},{},1e6,-1e6
      for line in open(x,"rb"):
        line = line.decode("utf-8","ignore").rstrip()
    
        if line[:6] == "HETATM" and line[17:17+3] == "MSE":
          line = line.replace("HETATM","ATOM  ")
          line = line.replace("MSE","MET")
    
        #Ligands will start with HETATM but for noncanonial stuff (may start with ATOM ?)? GRL
        #Currently one chain should be just the ligand itself.
        parse_atm_line = False
        if len(lig_names) > 0 and line[17:20] in lig_names:
            parse_atm_line = True
        if line[:4] == "ATOM":
            parse_atm_line = True
            
        if (parse_atm_line):
          ch = line[21:22]
          if ch == chain or chain is None:
            atom = line[12:12+4].strip()
            resi = line[17:17+3]
            resn = line[22:22+5].strip()
            x,y,z = [float(line[i:(i+8)]) for i in [30,38,46]]
    
            if resn[-1].isalpha(): 
                resa,resn = resn[-1],int(resn[:-1])-1
            else: 
                resa,resn = "",int(resn)-1
    #         resn = int(resn)
            if resn < min_resn: 
                min_resn = resn
            if resn > max_resn: 
                max_resn = resn
            if resn not in xyz: 
                xyz[resn] = {}
            if resa not in xyz[resn]: 
                xyz[resn][resa] = {}
            if resn not in seq: 
                seq[resn] = {}
            #for alternative coords
            if resa not in seq[resn]: 
                seq[resn][resa] = resi
            if atom not in xyz[resn][resa]:
              xyz[resn][resa][atom] = np.array([x,y,z])
    
      # convert to numpy arrays, fill in missing values
      seq_,xyz_,atype_ = [],[],[]
      is_lig_chain = False
      try:
          resn_to_ligName = {}
          for resn in range( min_resn,max_resn+1):
            if resn in seq:
              #20: not in the list, treat as gap
              for k in sorted(seq[resn]):
                  resi = seq[resn][k]
                  if resi in lig_names:
                      resn_to_ligName[resn] = resi
                      is_lig_chain = True
                      seq_.append(aa_3_N.get('LIG',29)) ###GRL:hard-coding 29 ok?
                  else:
                      seq_.append(aa_3_N.get(seq[resn][k],20))
            else: seq_.append(20)
            #
            #
            if is_lig_chain:
                #Get new atoms list just for the ligand as defined in the params file
                atoms = list(lig_to_atms[resn_to_ligName[resn]].keys())
            #
            #Ligand atoms in the same order with xyz_ (matching atom name -> atype as defined in the params file)
            if resn in xyz:
              for k in sorted(xyz[resn]):
                for atom in atoms:
                  if atom in xyz[resn][k]: xyz_.append(xyz[resn][k][atom])
                  else: xyz_.append(np.full(3,np.nan))
            else:
              for atom in atoms: xyz_.append(np.full(3,np.nan))
            #
            if is_lig_chain:
                lig_atm_d = lig_to_atms[resn_to_ligName[resn]]
                for atom in atoms:
                    atype_.append(lig_atm_d[atom])
          return np.array(xyz_).reshape(-1,len(atoms),3), N_to_AA(np.array(seq_)), np.array(atype_)
      except TypeError:
          return 'no_chain', 'no_chain', 'no_chain'
    

    def parse_PDB(path_to_pdb,extra_res_param={}):
        #extra_res_param dictionary to tie pdb file to ligand params files
    
    
        ref_atype_to_element = {'CNH2': 'C', 'COO': 'C', 'CH0': 'C', 'CH1': 'C', 'CH2': 'C', 'CH3': 'C', 'aroC': 'C', 'Ntrp': 'N', 'Nhis': 'N', 'NtrR': 'N', 'NH2O': 'N', 'Nlys': 'N', 'Narg': 'N', 'Npro': 'N', 'OH': 'O', 'OW': 'O', 'ONH2': 'O', 'OOC': 'O', 'Oaro': 'O', 'Oet2': 'O', 'Oet3': 'O', 'S': 'S', 'SH1': 'S', 'Nbb': 'N', 'CAbb': 'C', 'CObb': 'C', 'OCbb': 'O', 'Phos': 'P', 'Pbb': 'P', 'Hpol': 'H', 'HS': 'H', 'Hapo': 'H', 'Haro': 'H', 'HNbb': 'H', 'Hwat': 'H', 'Owat': 'O', 'Opoint': 'O', 'HOH': 'O', 'Bsp2': 'B', 'F': 'F', 'Cl': 'CL', 'Br': 'BR', 'I': 'I', 'Zn2p': 'ZN', 'Co2p': 'CO', 'Cu2p': 'CU', 'Fe2p': 'FE', 'Fe3p': 'FE', 'Mg2p': 'MG', 'Ca2p': 'CA', 'Pha': 'P', 'OPha': 'O', 'OHha': 'O', 'Hha': 'H', 'CO3': 'C', 'OC3': 'O', 'Si': 'Si', 'OSi': 'O', 'Oice': 'O', 'Hice': 'H', 'Na1p': 'NA', 'K1p': 'K', 'He': 'HE', 'Li': 'LI', 'Be': 'BE', 'Ne': 'NE', 'Al': 'AL', 'Ar': 'AR', 'Sc': 'SC', 'Ti': 'TI', 'V': 'V', 'Cr': 'CR', 'Mn': 'MN', 'Ni': 'NI', 'Ga': 'GA', 'Ge': 'GE', 'As': 'AS', 'Se': 'SE', 'Kr': 'KR', 'Rb': 'RB', 'Sr': 'SR', 'Y': 'Y', 'Zr': 'ZR', 'Nb': 'NB', 'Mo': 'MO', 'Tc': 'TC', 'Ru': 'RU', 'Rh': 'RH', 'Pd': 'PD', 'Ag': 'AG', 'Cd': 'CD', 'In': 'IN', 'Sn': 'SN', 'Sb': 'SB', 'Te': 'TE', 'Xe': 'XE', 'Cs': 'CS', 'Ba': 'BA', 'La': 'LA', 'Ce': 'CE', 'Pr': 'PR', 'Nd': 'ND', 'Pm': 'PM', 'Sm': 'SM', 'Eu': 'EU', 'Gd': 'GD', 'Tb': 'TB', 'Dy': 'DY', 'Ho': 'HO', 'Er': 'ER', 'Tm': 'TM', 'Yb': 'YB', 'Lu': 'LU', 'Hf': 'HF', 'Ta': 'TA', 'W': 'W', 'Re': 'RE', 'Os': 'OS', 'Ir': 'IR', 'Pt': 'PT', 'Au': 'AU', 'Hg': 'HG', 'Tl': 'TL', 'Pb': 'PB', 'Bi': 'BI', 'Po': 'PO', 'At': 'AT', 'Rn': 'RN', 'Fr': 'FR', 'Ra': 'RA', 'Ac': 'AC', 'Th': 'TH', 'Pa': 'PA', 'U': 'U', 'Np': 'NP', 'Pu': 'PU', 'Am': 'AM', 'Cm': 'CM', 'Bk': 'BK', 'Cf': 'CF', 'Es': 'ES', 'Fm': 'FM', 'Md': 'MD', 'No': 'NO', 'Lr': 'LR', 'SUCK': 'Z', 'REPL': 'Z', 'REPLS': 'Z', 'HREPS': 'Z', 'VIRT': 'X', 'MPct': 'X', 'MPnm': 'X', 'MPdp': 'X', 'MPtk': 'X'}
        
        chem_elements = ['C','N','O','P','S','AC','AG','AL','AM','AR','AS','AT','AU','B','BA','BE','BI','BK','BR','CA','CD','CE','CF','CL','CM','CO','CR','CS','CU','DY','ER','ES','EU','F','FE','FM','FR','GA','GD','GE','H','HE','HF','HG','HO','I','IN','IR','K','KR','LA','LI','LR','LU','MD','MG','MN','MO','NA','NB','ND','NE','NI','NO','NP','OS','PA','PB','PD','PM','PO','PR','PT','PU','RA','RB','RE','RH','RN','RU','SB','SC','SE','SM','SN','SR','Si','TA','TB','TC','TE','TH','TI','TL','TM','U','V','W','X','XE','Y','YB','Z','ZN','ZR']
        
        
        ref_atypes_dict = dict(zip(chem_elements, range(len(chem_elements))))
    
    
    
     
        pdb_dict_list = []
        c = 0
        dna_list = 'atcg'
        rna_list = 'dryu'
        protein_list = 'ARNDCQEGHILKMFPSTWYVX'
        protein_list_check = 'ARNDCQEGHILKMFPSTWYV'
        k_DNA = 10
        ligand_dumm_list = 'J'
        
        ### This was just a big copy-paste from the training script ###
        dna_rna_dict = {
        "a" : ['P', 'OP1', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", 'N1', 'C2', 'N3', 'C4', 'C5', 'C6', 'N7', 'C8','N9', "", ""],
        "t" : ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N1", "C2", "O2", "N3", "C4", "C5", "C6", "C7", "", "", ""],
        "c" : ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6", "", "", ""],
        "g" : ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N1", "C2", "N2", "N3", "C4", "C5", "C6", "N7", "C8", "N9", ""],
        "d" : ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "N3", "C4", "C5", "C6", "N7", "C8", "N9", ""],
        "r" : ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6", "", ""],
        "y" : ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6", "", ""],
        "u" : ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "N2", "N3", "C4", "C5", "C6", "N7", "C8", "N9"],
        "X" : 22*[""]}
        
        dna_rna_atom_types = np.array(["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "N2", "N3", "C4", "C5", "C6", "N7", "C8", "N9", "O4", "O2", "N4", "C7", ""])
    
        idxAA_22_to_27 = np.zeros((9, 22), np.int32)
        for i, AA in enumerate(dna_rna_dict.keys()):
            for j, atom in enumerate(dna_rna_dict[AA]):
                idxAA_22_to_27[i,j] = int(np.argwhere(atom==dna_rna_atom_types)[0][0])
        ### \end This was just a big copy-paste from the training script ###
        
        atoms = ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
            'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
            'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
            'CZ3', 'NZ']  # These are the 36 atom types mentioned in Justas's script
        
        all_atom_types = atoms + list(dna_rna_atom_types)
        init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']
        extra_alphabet = [str(item) for item in list(np.arange(300))]
        chain_alphabet = init_alphabet# + extra_alphabet
    
        biounit_names = [path_to_pdb]
        for biounit in biounit_names:
            my_dict = {}
            s = 0
            concat_seq = ''
            concat_seq_DNA = ''
            concat_N = []
            concat_CA = []
            concat_C = []
            concat_O = []
            concat_mask = []
            coords_dict = {}        
            visible_list = []
            chain_list = []
            Cb_list = []
            P_list = []
            dna_atom_list = []
            dna_atom_mask_list = []
            #
            ligand_atom_list = []
            ligand_atype_list = []
            ligand_total_length_type = 0
            ligand_total_length_coords = 0

            #
            #Check if ligand params file is given
            lig_params = []
            if biounit in list(extra_res_param.keys()):
                lig_params = extra_res_param[biounit]
            
            for letter in chain_alphabet:
    #             print(f'started parsing {letter}')
                xyz, seq, atype = parse_PDB_biounits(biounit, atoms = all_atom_types, chain=letter, lig_params=lig_params)
                if type(xyz) != str:
                    protein_seq_flag = any([(item in seq[0]) for item in protein_list_check])
                    dna_seq_flag = any([(item in seq[0]) for item in dna_list])
                    rna_seq_flag = any([(item in seq[0]) for item in rna_list])
                    lig_seq_flag = any([(item in seq[0]) for item in ligand_dumm_list])
                    
                    if protein_seq_flag: xyz, seq, atype = parse_PDB_biounits(biounit, atoms = atoms, chain=letter)
                    elif (dna_seq_flag or rna_seq_flag): xyz, seq, atype = parse_PDB_biounits(biounit, atoms = list(dna_rna_atom_types), chain=letter)
                    elif (lig_seq_flag): xyz,seq, atype = parse_PDB_biounits(biounit, atoms=[], chain=letter, lig_params=lig_params)
                    
                    if protein_seq_flag:
                        my_dict['seq_chain_'+letter]=seq[0]
                        concat_seq += seq[0]
                        chain_list.append(letter)
                        all_atoms = np.array(xyz) #[L, 14, 3] # deleted res index on xyz--I think this was useful when there were batches of structures at once?
                        b = all_atoms[:,1] - all_atoms[:,0]
                        c = all_atoms[:,2] - all_atoms[:,1]
                        a = np.cross(b, c, -1)
                        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + all_atoms[:,1] #virtual
                        Cb_list.append(Cb)
                        coords_dict_chain = {}
                        coords_dict_chain['all_atoms_chain_'+letter]=xyz.tolist()
                        my_dict['coords_chain_'+letter]=coords_dict_chain
                        s += 1
                    elif dna_seq_flag or rna_seq_flag: # This section is important for moving from 22-atom representation to the 27-atom representation...unless it's already in 27 format??
                        all_atoms = np.array(xyz)
                        P_list.append(all_atoms[:,0])
                        all_atoms_ones = np.ones((all_atoms.shape[0], 22)) # I believe all_atoms.shape[0] is supposed to be the length of the sequence
                        seq_ = "".join(list(np.array(list(seq))[0,]))
                        concat_seq_DNA += seq_
                        all_atoms27_mask = np.zeros((len(seq_), 27))
                        idx = np.array([idxAA_22_to_27[np.argwhere(AA==np.array(list(dna_rna_dict.keys())))[0][0]] for AA in seq_])
                        np.put_along_axis(all_atoms27_mask, idx, all_atoms_ones, 1) 
                        dna_atom_list.append(all_atoms) # was all_atoms27, but all_atoms is already in that format!!
                        dna_atom_mask_list.append(all_atoms27_mask)
                    elif lig_seq_flag:
                        temp_atype = -np.ones(len(atype))
                        for k_, ros_type in enumerate(atype):
                            if ros_type in list(ref_atype_to_element):
                                temp_atype[k_] = ref_atypes_dict[ref_atype_to_element[ros_type]]
                            else:
                                temp_atype[k_] = ref_atypes_dict['X']
                        all_atoms = np.array(xyz)
                        all_atoms = np.reshape(all_atoms, [-1,3])
                        ligand_atype = np.array(temp_atype)
                        if (1-np.isnan(all_atoms)).sum() != 0:
                            ligand_atom_list.append(all_atoms)
                            ligand_atype_list.append(ligand_atype)
                            ligand_total_length_coords += all_atoms.shape[0]
                            ligand_total_length_type += ligand_atype.shape[0]

                        
            if len(P_list) > 0:
                Cb_stack = np.concatenate(Cb_list, 0) #[L, 3]
                P_stack = np.concatenate(P_list, 0) #[K, 3]
                dna_atom_stack = np.concatenate(dna_atom_list, 0)
                dna_atom_mask_stack = np.concatenate(dna_atom_mask_list, 0)
                
                D = np.sqrt(((Cb_stack[:,None,:]-P_stack[None,:,:])**2).sum(-1) + 1e-7)
                idx_dna = np.argsort(D,-1)[:,:k_DNA] #top 10 neighbors per residue
                dna_atom_selected = dna_atom_stack[idx_dna]
                dna_atom_mask_selected = dna_atom_mask_stack[idx_dna]
                my_dict['dna_context'] = dna_atom_selected[:,:,:-1,:].tolist()
                my_dict['dna_context_mask'] = dna_atom_mask_selected[:,:,:-1].tolist()
            else:
                my_dict['dna_context'] = 'no_DNA'
                my_dict['dna_context_mask'] = 'no_DNA'
            if ligand_atom_list:
                ligand_atom_stack = np.concatenate(ligand_atom_list, 0)
                ligand_atype_stack = np.concatenate(ligand_atype_list, 0)
                my_dict['ligand_context'] = ligand_atom_stack.tolist()
                my_dict['ligand_atype'] = ligand_atype_stack.tolist()
            else:
                my_dict['ligand_context'] = 'no_ligand'
                my_dict['ligand_atype'] = 'no_ligand'
            if ligand_total_length_coords != ligand_total_length_type:
                print('WARNING: LIGAND PARSING MISMATCH!')
            my_dict['ligand_length'] = int(ligand_total_length_coords)
            #
            fi = biounit.rfind("/")
            my_dict['name']=biounit[(fi+1):-4]
            my_dict['num_of_chains'] = s
            my_dict['seq'] = concat_seq
            if s < len(chain_alphabet):
                pdb_dict_list.append(my_dict)
                c+=1
            return pdb_dict_list


    pdb_dict_list = []
    if folder_with_pdbs_path[-1]!='/':
        folder_with_pdbs_path = folder_with_pdbs_path+'/'
    
    biounit_names = glob.glob(folder_with_pdbs_path+'*.pdb')
    params_names = [name[:-4]+'.params' for name in biounit_names]
    for k, biounit in enumerate(biounit_names):
        if os.path.isfile(params_names[k]):
            out = parse_PDB(biounit, {biounit: [params_names[k]]})
        else:
            out = parse_PDB(biounit)
        pdb_dict_list += out
    
    with open(save_path, 'w') as f:
        for entry in pdb_dict_list:
            f.write(json.dumps(entry) + '\n')
           

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--input_path", type=str, help="Path to a folder with pdb files, e.g. /home/my_pdbs/")
    argparser.add_argument("--output_path", type=str, help="Path where to save .jsonl dictionary of parsed pdbs")

    args = argparser.parse_args()
    main(args)
