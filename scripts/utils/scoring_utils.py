"""
Extra modules for scoring protein structures
Authors: Chris Norn, Indrek Kalvet
"""
import os
import pyrosetta
import pyrosetta.rosetta
import numpy as np
from pyrosetta.rosetta.core.scoring import fa_rep

import Bio.PDB
BIO_PDB_parser = Bio.PDB.PDBParser(QUIET=True)


def get_one_and_twobody_energies(p, scorefxn):
    nres = p.size()    
    res_energy_z = np.zeros(nres)
    res_pair_energy_z = np.zeros( (nres,nres) )
    res_energy_no_two_body_z = np.zeros ( (nres) )

    totE = scorefxn(p)
    energy_graph = p.energies().energy_graph()

    twobody_terms = p.energies().energy_graph().active_2b_score_types()
    onebody_weights = pyrosetta.rosetta.core.scoring.EMapVector()
    onebody_weights.assign(scorefxn.weights())

    for term in twobody_terms:
        if 'intra' not in pyrosetta.rosetta.core.scoring.name_from_score_type(term):
            onebody_weights.set(term, 0)

    for i in range(1,p.size()+1):
        res_energy_no_two_body_z[i-1] = p.energies().residue_total_energies(i).dot(onebody_weights)
        res_energy_z[i-1]= p.energies().residue_total_energy(i)
    
        for j in range(1,p.size()+1):
            if i == j: continue
            edge = energy_graph.find_edge(i,j)
            if edge is None:
                energy = 0.0
            else:
                res_pair_energy_z[i-1][j-1]= edge.fill_energy_map().dot(scorefxn.weights())

    one_body_tot = np.sum(res_energy_z)
    one_body_no_two_body_tot = np.sum(res_energy_no_two_body_z)
    two_body_tot = np.sum(res_pair_energy_z)

    onebody_energies = res_energy_no_two_body_z
    twobody_energies = res_pair_energy_z  # This matrix is symmetrical, 0 diagonal, and when summed only the half matrix sohuld be summed

    return onebody_energies, twobody_energies


def calculate_ddg(pose, scorefxn, ser_idx=None):
    # anna added this
    size = pose.size()

    twobody_energies = get_one_and_twobody_energies(pose, scorefxn)[1]
    scorefxn.set_weight(fa_rep, 0.0)
    twobody_energies_no_fa_rep = get_one_and_twobody_energies(pose, scorefxn)[1]
    
    if ser_idx is None:
        ddg = np.sum(twobody_energies[size-1])
    else:
        ddg = np.sum(twobody_energies[size-1]) \
            - twobody_energies[size-1][ser_idx-1] \
            + twobody_energies_no_fa_rep[size-1][ser_idx-1] 
    return ddg


def mutate_pdb(pdb, site, mutant_aa, output_file):
    pose = pyrosetta.pose_from_pdb(pdb)
    pyrosetta.toolbox.mutants.mutate_residue(pose, site, mutant_aa)
    pyrosetta.dump_pdb(pose, output_file)


def apply_score_from_filter(pose, filter_obj):
    pyrosetta.rosetta.core.pose.setPoseExtraScore(pose,
                                                  filter_obj.get_user_defined_name(),
                                                  filter_obj.score(pose))


def dump_scorefile(df, filename):
    widths = {}
    for k in df.keys():
        if k in ["SCORE:", "description", "name"]:
            widths[k] = 0
        elif isinstance(df.at[df.index.values[0], k], str):
            max_val_len = max([len(row[k]) for index, row in df.iterrows()])
            widths[k] = max(max_val_len, len(k)) + 1
        elif len(k) >= 12:
            widths[k] = len(k) + 1
        else:
            widths[k] = 12

    keys = df.keys()
    write_title = True
    if os.path.exists(filename):
        write_title = False
        keys = open(filename, 'r').readlines()[0].split()
        keys = [x.rstrip() for x in keys]
        if len(keys) != len(df.keys()):
            print(f"Number of columns in existing scorefile {filename} and "
                  f"scores dataframe does not match: {len(keys)} != {len(df.keys())}")

    with open(filename, "a") as file:
        title = ""
        if write_title is True:
            for k in df.keys():
                if k == "SCORE:":
                    title += k
                elif k in ["description", "name"]:
                    continue
                else:
                    title += f"{k:>{widths[k]}}"
            if 'description' in df.keys():
                title += " description"
            file.write(title + "\n")

        for index, row in df.iterrows():
            line = ""
            for k in keys:
                if k not in df.keys():
                    val = f"{np.nan}"
                    widths[k] = 11
                elif isinstance(row[k], (float, np.float16, np.float64)):
                    val = f"{row[k]:.3f}"
                else:
                    val = row[k]
                if k == "SCORE:":
                    line += val
                elif k in ["description", "name"]:
                    continue
                else:
                    line += f"{val:>{widths[k]}}"
            # Always writing PDB name as the last item
            if 'description' in df.keys():
                line += f" {row['description']}"
            file.write(line + "\n")


def getSASA(pose, resno=None, SASA_atoms=None, ignore_sc=False):
    """
    Takes in a pose and calculates its SASA.
    Or calculates SASA of a given residue.
    Or calculates SASA of specified atoms in a given residue.

    Procedure by Brian Coventry
    """

    atoms = pyrosetta.rosetta.core.id.AtomID_Map_bool_t()
    atoms.resize(pose.size())

    n_ligands = 0
    for res in pose.residues:
        if res.is_ligand():
            n_ligands += 1

    for i, res in enumerate(pose.residues):
        if res.is_ligand():
            atoms.resize(i+1, res.natoms(), True)
        else:
            atoms.resize(i+1, res.natoms(), not(ignore_sc))
            if ignore_sc is True:
                for n in range(1, res.natoms()+1):
                    if res.atom_is_backbone(n) and not res.atom_is_hydrogen(n):
                        atoms[i+1][n] = True

    surf_vol = pyrosetta.rosetta.core.scoring.packing.get_surf_vol(pose, atoms, 1.4)

    if resno is not None:
        res_surf = 0.0
        for i in range(1, pose.residue(resno).natoms()+1):
            if SASA_atoms is not None and i not in SASA_atoms:
                continue
            res_surf += surf_vol.surf(resno, i)
        return res_surf
    else:
        return surf_vol


def find_hbonds_to_residue_atom(pose, lig_seqpos, target_atom):
    """
    Counts how many Hbond contacts input atom has with the protein.
    """
    HBond_res = 0

    for res in pose.residues:
        if res.seqpos() == lig_seqpos or res.is_ligand():
            break
        if (pose.residue(lig_seqpos).xyz(target_atom) - res.xyz('CA')).norm() < 10.0:
            for polar_H in res.Hpos_polar():
                if (pose.residue(lig_seqpos).xyz(target_atom) - res.xyz(polar_H)).norm() < 2.5:
                    # If the polar atom is from the backbone then check that the X-H...Y angle is close to linear.
                    # It is assumed that polar backbone H is only attached to backbone N
                    if res.atom_is_backbone(polar_H):
                        # print(res.seqpos(), target_atom, res.atom_name(polar_H), get_angle(res.xyz(1), res.xyz(polar_H), pose.residue(lig_seqpos).xyz(target_atom)))
                        if get_angle(res.xyz(1), res.xyz(polar_H), pose.residue(lig_seqpos).xyz(target_atom)) < 140.0:
                            continue
                    HBond_res += 1
                    break
    return HBond_res


def find_res_with_hbond_to_residue_atom(pose, lig_seqpos, target_atom):
    """
    Counts how many Hbond contacts input atom has with the protein.
    """
    HBond_res = 0
    residues = []
    for res in pose.residues:
        if res.seqpos() == lig_seqpos or res.is_ligand():
            continue
        if (pose.residue(lig_seqpos).xyz(target_atom) - res.xyz('CA')).norm() < 10.0:
            for polar_H in res.Hpos_polar():
                if (pose.residue(lig_seqpos).xyz(target_atom) - res.xyz(polar_H)).norm() < 2.5:
                    # If the polar atom is from the backbone then check that the X-H...Y angle is close to linear.
                    # It is assumed that polar backbone H is only attached to backbone N
                    if res.atom_is_backbone(polar_H):
                        print(get_angle(res.xyz(1), res.xyz(polar_H), pose.residue(lig_seqpos).xyz(target_atom)))
                        if get_angle(res.xyz(1), res.xyz(polar_H), pose.residue(lig_seqpos).xyz(target_atom)) < 140.0:
                            continue
                    residues.append(res.seqpos())
                    HBond_res += 1
                    break
    return residues


def get_angle(a1, a2, a3):
    a1 = np.array(a1)
    a2 = np.array(a2)
    a3 = np.array(a3)

    ba = a1 - a2
    bc = a3 - a2

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return round(np.degrees(angle), 1)


def get_dihedral(a1, a2, a3, a4):
    """
    a1, a2, a3, a4 (np.array)
    Each array has to contain 3 floats corresponding to X, Y and Z of an atom.
    Solution by 'Praxeolitic' from Stackoverflow:
    https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python#
    1 sqrt, 1 cross product
    Calculates the dihedral/torsion between atoms a1, a2, a3 and a4
    Output is in degrees
    """

    b0 = a1 - a2
    b1 = a3 - a2
    b2 = a4 - a3

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= np.linalg.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))


def calculate_CA_rmsd(input_pose, design_pose, residue_list=None):
    reslist = pyrosetta.rosetta.std.list_unsigned_long_t()
    if residue_list is None:
        for n in range(1, input_pose.size()+1):
            if input_pose.residue(n).is_ligand():
                continue
            if input_pose.residue(n).is_virtual_residue():
                continue
            reslist.append(n)
    else:
        for n in residue_list:
            reslist.append(n)

    rmsd_CA = pyrosetta.rosetta.core.scoring.CA_rmsd(input_pose, design_pose, residue_selection=reslist)
    return rmsd_CA


def find_mutations(parent_pose, designed_pose):
    mutations = {}
    for res_p, res_d in zip(parent_pose.residues, designed_pose.residues):
        if res_p.name3() != res_d.name3():
            mutations[res_p.seqpos()] = {'from': res_p.name3(),
                                         'to': res_d.name3()}
    return mutations


def get_residue_subset_lddt(lddt, residues):
    """
    Calculates the average lDDT of a subset of residues
    """
    lddt_desres = [lddt[x-1] for x in residues]
    return np.average(lddt_desres)


def _fix_CYX_pdbfile(pdbfile):
    pdbf = open(pdbfile, "r").readlines()
    
    new_pdbf = []
    for l in pdbf:
        if "CYX" in l:
            new_pdbf.append(l.replace("CYX", "CYS"))
        else:
            new_pdbf.append(l)
    temp_pdb = pdbfile.replace(".pdb", "_tmp.pdb")
    with open(temp_pdb, "w") as file:
        for l in new_pdbf:
            file.write(l)
    return temp_pdb

