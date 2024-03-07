import argparse
import numpy as np
import os, json, sys
import glob
import random
import pyrosetta as pyr
import pyrosetta.rosetta
from pyrosetta.rosetta.utility import vector1_numeric_xyzVector_double_t as xyzVector
from pyrosetta.rosetta.numeric import xyzMatrix_double_t
from pyrosetta.rosetta.numeric import xyzVector_double_t
from pyrosetta.rosetta.protocols.toolbox import superposition_transform
from pyrosetta.rosetta.protocols.toolbox import apply_superposition_transform
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.core.pose import append_subpose_to_pose

# HOME = ""
# if not os.path.exists("/home/ikalvet"):
#     HOME = "/home/indrek/UW_Digs"
# sys.path.append(f"{HOME}/home/ikalvet/git/proteinmpnn/ligand_proteinmpnn/pyrosetta_tools")

from parser_tools import parse_pose


def match_sidechain_atoms(target, target_res_to_match,
                          target_atom_list_to_match, loop, loop_res_to_match,
                          loop_res_atom_list_to_match):
    """
    Given a pose of the target+a residue, and a pose of a loop, 
    and which residue to match

    Moves the loop to the stub and returns the overlayed pose

    returns a pose that has the loop in the last chain,
    and the target in other chains...
    Arguments:
        target (obj, pose) :: pose object -> metal thing
        target_res_to_match (int) :: rosetta residue index -> which cysetine you want to match
        target_atom_list_to_match (list) :: python array of atoms names ['SG', 'CB', 'CA']
        loop (obj, pose) :: pose of your macrocycle
        loop_res_to_match (int) :: Cys residue # in macrocycles
        loop_res_atom_list_to_match (list) :: python array of atoms names ['SG', 'CB', 'CA']

    Written by Patrick Salveson and Meerit Said
    """

    # store the lengths of the two poses
    target_length = len(target.residues)
    loop_length = len(loop.residues)

    # make the rosetta object to hold xyz crds of the residue we want to align
    target_match_coords = xyzVector()
    loop_coords = xyzVector()

    # add the coords of the residues to be aligned to their respctive holders
    for target_name, loop_name in zip(target_atom_list_to_match, loop_res_atom_list_to_match):
        target_match_coords.append(
            target.residues[target_res_to_match].xyz(target_name))
        loop_coords.append(loop.residues[loop_res_to_match].xyz(loop_name))

    # make the rotation matrix and pose center rosetta objects
    rotation_matrix = xyzMatrix_double_t()
    target_center = xyzVector_double_t()
    loop_center = xyzVector_double_t()

    superposition_transform(
        loop_coords,
        target_match_coords,
        rotation_matrix,
        loop_center,
        target_center)

    apply_superposition_transform(
        loop,
        rotation_matrix,
        loop_center,
        target_center)

    # at this point "loop" is super imposed on the res in target
    # loop.dump_pdb('name.pdb')
    #########################################
    new_loop = Pose()
    new_loop.assign(loop)

    # at this point, the two objects are aligned
    # create a new empy pose object
    # splice the poses together and return
    if loop.residue(loop_res_to_match).is_ligand():
        spliced_pose = target.clone()
        drm = pyrosetta.rosetta.protocols.grafting.simple_movers.DeleteRegionMover()
        drm.region(str(target_res_to_match), str(target_res_to_match))
        drm.apply(spliced_pose)
        new_chain = True
    else:
        spliced_pose = Pose()
        append_subpose_to_pose(spliced_pose, target, 1, target.size() - 1)
        new_chain = False
    append_subpose_to_pose(spliced_pose, loop, 1, loop_length, new_chain)
    ###############################################################

    return spliced_pose


def replace_ligand_in_pose(pose, residue, resno, ref_atoms, new_atoms):
    _tmp_ligpose = pyrosetta.rosetta.core.pose.Pose()
    _tmp_ligpose.append_residue_by_jump(residue, 0)
    new_pose = match_sidechain_atoms(pose, resno, ref_atoms,
                                     _tmp_ligpose, 1, new_atoms)
    return new_pose


def get_rotamers_for_res_in_pose(in_pose, target_residue, sfx, ex1=True,
                                 ex2=True, ex3=False, ex4=False,
                                 check_clashes=True):
    packTask = pyrosetta.rosetta.core.pack.task.TaskFactory.create_packer_task(in_pose)
    packTask.set_bump_check(check_clashes)
    resTask = packTask.nonconst_residue_task(target_residue)
    resTask.or_ex1(ex1)
    resTask.or_ex2(ex2)
    resTask.or_ex3(ex3)
    resTask.or_ex4(ex4)
    resTask.restrict_to_repacking()
    packer_graph = pyrosetta.rosetta.utility.graph.Graph(in_pose.size())
    rsf = pyrosetta.rosetta.core.pack.rotamer_set.RotamerSetFactory()
    rotset = rsf.create_rotamer_set(in_pose)
    rotset.set_resid(target_residue)
    rotset.build_rotamers(in_pose, sfx, packTask, packer_graph, False)
    print("Found num_rotamers:", rotset.num_rotamers())
    rotamer_set = []
    for irot in range(1, rotset.num_rotamers() + 1):
        rotamer_set.append(rotset.rotamer(irot).clone())
    return rotamer_set


def get_ligand_clash(pose, ligand_atoms, cutoff=2.5):
    ligand = pose.residue(pose.size())
    dists = []
    for res in pose.residues:
        if res.is_ligand():
            continue
        if (res.nbr_atom_xyz() - ligand.nbr_atom_xyz()).norm() > 15.0:
            continue
        for atm in ["CA", "C", "N", "O", "CB", "H", "HA"]:
            if not res.has(atm):
                continue
            for ha in ligand_atoms:
                dists.append((res.xyz(atm) - ligand.xyz(ha)).norm())

    return min(dists) < cutoff


def get_residue_heavyatoms(residue):
    heavyatoms = []
    for atomno in range(1, residue.natoms()):
        if residue.atom_type(atomno).is_heavyatom():
            heavyatoms.append(residue.atom_name(atomno).strip())
    return heavyatoms


def main(args):
    if args.pdb is not None:
        _pdbfiles = args.pdb

    if args.pdblist is not None:
        assert args.input_path is None, "Can't give pdblist and input_path together"
        _pdbfiles = open(args.pdblist, "r").readlines()
        _pdbfiles = [f.rstrip() for f in _pdbfiles if ".trb" in f]

    if args.input_path is not None:
        assert args.pdblist is None, "Can't give pdblist and input_path together"
        assert os.path.exists(args.input_path)
        _pdbfiles = glob.glob(f"{args.input_path}/*.pdb")

    setattr(args, "pdblist", _pdbfiles)

    extra_res_fa = ""
    if args.params is not None:    
        extra_res_fa = '-extra_res_fa'
        for p in args.params:
            if os.path.exists(p):
                extra_res_fa += " {}".format(p)

    overlay_atoms = args.overlay_atoms

    pyr.init(f"{extra_res_fa} -beta -run:preserve_header -mute all")
    scorefxn = pyr.get_fa_scorefxn()

    pdb_dict_list = []
    for pdbfile in args.pdblist:
        print(f"Working on {pdbfile}")
        pose = pyr.pose_from_file(pdbfile)
        
        # Collecting the rotamers for the ligand in the pose
        rotset_ligand = get_rotamers_for_res_in_pose(pose, target_residue=pose.size(), sfx=scorefxn)

        rotset_ligand = [(i, rotamer) for i, rotamer in enumerate(rotset_ligand)]  # Storing the original indeces for randomization
        random.shuffle(rotset_ligand)  # Randomizing the rotamers

        # Getting the names of heavyatoms in the ligand
        ligand_atoms = get_residue_heavyatoms(pose.residue(pose.size()))

        # Iterating over the rotamers and checking if any heavyatoms clash with
        # protein backbone atoms.
        # pdb_dict_list = []
        for (i, _rotamer) in rotset_ligand:
            pose3 = pose.clone()
            pose3 = replace_ligand_in_pose(pose3, _rotamer, pose3.size(), overlay_atoms, overlay_atoms)
            clash = get_ligand_clash(pose3, ligand_atoms, cutoff=1.8)  # Is 2.5 a good cutoff?
            if clash is False:
                parsed_pose_dict = parse_pose(pose3, os.path.basename(pdbfile).replace(".pdb", f"_r{i}"), args.params)

                # Optional: include some RMSD check?

                pdb_dict_list += parsed_pose_dict
                pose3.dump_pdb(os.path.basename(pdbfile).replace(".pdb", f"_r{i}.pdb"))
                if args.n_rotamers is not None:
                    if len(pdb_dict_list) == args.n_rotamers:
                        break

    with open(args.output_path, 'w') as f:
        for entry in pdb_dict_list:
            f.write(json.dumps(entry) + '\n')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--input_path", type=str, help="Path to a folder with pdb files, e.g. /home/my_pdbs/")
    argparser.add_argument("--pdb", nargs="+", dest="pdb", help="Input PDBs. If argument not given, all PDBs in working directory are taken.")
    argparser.add_argument("--pdblist", type=str, dest="pdblist", help="File containing list of PDB filenames")
    argparser.add_argument("--output_path", type=str, required=True, help="Path where to save .jsonl dictionary of parsed pdbs")
    argparser.add_argument("--params", nargs="+", help="Params files.")
    argparser.add_argument("--overlay_atoms", nargs="+", help="Ligand atom names that will be fixed in space and used to overlay different ligand rotamers.")
    argparser.add_argument("--n_rotamers", type=int, help="Maximum number of rotamers to be included per input.")
    # Optional: include some RMSD cutoff for only keeping rotamers that are above X RMSD different?

    args = argparser.parse_args()
    main(args)

