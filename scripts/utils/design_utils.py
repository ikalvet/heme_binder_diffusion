"""
Extra modules for design protocols
Authors: Indrek Kalvet
"""
import pyrosetta as pyr
import pyrosetta.rosetta
import re
import numpy as np
from pyrosetta.rosetta.core.select.residue_selector import ResidueIndexSelector
from pyrosetta.rosetta.utility import vector1_numeric_xyzVector_double_t as xyzVector
from pyrosetta.rosetta.numeric import xyzMatrix_double_t
from pyrosetta.rosetta.numeric import xyzVector_double_t
from pyrosetta.rosetta.protocols.toolbox import superposition_transform
from pyrosetta.rosetta.protocols.toolbox import apply_superposition_transform
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.core.pose import append_subpose_to_pose
import sys, os


alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
alpha_3 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
           'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','GAP']

aa_1_3 = {a:b for a,b in zip(alpha_1,alpha_3)}
aa_3_1 = {b:a for a,b in zip(alpha_1,alpha_3)}


class CSTs():
    def __init__(self, cstfile, scorefxn):
        self.__scorefxn = scorefxn
        self.__addcst_mover = pyrosetta.rosetta.protocols.enzdes.AddOrRemoveMatchCsts()
        self.__chem_manager = pyrosetta.rosetta.core.chemical.ChemicalManager.get_instance()
        self.__residue_type_set = self.__chem_manager.residue_type_set("fa_standard")
        self.__cst_io = pyrosetta.rosetta.protocols.toolbox.match_enzdes_util.EnzConstraintIO(self.__residue_type_set)
        self.__cst_io.read_enzyme_cstfile(cstfile)
        pass
    
    def add_cst(self, pose):
        self.__cst_io.add_constraints_to_pose(pose, self.__scorefxn, True)
    
    def remove_cst(self, pose):
        self.__cst_io.remove_constraints_from_pose(pose, True, True)
        
    def cst_io(self):
        return self.__cst_io


def aa321(aa3):
    aa123 = dict(A="ALA", C="CYS", D="ASP", E="GLU", F="PHE", G="GLY", H="HIS", I="ILE", K="LYS",
             L="LEU", M="MET", N="ASN", P="PRO", Q="GLN", R="ARG", S="SER", T="THR", V="VAL",
             W="TRP", Y="TYR")
    aa321 = {aa123[aa1]:aa1 for aa1 in aa123}
    return aa321[aa3]

def aa123(aa1):
    aa123 = dict(A="ALA", C="CYS", D="ASP", E="GLU", F="PHE", G="GLY", H="HIS", I="ILE", K="LYS",
             L="LEU", M="MET", N="ASN", P="PRO", Q="GLN", R="ARG", S="SER", T="THR", V="VAL",
             W="TRP", Y="TYR")
    return aa123[aa1]


def get_matcher_residues(filename):
    """
    Creates a dictionary that contains information about the matched residues
    that are stored in the REMARKS 666 lines in the PDBfile header.
    Arguments:
        filename (str) :: path to a PDB file
          or
        filename (pyrosetta.rosetta.core.pose.Pose) :: pyrosetta pose
    """
    if isinstance(filename, str):
        pdbfile = open(filename, 'r').readlines()
    elif isinstance(filename, pyrosetta.rosetta.core.pose.Pose):
        pdbfile = pyrosetta.distributed.io.to_pdbstring(filename).split("\n")

    matches = {}
    for line in pdbfile:
        if "ATOM" in line:
            break
        if "REMARK 666" in line:
            lspl = line.split()
            chain = lspl[9]
            res3 = lspl[10]
            resno = int(lspl[11])

            matches[resno] = {'name3': res3,
                              'chain': chain}
    return matches


def get_single_pose_from_cloudpdb(pose):
    """
    Removes extra instances of matched residues and ligands from the pose
    """
    pose2 = pose.clone()
    for n in range(pose.num_chains(), 0, -1):
        if len(pose.chain_sequence(n)) == 1:
            is_current_res_ligand = pose.residue(pose.chain_begin(n)).is_ligand()
            is_prev_res_ligand = pose.residue(pose.chain_end(n-1)).is_ligand()
            if is_current_res_ligand is True and is_prev_res_ligand is False:
                # First ligand instance is kept
                continue
            drm = pyrosetta.rosetta.protocols.grafting.simple_movers.DeleteRegionMover()
            drm.region(str(pose.chain_begin(n)), str(pose.chain_end(n)))
            drm.apply(pose2)
    return pose2


def add_match_cst(input_pose, cst_file, scorefxn):
    if not os.path.exists(cst_file):
        print(f"CSTfile {cst_file} not found")
        pass
    # scorefxn = pyrosetta.get_score_function()
    addcst_mover = pyrosetta.rosetta.protocols.enzdes.AddOrRemoveMatchCsts()
    chem_manager = pyrosetta.rosetta.core.chemical.ChemicalManager.get_instance()
    residue_type_set = chem_manager.residue_type_set("fa_standard")
    cst_io = pyrosetta.rosetta.protocols.toolbox.match_enzdes_util.EnzConstraintIO(residue_type_set)
    cst_io.read_enzyme_cstfile(cst_file)
    cst_io.add_constraints_to_pose(input_pose, scorefxn, True)


def get_ligand_heavyatoms(pose, resno=None):
    """
    Returns a list of non-hydrogen atomnames
    """
    if resno is None:
        resno = pose.size()
    if pose.residue(resno).is_ligand() is False:
        print("Last residue is not ligand!")
        return None
    heavyatoms = []
    for n in range(1, pose.residue(resno).natoms() + 1):
        element = pose.residue(resno).atom_type(n).element()
        if element != 'H':
            heavyatoms.append(pose.residue(resno).atom_name(n).lstrip().rstrip())
    return heavyatoms


def get_packer_layers(pose, ref_resno, cuts, target_atoms, do_not_design=None, allow_design=None, design_GP=False):
    """
    Finds residues that are within certain distances from target atoms, defined though <cuts>.
    Returns a list of lists where each embedded list contains residue numbers belongin to that layer.
    Last list contains all other residue numbers that were left over.
    get_packer_layers(pose, target_atoms, do_not_design, cuts) -> list
    Arguments:
        pose (object, pyrosetta.rosetta.core.pose.Pose)
        target_atoms (list) :: list of integers.
                               Atom numbers of a residue or a ligand that are used to calculate residue distances.
        do_not_design (list) :: list of integers. Residue numbers that are not allowed to be designed.
                                Used to override layering to not design matched residues.
        cuts (list) :: list of floats.
        design_GP (bool) :: if True, allows redesign of GLY and PRO
    """
    assert len(cuts) > 3, f"Not enough layer cut distances defined {cuts}"
    # print("Determining design/repack/do-not-touch layers "
    #       "based on cuts: {}".format(cuts))

    if do_not_design is None:
        do_not_design = []
    if allow_design is None:
        allow_design = []

    KEEP_RES = ["GLY", "PRO"]
    if design_GP is True:
        KEEP_RES = []

    residues = []
    ligand_atoms = []
    if isinstance(target_atoms, list):
        for a in target_atoms:
            ligand_atoms.append(pose.residue(ref_resno).xyz(a))
    if isinstance(target_atoms, dict):
        for k in target_atoms:
            for a in target_atoms[k]:
                ligand_atoms.append(pose.residue(k).xyz(a))

    residues = [[] for i in cuts] + [[]]

    for resno in range(1, pose.size()):
        if pose.residue(resno).is_ligand():
            continue
        if pose.residue(resno).is_virtual_residue():
            continue
        if resno in do_not_design:
            # do_not_design residues are added to repack layer
            residues[2].append(resno)
            continue
        if resno in allow_design:
            # allow_design residues are added to design layer
            residues[0].append(resno)
            continue
        resname = pose.residue(resno).name3()
        CA = pose.residue(resno).xyz('CA')
        CA_distances = []

        if 'GLY' not in resname:
            CB = pose.residue(resno).xyz('CB')
            CB_distances = []

        for a in ligand_atoms:
            CA_distances.append((a - CA).norm())
            if 'GLY' not in resname:
                CB_distances.append((a - CB).norm())
        CA_mindist = min(CA_distances)

        if 'GLY' not in resname:
            # CB_mindist = CB_distances[CA_distances.index(CA_mindist)]
            CB_mindist = min(CB_distances)

        # Figuring out into which cut that residue belongs to,
        # based on the smallest CA distance and whether the CA is further away from the ligand than CB or not.
        # PRO and GLY are disallowed form the first two cuts that would allow design,
        # they can only be repacked.
        if CA_mindist <= cuts[0] and resname not in KEEP_RES:
            residues[0].append(resno)
        elif CA_mindist <= cuts[1] and resname not in KEEP_RES:
            if CB_mindist < CA_mindist:
                residues[1].append(resno)
            elif CA_mindist < cuts[1]-1.0 and CB_mindist < cuts[1]-1.0:
                # Special case when the residue is close, but CB is pointing slightly away.
                residues[1].append(resno)
            else:
                residues[2].append(resno)
        elif CA_mindist <= cuts[2]:
            residues[2].append(resno)
        elif CA_mindist <= cuts[3] and resname not in KEEP_RES:
            if resname == "GLY":
                residues[3].append(resno)
            elif CB_mindist < CA_mindist:
                residues[3].append(resno)
            else:
                residues[-1].append(resno)
        else:
            residues[-1].append(resno)

    return residues


def get_layer_selections(pose, repack_only_pos, design_pos, ref_resno, heavyatoms, cuts=[6.0, 8.0, 10.0, 12.0], design_GP=False):
    """
    Gets residue selectors for design, repack and do-not-touch based on the cut values.
    Also returns a list with the corresponding position numbers.
    Will not allow redesign of PRO and GLY.
    repack_only_pos and design_pos will override the cuts
    Arguments:
        pose (obj, pyrosetta.rosetta.core.pose.Pose)
        repack_only_pos (list) :: list of integers of positions that should only repacked
        design_pos (list) :: list of integers of positions that must be redesigned
        ref_resno (int) :: resno of the target residue
        heavyatoms (list) :: list of target residue heavyatoms (target residue is the last one in the pose)
        cuts (list) :: distances from any heavyatom in the ligand to determine in which layer a residue should belong to.
        design_GP (bool) :: if True, allows redesign of GLY and PRO
    """
    # Getting the redesign / repack / do-not-touch layers for FastDesign
    # (It is not possible to create these layers using ResidueSelectors the way EnzDes does it)
    residues = get_packer_layers(pose, ref_resno, cuts, heavyatoms, repack_only_pos, design_pos, design_GP)

    # Creating a ResidueSelector for repack layercuts
    SEL_repack_residues = ResidueIndexSelector()
    for res in residues[2] + residues[3]:
        SEL_repack_residues.append_index(res)

    # Creating a ResidueSelector for do-not-touch layer
    SEL_do_not_repack = ResidueIndexSelector()
    for res in residues[4]:
        SEL_do_not_repack.append_index(res)

    # Creating a list of residues that will be mutated.
    # Including RIF residues that ended up not being used in the constraints
    # This means that almost all apolar RIF residues will be redesigned
    SEL_mutate_residues = ResidueIndexSelector()
    for res in residues[0] + residues[1]:
        SEL_mutate_residues.append_index(res)
    return SEL_mutate_residues, SEL_repack_residues, SEL_do_not_repack, residues


def check_cst_score(pose, cst_filter, cst=None):
    _pose = pose.clone()
    if not _pose.constraint_set().has_constraints():
        cst.apply(_pose)
    return cst_filter.score(_pose)


def mutate_residues(pose, resnos, resname3):
    """
    Mutates given residues to alanines.
    Arguments:
        pose (obj, pyrosetta.rosetta.core.pose.Pose)
        resnos (list) :: list of integers
    """
    # assert isinstance(pose, pyrosetta.rosetta.core.pose.Pose), f"pose: Invalid input type: {pose.__class__}"
    assert isinstance(resnos, list), "mutate_residues: Invalid input type"
    assert all([isinstance(x, int) for x in resnos]), "mutate_residues: expected a list of integers"

    # if isinstance(pose, pyrosetta.distributed.packed_pose.core.PackedPose):
    #     pose2 = pose.pose.clone()
    # else:
    #     pose2 = pose.clone()
    pose2 = pose.clone()
    mutres = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()
    for res in resnos:
        mutres.set_target(res)
        mutres.set_res_name(resname3)
        mutres.apply(pose2)
    return pose2


def fix_bb_hb_constraint(pose, ref_pose, add_cst_mover, cstfile, sfx):
    """
    Fixes the missing backbone HBond contraints that is defined in a variable
    CST block 2 as cst-5.
    """
    cstio = add_cst_mover.get_const_EnzConstraintIO_for_cstfile(cstfile)
    cstio.process_pdb_header(ref_pose, True)
    # chem_manager = pyrosetta.rosetta.core.chemical.ChemicalManager.get_instance()
    # residue_type_set = chem_manager.residue_type_set("fa_standard")
    # cst_io = pyrosetta.rosetta.protocols.toolbox.match_enzdes_util.EnzConstraintIO(residue_type_set)
    # cst_io.read_enzyme_cstfile(enzyme_cst_f)

    if cstio.num_mcfi_lists() == 2:
        obs = pyrosetta.rosetta.protocols.toolbox.match_enzdes_util.get_enzdes_observer(ref_pose)
        cst_positions = obs.cst_cache().param_cache(2).active_pose_constraints()[1].residues()
        mcfi = cstio.mcfi_list(2)

        # Figuring the mcfi number out from the pdbfile because this information
        # cannot be parsed from the pose.
        mcfi_no = None
        _pdbf = open(pose.pdb_info().name(), "r").readlines()
        for l in _pdbf:
            if "REMARK 666" in l:
                if l.split()[11] == str(cst_positions[2]):
                    mcfi_no = int(l.split()[13])
        # if cstio.is_backbone_only_cst(ref_pose, cst_positions[2]):
        #     mcfi_no = 5
        # else:
        #     for n in range(1, mcfi.num_mcfis()+1):
        #         if pose.residue(cst_positions[2]).name3() in mcfi.mcfi(5).allowed_res_name3s(2):
        #             mcfi_no = n
        #             break
        if mcfi_no is None:
            sys.exit("Unable to find correct MCFI to apply")

        cstio.enz_cst_params(2).set_mcfi(cstio.mcfi_list(2).mcfi(mcfi_no))
        cstio.enz_cst_params(2).update_pdb_remarks(pose)
        cstio.add_pregenerated_constraints_to_pose(pose, sfx)
        cstio.add_constraints_to_pose_for_block_without_clearing_and_header_processing(pose, sfx, 2)
        print("CST exists ", cst_positions,
              pose.constraint_set().residue_pair_constraint_exists(cst_positions[1], cst_positions[2]))

        # In case somehow there are duplicate constraints on the pose,
        # then remove one of them
        _remove_duplicate_csts(pose)


def _remove_duplicate_csts(pose):
    cst_set = pose.constraint_set() 
    csts = cst_set.get_all_constraints()

    duplicate_csts = {}
    
    for cst1 in csts:
        for cst2 in csts:
            if id(cst1) == id(cst2):  # Skipping those with same memory address
                continue
            if cst1 == cst2:
                residues = cst1.residues()
                label = f"{residues[1]}-{residues[2]}"
                if label not in duplicate_csts.keys():
                    duplicate_csts[label] = (cst1, cst2)

    for label in duplicate_csts:
        pose.constraint_set().remove_constraint(duplicate_csts[label][0], True)


def add_cst_wrapped_to_fix_bb(pose, ref_pose, add_cst_mover, cstfile, sfx):
    error = False
    try:
        add_cst_mover.apply(pose.clone())
    except RuntimeError:
        error = True
        fix_bb_hb_constraint(pose, ref_pose, add_cst_mover, cstfile, sfx)
    if error is False:
        add_cst_mover.apply(pose)


def get_crude_fastrelax(fastrelax):
    """
    Modifies your fastrelax method to run a very crude relax script
    """
    _fr = fastrelax.clone()
    script = ["coord_cst_weight 0.5",
              "scale:fa_rep 0.280",
              "repack",
              "min 0.01",
              "coord_cst_weight 0.0",
              "scale:fa_rep 1",
              "repack",
              "min 0.01",
              "accept_to_best"]
    filelines = pyrosetta.rosetta.std.vector_std_string()
    [filelines.append(l.rstrip()) for l in script]
    _fr.set_script_from_lines(filelines)
    return _fr


def remove_ligand_from_pose(pose, resno=None):
    """
    Returns a pose object that has ligand residue removed.
    Arguments:
        pose (object, pyrosetta.rosetta.core.pose.Pose)
        resno (int) :: optional; residue number that will be deleted
    """
    nt = type(None)
    assert isinstance(pose, pyrosetta.rosetta.core.pose.Pose), "pose: Invalid input type"
    assert isinstance(resno, (nt, int)), "resno: Invalid input type"

    drm = DeleteRegionMover()
    pose2 = pose.clone()
    if resno is not None:
        if pose.residue(resno).is_ligand():
            drm.region(str(resno), str(resno))
    else:
        for rn in range(pose.size()):
            if pose.residue(rn+1).is_ligand():
                drm.region(str(rn+1), str(rn+1))
                break
    drm.apply(pose2)
    return pose2


def separate_protein_and_ligand(pose):
    """
    Separates the ligand and the rest of the pose to 'infinity'
    (to 666 angstrom).
    Assumes ligand is the last residue in the pose.
    Arguments:
        pose (object, pyrosetta.rosetta.core.pose.Pose)
    """
    assert isinstance(pose, pyrosetta.rosetta.core.pose.Pose), "pose: Invalid input type"

    tmp_pose = pose.clone()
    lig_seqpos = tmp_pose.size()
    lig_jump_no = tmp_pose.fold_tree().get_jump_that_builds_residue(lig_seqpos)
    rbt = pyr.rosetta.protocols.rigid.RigidBodyTransMover(tmp_pose, lig_jump_no)
    rbt.step_size(666)
    rbt.apply(tmp_pose)
    return tmp_pose


def check_bb_clash(pose, resno):
    """
    Checks if any heavyatom in the defined residue clashes with any backbone
    atom in the pose
    check_bb_clash(pose, resno) -> bool
    Arguments:
        pose (obj, pose)
        resno (int)
    """

    # Calculating the center of mass of the target residue
    trgt = pose.residue(resno)
    target_com = xyzVector_double_t()
    for g in 'xyz':
        crd = np.average([a.xyz().__getattribute__(g) for a in trgt.atoms()])
        target_com.__setattr__(g, crd)


    # Selecting residues that have CA withing 14A of the center of the mass
    # of the target residue. This is to reduce the search space.
    nbrs = []
    for res in pose.residues:
        if not res.is_ligand():
            if (res.xyz('CA') - target_com).norm() < 14.0:
                nbrs.append(res)

    # Iterating over each heavyatom in the target and checking if it clashes
    # with any backbone atom of any of the neighboring residues
    LIMIT = 1.0
    clash = False
    for res_atom_no in range(1, trgt.natoms()+1):
        if not trgt.atom_is_hydrogen(res_atom_no):
            res_atom = trgt.atom(res_atom_no)
            for res in nbrs:
                if res.seqpos == resno:
                    continue
                for bb_no in res.all_bb_atoms():
                    dist = (res_atom.xyz() - res.xyz(bb_no)).norm()
                    if dist < LIMIT:
                        clash = True
                        break
                if clash:
                    break
            if clash:
                break
    return clash


def find_clashes_between_target_and_sidechains(pose, target_resno, target_atoms=None, residues=None):
    if residues is None:
        residues = list(range(1, pose.size()+1))
        residues = [r for r in residues if pose.residue(r).is_protein()]
    
    if target_atoms is None:
        res = pose.residue(target_resno)
        target_atoms = [res.atom_name(n).strip() for n in range(1, res.natoms()+1)]
    clashes = []
    for resno in residues:
        for atm in pose.residue(resno).atoms():
            if min([(pose.residue(target_resno).xyz(ha)-atm.xyz()).norm() for ha in target_atoms]) < 2.5:
                clashes.append(resno)
                break
    return clashes


def translate_to_target(pose, target_pose, resno, target_resno, atomname, target_atomname):
    pose2 = pose.clone()
    shift = pose.residue(resno).xyz(atomname) - target_pose.residue(target_resno).xyz(target_atomname)
    for n in range(pose2.residue(resno).natoms()):
        pose2.residue(resno).set_xyz(n+1, pose2.residue(resno).xyz(n+1) - shift)
    return pose


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

    Written by Patrick Salveson, provided by Meerit Said
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
    spliced_pose = Pose()
    append_subpose_to_pose(spliced_pose, target, 1, target.size() - 1)
    append_subpose_to_pose(spliced_pose, loop, 1, loop_length)
    ###############################################################

    return spliced_pose


def replace_ligand_in_pose(pose, residue, resno, ref_atoms, new_atoms):
    _tmp_ligpose = pyrosetta.rosetta.core.pose.Pose()
    _tmp_ligpose.append_residue_by_jump(residue, 0)
    new_pose = match_sidechain_atoms(pose, resno, ref_atoms,
                                     _tmp_ligpose, 1, new_atoms)
    return new_pose


# def get_rotamers_for_res_in_pose(in_pose, target_residue, sfx, ex1=True,
#                                  ex2=True, ex3=False, ex4=False,
#                                  check_clashes=True):
#     packTask = pyrosetta.rosetta.core.pack.task.TaskFactory.create_packer_task(in_pose)
#     packTask.set_bump_check(check_clashes)
#     resTask = packTask.nonconst_residue_task(target_residue)
#     resTask.or_ex1(ex1)
#     resTask.or_ex2(ex2)
#     resTask.or_ex3(ex3)
#     resTask.or_ex4(ex4)
#     resTask.restrict_to_repacking()
#     packer_graph = pyrosetta.rosetta.utility.graph.Graph(in_pose.size())
#     rsf = pyrosetta.rosetta.core.pack.rotamer_set.RotamerSetFactory()
#     rotset = rsf.create_rotamer_set(in_pose)
#     rotset.set_resid(target_residue)
#     rotset.build_rotamers(in_pose, sfx, packTask, packer_graph, False)
#     print("Found num_rotamers:", rotset.num_rotamers())
#     rotamer_set = []
#     for irot in range(1, rotset.num_rotamers() + 1):
#         rotamer_set.append(rotset.rotamer(irot).clone())
#     return rotamer_set


def get_rotamers_for_res_in_pose(in_pose, target_residue, sfx, ex1=True,
                                 ex2=True, ex3=False, ex4=False,
                                 check_clashes=True, all_rotamers=False):
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
    
    if all_rotamers is True:
        ## This hack will go around the Rosetta internal limitation
        ## for the number of proton_chi subsamplings
        rotlib = pyrosetta.rosetta.core.pack.rotamers.SingleResidueRotamerLibrary()
        samplings = []
        if len(resTask.allowed_residue_types()) != 1:
            print(f"Residue has multiple types or something: {resTask.allowed_residue_types()}")
        rotamer_set = []
        for _rt in resTask.allowed_residue_types():
            samplings.append(rotlib.compute_proton_chi_samplings(_rt, resTask, True))
            proton_chi_chisets = rotlib.expand_proton_chis(samplings[-1], _rt, np.prod([len(x) for x in samplings[-1]]))
    
            rotamers = [None]*len(proton_chi_chisets)
            for i, cs in enumerate(proton_chi_chisets):
                rotamers[i] = in_pose.residue(target_residue).clone()
                for jj in range(1, _rt.n_proton_chi()+1):
                    jj_chi = _rt.proton_chi_2_chi(jj)
                    rotamers[i].set_chi(jj_chi, cs.chi[jj_chi])
            rotamer_set += rotamers
        print(f"Found {len(rotamer_set)} rotamers")
    else:
        rotset.build_rotamers(in_pose, sfx, packTask, packer_graph, False)
        print("Found num_rotamers:", rotset.num_rotamers())
        rotamer_set = []
        for irot in range(1, rotset.num_rotamers() + 1):
            rotamer_set.append(rotset.rotamer(irot).clone())
    return rotamer_set


def get_residues_with_close_sc(pose, ref_atoms, residues=None, exclude_residues=None, cutoff=4.5):
    """
    """
    if residues is None:
        residues = [x for x in range(1, pose.size()+1)]
    if exclude_residues is None:
        exclude_residues = []

    close_ones = []
    for resno in residues:
        if resno in exclude_residues:
            continue
        if pose.residue(resno).is_ligand():
            continue
        res = pose.residue(resno)
        close_enough = False
        for atomno in range(1, res.natoms()):
            if res.atom_type(atomno).is_heavyatom():
                for ha in ref_atoms:
                    if (res.xyz(atomno) - pose.residue(pose.size()).xyz(ha)).norm() < cutoff:
                        close_enough = True
                        close_ones.append(resno)
                        break
                    if close_enough is True:
                        break
                if close_enough is True:
                    break
    return close_ones


def repack(pose, scorefxn):
    """
    Performs repacking of an input pose.
    If repack_residues_list is provided then it generates a ResidueSelector
    out of that list, allowing only these residues to be repacked.
    All other residues are nto touched.
    Arguments:
        pose (object, pyrosetta.rosetta.core.pose.Pose)
        scrfxn (obj, pyrosetta.rosetta.core.scoring.ScoreFunction)
        repack_residues_list (list) :: list of integers
    """
    nt = type(None)
    assert isinstance(pose, pyrosetta.rosetta.core.pose.Pose), f"pose: Invalid input type: {pose.__class__}"
    assert isinstance(scorefxn, pyrosetta.rosetta.core.scoring.ScoreFunction), f"scorefxn: Invalid input type: {scorefxn.__class__}"

    tmp_pose = pose.clone()

    # Now lets see how to set up task operations
    # The task factory accepts all the task operations
    tf = pyrosetta.rosetta.core.pack.task.TaskFactory()

    # These three are pretty standard
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.IncludeCurrent())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.NoRepackDisulfides())

    e = pyrosetta.rosetta.core.pack.task.operation.ExtraRotamersGeneric()
    e.ex1(False)
    e.ex2(False)
    e.ex1aro(False)
    tf.push_back(e)

    # We are not designing, so this will also be used
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.RestrictToRepacking())

    # Have to convert the task factory into a PackerTask
    task = tf.create_task_and_apply_taskoperations(tmp_pose)

    pack_mover = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(scorefxn, task)
    pack_mover.apply(tmp_pose)

    return tmp_pose


def thread_and_repack(pose, sequence, sfx):
    _pose2 = thread_seq_to_pose(pose, sequence)
    pose3 = repack(_pose2, sfx)
    return pose3


def thread_seq_to_pose(pose, sequence):
    pose2 = pose.clone()
    for i, r in enumerate(sequence):
        if pose.residue(i+1).name1() == r:
            continue
        mutres = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()
        mutres.set_target(i+1)
        mutres.set_res_name(aa_1_3[r])
        mutres.apply(pose2)
    return pose2


