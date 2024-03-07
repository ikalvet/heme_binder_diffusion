#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 22:07:05 2022

@author: indrek
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:47:06 2019

@author: ikalvet
"""

import pyrosetta as pyr
import pyrosetta.rosetta
import os
import sys
import numpy as np
import time
import random
import datetime
import pandas as pd
import json
import argparse
from pyrosetta.rosetta.protocols.grafting.simple_movers import DeleteRegionMover
from pyrosetta.rosetta.utility import vector1_numeric_xyzVector_double_t as xyzVector
from pyrosetta.rosetta.numeric import xyzMatrix_double_t
from pyrosetta.rosetta.numeric import xyzVector_double_t
from pyrosetta.rosetta.protocols.toolbox import superposition_transform
from pyrosetta.rosetta.protocols.toolbox import apply_superposition_transform
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.core.pose import append_subpose_to_pose
from pyrosetta.rosetta.core.select.residue_selector import ResidueIndexSelector

SCRIPT_PATH = os.path.dirname(__file__)
sys.path.append(f"{SCRIPT_PATH}/../utils")
import design_utils
import no_ligand_repack
import scoring_utils


def find_ligand_seqpos(pose):
    ligand_seqpos = None
    for res in pose.residues:
        if res.is_ligand() and not res.is_virtual_residue():
            ligand_seqpos = res.seqpos()
    return ligand_seqpos


"""
Parsing user input
"""
if __name__ == "__main__":
    # Parsing keyword arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--pdb", type=str, required=True, help="input PDB from hallucination or inpainting")
    parser.add_argument("--suffix", type=str, default="", help="suffix used in output files")
    parser.add_argument("--nstruct", type=int, default=1, help="how many design iterations?")
    parser.add_argument("--cstfile", type=str, help="What CST file to use")
    parser.add_argument("--params", nargs="+", help="params files")
    parser.add_argument("--norelax", action="store_true", default=False, help="Unconstrained relax will not be performed after design.")

    args = parser.parse_args()


    special_rotamers = False
    random_seed = False

    pdbfile = args.pdb
    NSTRUCT = args.nstruct

    suffix = args.suffix
    if suffix != "":
        suffix += "_"

    cstfile = args.cstfile
    scorefilename = "scorefile.txt"

    outname = f"{os.path.basename(pdbfile).replace('.pdb','')}_{suffix}DE"

    """
    Setting up Rosetta
    """
    extra_res_fa = ""
    if args.params is not None:
        extra_res_fa = '-extra_res_fa'
        for p in args.params:
            if os.path.exists(p):
                extra_res_fa += " {}".format(p)

    DAB = f"{SCRIPT_PATH}/../utils/DAlphaBall.gcc" # This binary was compiled on UW systems. It may or may not work correctly on yours
    assert os.path.exists(DAB), "Please compile DAlphaBall.gcc and manually provide a path to it in this script under the variable `DAB`\n"\
                            "For more info on DAlphaBall, visit: https://www.rosettacommons.org/docs/latest/scripting_documentation/RosettaScripts/Filters/HolesFilter"

    pyr.init(f"-dalphaball {DAB} -beta_nov16 {extra_res_fa} -run:preserve_header")


    pyrosetta.rosetta.basic.options.set_integer_option("packing:ex1:level", 2)
    pyrosetta.rosetta.basic.options.set_integer_option("packing:ex2:level", 2)
    pyrosetta.rosetta.basic.options.set_integer_option("packing:ex3:level", 1)
    pyrosetta.rosetta.basic.options.set_integer_option("packing:ex4:level", 1)

    scorefxn = pyr.get_fa_scorefxn()


    """
    Processing the input PDB
    """
    pose = pyr.pose_from_file(pdbfile)

    # Finding the correct ligand
    ligand_name = pose.residue(pose.size()).name3()

    for p in args.params:
        if ligand_name in p:
            ligand_pdb = p.replace('.params', '.pdb')
            if not os.path.exists(ligand_pdb):
                sys.exit("Can't find ligand PDB file. Exiting...")
            ligand_pose = pyr.pose_from_file(ligand_pdb)

    posex = pose.clone()

    ligand_seqpos = find_ligand_seqpos(pose)

    # Hardcoded atom names for substrate atoms in the ligand
    substrate_atom_names = {"HBA": [f"C{n}" for n in range(35, 48)] + ["O6"]}

    if pose.residue(ligand_seqpos).name3() not in substrate_atom_names:
        sys.exit(f"Please define substrate atom names for ligand {pose.residue(ligand_seqpos).name3()}")

    substrate_atomnos = []
    for n in range(1, pose.residue(ligand_seqpos).natoms()+1):
        if pose.residue(ligand_seqpos).atom_name(n).strip() in substrate_atom_names[ligand_name]:
            substrate_atomnos.append(n)

    pose2 = pose.clone()


    """
    Setting up design/repack layers
    """
    matched_residues = design_utils.get_matcher_residues(pdbfile)
    his_seqpos = list(matched_residues.keys())[0]
    assert pose.residue(his_seqpos).name3() in ["HIS", "CYS", "CYX"], f"Bad coordinating residue: {pose.residue(his_seqpos).name3()}-{his_seqpos}"

    bad_residues = [res.seqpos() for res in pose.residues if res.name3() in ["MET", "CYS"] and res.seqpos() not in matched_residues.keys()]



    # Finding out what residues belong to what layer, based on the CA distance
    # from ligand heavyatoms.
    heavyatoms = design_utils.get_ligand_heavyatoms(pose2)
    SEL_mutate_residues, SEL_repack_residues, SEL_do_not_repack, residues\
        = design_utils.get_layer_selections(pose2, repack_only_pos=list(matched_residues.keys()),
                                            design_pos=bad_residues, ref_resno=ligand_seqpos, heavyatoms=heavyatoms)


    # Only designing positions that are close to ligand
    design_residues = []
    repack_residues = []
    do_not_touch_residues = []
    for resno in residues[0]+residues[1]:
        design_residues.append(resno)


    for resno in residues[2]+residues[3]:
        repack_residues.append(resno)
    
    for resno in residues[-1]:
        do_not_touch_residues.append(resno)


    repack_residues.append(ligand_seqpos)

    design_residues = list(set(design_residues))
    repack_residues = list(set(repack_residues))
    do_not_touch_residues = list(set(do_not_touch_residues))

    unclassified_residues = [res.seqpos() for res in pose.residues if res.seqpos() not in design_residues+repack_residues+do_not_touch_residues]
    assert len(unclassified_residues) == 0, f"Some residues have not been layered: {unclassified_residues}"

    # Saving no-ligand-repack residues.
    # Basically a list of all residues that are close-ish to the ligand.
    nlr_repack_residues = design_residues + repack_residues

    design_res = ','.join([str(x) for x in design_residues])
    repack_res = ','.join([str(p) for p in repack_residues])
    do_not_touch_res = ','.join([str(p) for p in do_not_touch_residues])
    nlr_repack_res = ','.join([str(p) for p in nlr_repack_residues])

    ws = '         '
    xml_script = f"""
    <ROSETTASCRIPTS>  
      <SCOREFXNS>
            <ScoreFunction name="sfxn_design" weights="beta_nov16">
                <Reweight scoretype="res_type_constraint" weight="0.3"/>
                <Reweight scoretype="arg_cation_pi" weight="1.5"/>
                Reweight scoretype="approximate_buried_unsat_penalty" weight="5"/>
                <Reweight scoretype="aa_composition" weight="1.0" />
                <Reweight scoretype="fa_elec" weight="1.2" />
                Set approximate_buried_unsat_penalty_burial_atomic_depth="3.5"/>
                Set approximate_buried_unsat_penalty_hbond_energy_threshold="-0.5"/>
                Set approximate_buried_unsat_penalty_hbond_bonus_cross_chain="-1"/>

                for enzyme design
                <Reweight scoretype="atom_pair_constraint" weight="1.0"/>
                <Reweight scoretype="dihedral_constraint" weight="1.0"/>
                <Reweight scoretype="angle_constraint" weight="1.0"/>
                <Reweight scoretype="coordinate_constraint" weight="1.0"/>
            </ScoreFunction>

            <ScoreFunction name="sfxn_clean" weights="beta_nov16">
                <Reweight scoretype="arg_cation_pi" weight="3"/>
            </ScoreFunction>

            <ScoreFunction name="fa_csts" weights="beta_nov16">
                <Reweight scoretype="atom_pair_constraint" weight="1.0"/>
                <Reweight scoretype="dihedral_constraint" weight="1.0"/>
                <Reweight scoretype="angle_constraint" weight="1.0"/>
                <Reweight scoretype="coordinate_constraint" weight="1.0"/>
            </ScoreFunction>
      </SCOREFXNS>

      <RESIDUE_SELECTORS>
          <Layer name="init_core_SCN" select_core="True" use_sidechain_neighbors="True" surface_cutoff="1.0" />
          <Layer name="init_boundary_SCN" select_boundary="True" use_sidechain_neighbors="True" surface_cutoff="1.0" />
          <Layer name="surface_SCN" select_surface="True" use_sidechain_neighbors="True" surface_cutoff="1.0" />
          <Layer name="coreRes" select_core="true" use_sidechain_neighbors="true" core_cutoff="2.1" surface_cutoff="1.0"/>

          <Index name="cat_residues" resnums="{his_seqpos}"/>
          <Index name="cat_his"      resnums="{his_seqpos}"/>

          <Index name="ligand_idx" resnums="{ligand_seqpos}"/>
          
          <Index name="design_idx" resnums="{design_res}"/>
          <Index name="repack_idx" resnums="{repack_res}"/>
          <Index name="do_not_touch_idx" resnums="{do_not_touch_res}"/>
          <Index name="nlr_repack_idx" resnums="{nlr_repack_res}"/>

          <Chain name="chainA" chains="A"/>
          <Chain name="chainB" chains="B"/>

      </RESIDUE_SELECTORS>

      <TASKOPERATIONS>
          <RestrictAbsentCanonicalAAS name="noCys" keep_aas="ADEFGHIKLMNPQRSTVWY"/>
          <PruneBuriedUnsats name="prune_buried_unsats" allow_even_trades="false" atomic_depth_cutoff="3.5" minimum_hbond_energy="-0.5" />
          <LimitAromaChi2 name="limitchi2" chi2max="110" chi2min="70" include_trp="True" />
          <ExtraRotamersGeneric name="ex1_ex2aro" ex1="1" ex2aro="1" />
          <SetCatalyticResPackBehavior name="catpack" fix_catalytic_aa="0" />
          <DetectProteinLigandInterface name="dpli" cut1="6.0" cut2="8.0" cut3="10.0" cut4="12.0" design="1" catres_only_interface="0" arg_sweep_interface="0" />
          <IncludeCurrent name="ic"/>

          <OperateOnResidueSubset name="repack_extended" selector="nlr_repack_idx">
              <RestrictToRepackingRLT/>
          </OperateOnResidueSubset>

          <OperateOnResidueSubset name="only_repack" selector="repack_idx">
              <RestrictToRepackingRLT/>
          </OperateOnResidueSubset>

          <OperateOnResidueSubset name="do_not_touch" selector="do_not_touch_idx">
              <PreventRepackingRLT/>
          </OperateOnResidueSubset>

            <OperateOnResidueSubset name="catres_rotamer" selector="cat_his">
                <ExtraRotamersGenericRLT ex1="1" ex2aro="1" ex1_sample_level="3"/>
            </OperateOnResidueSubset>

          <OperateOnResidueSubset name="disallow_MCP" selector="design_idx">
              <RestrictAbsentCanonicalAASRLT aas="EDQNKRHSTYLIAVFWG"/>
          </OperateOnResidueSubset>

          <ProteinInterfaceDesign name="pack_long" design_chain1="0" design_chain2="0" jump="1" interface_distance_cutoff="15"/>
      </TASKOPERATIONS>

      <MOVERS>
          <AddOrRemoveMatchCsts name="add_enz_csts" cstfile="{cstfile}" cst_instruction="add_new"/>

          <EnzRepackMinimize name="cst_opt_new" scorefxn_repack="sfxn_design" scorefxn_minimize="sfxn_design" cst_opt="1" minimize_sc="1" repack_only="1" minimize_bb="1" cycles="1" min_in_stages="0" minimize_lig="1" task_operations="dpli,ic,catpack"/>

          <TaskAwareMinMover name="min" scorefxn="sfxn_clean" bb="0" chi="1" task_operations="pack_long" />
          <PackRotamersMover name="pack" scorefxn="sfxn_clean" task_operations="repack_extended,do_not_touch,ic,limitchi2,ex1_ex2aro,catres_rotamer"/>

          <AddHelixSequenceConstraints name="helix_constraint" residue_selector="design_idx" 
           min_n_terminal_charges="0" min_c_terminal_charges="0" 
           n_terminal_constraint_strength="5.0" c_terminal_constraint_strength="5.0"
           overall_max_count="1"/>

          <FastRelax name="fastRelax" scorefxn="sfxn_clean" repeats="1" task_operations="ex1_ex2aro,ic,catres_rotamer"/>

          <FastDesign name="fastDesign" scorefxn="sfxn_design" repeats="1" task_operations="ex1_ex2aro,only_repack,do_not_touch,ic,limitchi2,disallow_MCP,catres_rotamer" batch="false" ramp_down_constraints="false" cartesian="False" bondangle="false" bondlength="false" min_type="dfpmin_armijo_nonmonotone" relaxscript="MonomerDesign2019"/> 

          <ClearConstraintsMover name="rm_csts" />

          <ScoreMover name="scorepose" scorefxn="sfxn_clean" verbose="false" />

      </MOVERS>

      <FILTERS>

          <ScoreType name="totalscore" scorefxn="sfxn_clean" threshold="9999" confidence="1"/>
          <ResidueCount name="nres" confidence="1" />
          <CalculatorFilter name="score_per_res" confidence="1" equation="SCORE/NRES" threshold="999">
              <Var name="SCORE" filter_name="totalscore" />
              <Var name="NRES" filter_name="nres" />
          </CalculatorFilter>

          <Geometry name="geom" count_bad_residues="true" confidence="0"/>

          <Ddg name="ddg_norepack"  threshold="0" jump="1" repeats="1" repack="0" relax_mover="min" confidence="0" scorefxn="sfxn_clean"/>	
          <Report name="ddg2" filter="ddg_norepack"/>

          <Sasa name="interface_buried_sasa" confidence="0" />
          <InterfaceHydrophobicResidueContacts name="hydrophobic_residue_contacts" target_selector="ligand_idx" binder_selector="chainA" scorefxn="sfxn_clean" confidence="0"/>

          <EnzScore name="all_cst" scorefxn="fa_csts" confidence="0" whole_pose="1" score_type="cstE" energy_cutoff="100.0"/>

      </FILTERS>

    <PROTOCOLS>
        <Add mover="add_enz_csts"/>
    </PROTOCOLS>
    </ROSETTASCRIPTS>
    """


    # run protocol and save output

    # Creating an XML object from string.
    # This allows extracting movers and stuff from the object,
    # and apply them separately.
    obj = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string(xml_script)

    # Fetching movers
    add_enz_csts = obj.get_mover("add_enz_csts")
    fastDesign_all_new = obj.get_mover("fastDesign")
    rm_csts = obj.get_mover("rm_csts")
    cst_opt_new = obj.get_mover("cst_opt_new")
    fastRelax = obj.get_mover("fastRelax")
    fastRelax.constrain_relax_to_start_coords(True)
    fastDesign_all_new.constrain_relax_to_start_coords(True)
    
    packer = obj.get_mover("pack")
    
    sfx = obj.get_score_function("sfxn_clean")
    sfx_cst = obj.get_score_function("sfxn_design")
    
    fastRelax_cst = fastRelax.clone()
    fastRelax_cst.set_scorefxn(sfx_cst)
    fastRelax_cst = design_utils.get_crude_fastrelax(fastRelax_cst)
    
    all_cst = obj.get_filter("all_cst")


    for N in range(1, NSTRUCT+1):
        print(f"Design iteration {N} on pose: {os.path.basename(pdbfile)}")

        if os.path.exists(outname + f"_{N}.pdb"):
            print(f"Design {outname}_{N}.pdb already exists. Skipping iteration.")
            continue

        t0 = time.time()

        pose3 = pose2.clone()

        # Applying movers
        add_enz_csts.apply(pose3)
        _pose2 = pose3.clone()

        _pose3 = pose3.clone()

        # Mutating residues that are too close to heme to ALA
        clashes = design_utils.find_clashes_between_target_and_sidechains(_pose3, ligand_seqpos,
                                                                          target_atoms=heavyatoms, residues=design_residues)
        __pose3 = design_utils.mutate_residues(_pose3, clashes, "ALA")

        # Performing cst_opt until the cst score gets good enough
        # Using constrained FastRelax for cst optimization
        cst_scores = []
        for n in range(4):
            _pose3 = __pose3.clone()
            fastRelax_cst.apply(_pose3)
            cst_scores.append(all_cst.score(_pose3))
            if cst_scores[-1] <= 3.0:
                break

        if min(cst_scores) > 3.0:
            print(f"{N}: all_cst = "
                  f"{min(cst_scores):.3f}. "
                  "Too high before design, not designing.")
            with open("failed_designs.txt", "a") as file:
                file.write(f"{' '.join([f'{x:.2f}' for x in cst_scores])} {outname}_{N}.pdb\n")
            continue

        fastDesign_all_new.apply(_pose3)

        rm_csts.apply(_pose3)


        if not args.norelax:
            # Repacking without CST's
            packer.apply(_pose3)

            # Relaxing without CST's
            fastRelax.apply(_pose3)

        pose3 = _pose3.clone()

        # Fetching filters
        filters = {f: obj.get_filter(f) for f in obj.list_filters()}

        # Applying filter scores to pose
        for fltr in filters:
            if fltr == "all_cst":
                design_utils.add_cst_wrapped_to_fix_bb(pose3, _pose2, add_enz_csts, cstfile, sfx_cst)
            scoring_utils.apply_score_from_filter(pose3, filters[fltr])
            if fltr == "all_cst":
                rm_csts.apply(pose3)

        for k in ['', 'defaultscorename']:
            try:
                pose3.scores.__delitem__(k)
            except KeyError:
                continue

        sfx(pose3)

        try:
            df_scores = pd.DataFrame.from_records([pose3.scores])
        except:
            print("The protocol failed. See log.")
            sys.exit(1)

        output_pdb_iter = outname + f"_{N}.pdb"
        pose3.dump_pdb(output_pdb_iter)

        #========================================================================
        # Extra filters
        #========================================================================

        df_scores['description'] = outname + f"_{N}"


        # Get the ligand ddg, without including serine-ligand repulsion
        from scoring_utils import calculate_ddg
        sf = pyr.get_fa_scorefxn()
        def fix_scorefxn(sfxn, allow_double_bb=False):
            opts = sfxn.energy_method_options()
            opts.hbond_options().decompose_bb_hb_into_pair_energies(True)
            opts.hbond_options().bb_donor_acceptor_check(not allow_double_bb)
            sfxn.set_energy_method_options(opts)
        fix_scorefxn(sf)

        df_scores['corrected_ddg'] = calculate_ddg(pose3, sf, his_seqpos)

        # Calculating relative ligand SASA
        # First figuring out what is the path to the ligand PDB file

        free_ligand_sasa = scoring_utils.getSASA(ligand_pose, resno=1)
        ligand_sasa = scoring_utils.getSASA(pose3, resno=ligand_seqpos)
        df_scores.at[0, 'L_SASA'] = ligand_sasa / free_ligand_sasa

        # Calculating absolute SASA of the substrate atoms
        df_scores.at[0, 'substrate_SASA'] = scoring_utils.getSASA(pose3, ligand_seqpos, substrate_atomnos)

        cms = pyrosetta.rosetta.protocols.simple_filters.ContactMolecularSurfaceFilter()
        lig_sel = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(ligand_seqpos)
        protein_sel = pyrosetta.rosetta.core.select.residue_selector.ChainSelector("A")
        cms.use_rosetta_radii(True)
        cms.distance_weight(0.5)
        cms.selector1(protein_sel)
        cms.selector2(lig_sel)
        df_scores.at[0, "contact_molecular_surface"] =  cms.compute(pose3)  # ContactMolecularSurface
        df_scores.at[0, "cms_per_atom"] = df_scores.at[0, "contact_molecular_surface"] / pose3.residue(ligand_seqpos).natoms()


        ## Calculating shape complementarity
        sc = pyrosetta.rosetta.protocols.simple_filters.ShapeComplementarityFilter()
        sc.use_rosetta_radii(True)
        sc.selector1(protein_sel)
        sc.selector2(lig_sel)
        df_scores.at[0, "sc"] = sc.score(pose3)


        # Using a custom function to find HBond partners of the COO groups.
        # The XML implementation misses some interations it seems.
        for n in range(1, 5):
            df_scores.at[0, f"O{n}_hbond"] = scoring_utils.find_hbonds_to_residue_atom(pose3, ligand_seqpos, f"O{n}")


        # Checking if both COO groups on heme are hbonded
        if any([df_scores.at[0, x] > 0.0 for x in ['O1_hbond', 'O3_hbond']]) and any([df_scores.at[0, x] > 0.0 for x in ['O2_hbond', 'O4_hbond']]):
            df_scores.at[0, 'COO_hbond'] = 1.0
        else:
            df_scores.at[0, 'COO_hbond'] = 0.0


        # Adding the constraints to the pose and using it to figure out
        # exactly which position belongs to which constraint
        _pose3 = pose3.clone()
        design_utils.add_cst_wrapped_to_fix_bb(_pose3, _pose2, add_enz_csts, cstfile, sfx_cst)
        target_resno = no_ligand_repack.get_target_residues_from_csts(_pose3)
        for i, resno in enumerate(target_resno):
            df_scores.at[0, f'SR{i+1}'] = resno

        # Calculating the no-ligand-repack as it's done in enzdes
        nlr_scores = no_ligand_repack.no_ligand_repack(_pose3, sfx, ligand_resno=ligand_seqpos)

        # Measuring the Heme-CYS/HIS angle because not all deviations
        # are reflected in the all_cst score
        heme_atoms = ["N1", "N2", "N3", "N4"]
        if pose3.residue(target_resno[0]).name3() in ["CYX", "CYS"]:
            target_atom = "SG"
        elif pose3.residue(target_resno[0]).name3() == "HIS":
            target_atom = "NE2"
        else:
            target_atom = None

        if target_atom is not None:
            angles = [scoring_utils.get_angle(pose3.residue(ligand_seqpos).xyz(nx),
                                              pose3.residue(ligand_seqpos).xyz("FE1"),
                                              pose3.residue(target_resno[0]).xyz(target_atom)) for nx in heme_atoms]
            df_scores.at[0, "heme_angle_wrst"] = min(angles)

        for k in nlr_scores.keys():
            df_scores.at[0, k] = nlr_scores.iloc[0][k]

        print(df_scores.iloc[0])
        scoring_utils.dump_scorefile(df_scores, scorefilename)

        t1 = time.time()
        print(f"Design iteration {N} took {(t1-t0):.3f} seconds.")
