import os,sys
import mock
import numpy as np
import tempfile
from typing import Dict
from timeit import default_timer as timer

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.append(f"{SCRIPT_DIR}/../../lib/alphafold")
from alphafold.common import protein
from alphafold.data import pipeline
from alphafold.data import templates
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model

from jax.lib import xla_bridge

os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '2.0'


def predict_sequences(sequences, models, nrecycles, scorefile=None, random_seed=None, nstruct=1, npy=False):
    # setup which models to use
    # note for demo, we are only using model_4
    _models_start = timer()
    model_runners = {}
    for m in models:
        model_name = 'model_'+m
        model_config = config.model_config(model_name)
        model_config.data.eval.num_ensemble = 1

        model_config.model.num_recycle = nrecycles
        model_config.data.common.num_recycle = nrecycles
    
        model_config.data.common.max_extra_msa = 1
        model_config.data.eval.max_msa_clusters = 1

        model_params = data.get_model_haiku_params(model_name=model_name, data_dir=f"{SCRIPT_DIR}/../../lib/alphafold/model_weights")
        model_runner = model.RunModel(model_config, model_params)
        model_runners[model_name] = model_runner
    print(f"Setting up models took {(timer() - _models_start):.3f} seconds.")

    i = 0
    predictions = []
    prefix = tempfile._get_default_tempdir() + '/' + next(tempfile._get_candidate_names())
    print(prefix)

    _st = timer()
    for sequence in sequences:
        query_sequence = sequence[0]
        for n in range(nstruct):
            pdb_file = f"{prefix}{i}_{n}"
            start = timer()

            # mock pipeline for testing
            data_pipeline_mock = mock.Mock()
            data_pipeline_mock.process.return_value = {
                **pipeline.make_sequence_features(sequence=query_sequence,
                                                  description="none",
                                                  num_res=len(query_sequence)),
                **pipeline.make_msa_features(msas=[[query_sequence]],
                                             deletion_matrices=[[[0]*len(query_sequence)]]),
                **mk_mock_template(query_sequence)
            }

            if random_seed is None and nstruct == 1:
                random_seed = 0
            elif random_seed is not None or nstruct > 1:
                random_seed = np.random.randint(99999)
                print(f"Random seed = {random_seed}")

            results = predict_structure(
                 pdb_file=pdb_file,
                 data_pipeline=data_pipeline_mock,
                 model_runners=model_runners,
                 random_seed=random_seed
            )

            time = timer() - start;
  
            for result in results:      
                lddt = np.mean(result['lddts'])

                pred = {'i': i,
                        'tag': result['model'],
                        'sequence': query_sequence,
                        'description': sequence[1],
                        'nrecycles': nrecycles,
                        'lddt': lddt,
                        # 'pdb_file': result['pdb_file'],
                        # 'npy_file': result['npy_file'],
                        'time': time}

                predictions.append(pred)
                
                # Dump PDB file
                fn = f"{sequence[1]}_{result['model']}.{n}_r{nrecycles}_af2"
                _pdbf = open(result['pdb_file'], "rb").read()
                with open(f"{fn}.pdb", "wb") as file:
                    file.write(_pdbf)

                if npy is True:
                    # Dumping the NPZ file
                    _npy = open(result['npy_file'], "rb").read()
                    with open(f"{fn}.npz", "wb") as file:
                        file.write(_npy)


                # Add line to scorefile
                if scorefile is not None:
                    with open(scorefile, "a") as sf:
                        sf.write("%d,%s,%s,%s,%s,%.3f,%.1f\n" % (
                                pred['i'],
                                pred['description'],
                                pred['sequence'],
                                pred['tag'],
                                fn,
                                pred['lddt'],
                                pred['time']
                                ))
            print("Sequence %d completed in %.1f sec with %d models; lDDT=%.3f" % (i, time, len(results), lddt))
        i += 1

    print(f"Done with {i} sequences. {(timer() - _st):.3f} sec.")

    return predictions


def mk_mock_template(query_sequence):
    # mock template features
    output_templates_sequence = []
    output_confidence_scores = []
    templates_all_atom_positions = []
    templates_all_atom_masks = []

    for _ in query_sequence:
        templates_all_atom_positions.append(np.zeros((templates.residue_constants.atom_type_num, 3)))
        templates_all_atom_masks.append(np.zeros(templates.residue_constants.atom_type_num))
        output_templates_sequence.append('-')
        output_confidence_scores.append(-1)
    output_templates_sequence = ''.join(output_templates_sequence)
    templates_aatype = templates.residue_constants.sequence_to_onehot(output_templates_sequence,
                                                                    templates.residue_constants.HHBLITS_AA_TO_ID)

    template_features = {'template_all_atom_positions': np.array(templates_all_atom_positions)[None],
        'template_all_atom_masks': np.array(templates_all_atom_masks)[None],
        'template_sequence': [f'none'.encode()],
        'template_aatype': np.array(templates_aatype)[None],
        'template_confidence_scores': np.array(output_confidence_scores)[None],
        'template_domain_names': [f'none'.encode()],
        'template_release_date': [f'none'.encode()]}
        
    return template_features


def predict_structure(
    pdb_file: str,
    data_pipeline: pipeline.DataPipeline,
    model_runners: Dict[str, model.RunModel],
    random_seed: int):
  
    """Predicts structure using AlphaFold for the given sequence."""

    # Get features.
    feature_dict = data_pipeline.process()

    # Run the models.
    results = []
    for model_name, model_runner in model_runners.items():
      print("Predicting with ", model_name)
      processed_feature_dict = model_runner.process_features(feature_dict, random_seed=random_seed)
      prediction_result = model_runner.predict(processed_feature_dict)
      unrelaxed_protein = protein.from_prediction(processed_feature_dict,prediction_result)

      model_pdb_file = pdb_file + '_' + model_name
      with open(model_pdb_file, 'w') as f:
          f.write(protein.to_pdb(unrelaxed_protein))

      model_npy_file = model_pdb_file + '.npy'
      np.save(model_npy_file, prediction_result['plddt'])

      results.append({ 'lddts': prediction_result['plddt'], 'pdb_file': model_pdb_file, 'npy_file': model_npy_file, 'model': model_name })

    return results

