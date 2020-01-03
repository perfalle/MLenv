import os
import pickle, json
import pandas as pd

import cv2

MODEL_FILE = 'model.mdl'
EVAUATION_DIR = 'evaluation'
RUN_META_FILE = 'run_meta.json'
CHECKPOINT_META_FILE = 'checkpoint_meta.json'
EVALUATION_FILE_EXT = '.ev'

def _save_evaluation(evaluation, checkpoint_path):
    _validate_evaluation(evaluation)
    # a file for each key in en extra evaluation directory and pickeled content
    evaluation_path = os.path.join(checkpoint_path, EVAUATION_DIR)
    if not os.path.exists(evaluation_path):
        os.makedirs(evaluation_path)
    for key in evaluation or {}:
        if evaluation[key]['type'] == 'image':
            cv2.imwrite(os.path.join(evaluation_path, key + '.png'), evaluation[key]['value'])
        with open(os.path.join(evaluation_path, key + EVALUATION_FILE_EXT), 'wb') as file:
            pickle.dump(evaluation[key], file)

def _load_evaluation(checkpoint_path, filter_fn=None):
    # a file for each key in en extra evaluation directory and pickeled content
    evaluation = {}
    # find the directory with all evaluations of the checkpoint
    evaluation_path = os.path.join(checkpoint_path, EVAUATION_DIR)
    if not os.path.exists(evaluation_path):
        return evaluation
    # find all files in it, not recursively
    evaluation_file_names = next(os.walk(evaluation_path))[2]
    # each of the files is one evaluation
    for evaluation_file_name in evaluation_file_names:
        # find the full path of each file
        evaluation_file_path = os.path.join(evaluation_path, evaluation_file_name)
        # only regard files with the proper file extention
        if evaluation_file_path.endswith(EVALUATION_FILE_EXT):
            # the key or name of the evaluation is just the file name, but without the file extention
            evaluation_key = evaluation_file_name[:-len(EVALUATION_FILE_EXT)]
            # read the evaluation from file
            with open(evaluation_file_path, 'rb') as file:
                evaluation_value = pickle.load(file)
                # filter by the custom filter function, if given
                if filter_fn is None or filter_fn(evaluation_key, evaluation_value):
                    # and insert it to results with the proper key
                    evaluation[evaluation_key] = evaluation_value
    _validate_evaluation(evaluation)
    return evaluation

def _validate_evaluation(evaluation):
    for key in evaluation or {}:
        if not 'type' in evaluation[key] or not 'value' in evaluation[key]:
            raise KeyError(f'Evaluation should have a type and a value, but f{key} had keys: {evaluation[key].keys()}.')

def _get_experience_scalars(experiment_logdir):
    # gather scalars over all past runs and checkpoints together with the metadata
    experience_list = []
    # loop through all runs
    for run_path in _get_runs(experiment_logdir):
        # read metaparams for the run
        run_meta_path = os.path.join(run_path, RUN_META_FILE)
        with open(run_meta_path, 'r') as file:
            run_meta = json.load(file)
        metaparams = run_meta['metaparams']

        # loop through all checkpoints in the run
        for checkpoint in _get_checkpoints(run_path):
            # open checkpoint meta file for training time and epoch
            checkpoint_path = os.path.join(run_path, checkpoint)
            checkpoint_meta_path = os.path.join(checkpoint_path, CHECKPOINT_META_FILE)
            if not os.path.exists(checkpoint_meta_path):
                print('no meta file at', checkpoint_meta_path)
                continue
            with open(checkpoint_meta_path, 'r') as file:
                checkpoint_meta = json.load(file)

            # but should also be the folder name
            if 'epoch' not in checkpoint_meta or 'time' not in checkpoint_meta:
                raise KeyError(f'Checkpoint meta must contain "epoch" and "time" of training: {checkpoint_meta.keys()}.')

            epoch_folder_name = int(os.path.basename(checkpoint_path))
            epoch_meta = int(checkpoint_meta['epoch'])
            if epoch_meta != epoch_folder_name:
                raise KeyError(f'Checkpoint folder name must be the epoch of the checkpoint and matching the epoch in the meta file, ' +
                               f'but they were not equal: Folder name was {epoch_folder_name}, meta file says {epoch_meta}.')

            # load only scalar evaluations from the checkpoint
            evaluation = _load_evaluation(checkpoint_path, lambda k,v: v['type'] == 'scalar')

            # make sure, keys ara unique among evaluation, run metaparams and checkpoint meta information (time and epoch)
            if not (len(evaluation.keys() | metaparams.keys() | checkpoint_meta.keys()) ==
                    len(evaluation.keys()) +  len(metaparams.keys()) + len(checkpoint_meta.keys())):
                raise KeyError(f'evaluation keys, metaparam keys must be disjoint and cannot be one of {checkpoint_meta.keys()}: {evaluation.keys()}, {metaparams.keys()}.')

            # merge everything into one directory
            experience_row = {}
            experience_row.update(metaparams)
            experience_row.update(checkpoint_meta)
            for key in evaluation:
                experience_row[key] = float(evaluation[key]['value'])#.cpu().detach().numpy()

            # and append it to the exper table
            experience_list.append(experience_row)

    # return experience ad pandas DataFrame
    experience = pd.DataFrame(experience_list)
    return experience

def _file_name_from_metaparams(metaparams):
    if metaparams is None:
        raise ValueError(f'Metaparams cannot be None.')
    def cvt(p):
        d = str(p)
        try:
            f = float(p)
            d = f'{p:.4e}'
            d_parts = d.split('e')
            d_base = d_parts[0]
            d_exp = int(d_parts[1])
            d_base = d_base.strip('0')
            d_base = d_base.rstrip('.')
            d = d_base
            if d_exp != 0:
                d = f'{d_base}e{d_exp}'
        except:
            pass
        return d

    plist = list(map(lambda mp: f'{mp}={cvt(metaparams[mp])}', metaparams))
    name = ','.join(plist)
    print('_file_name_from_metaparams', name)
    return name

def _get_checkpoints(run_path):
    # return a (as int) sorted list of all subdirectories of run_path, that names are integers (referring to its epoch)
    return list(map(str, sorted(map(int, filter(lambda s: s.isdigit(), next(os.walk(run_path))[1])))))

def _get_runs(experiment_logdir):
    # return all subdirectories of the experiment_logdir, that contain a run meta file
    return map(lambda rdf: rdf[0], filter(lambda rdf: RUN_META_FILE in rdf[2], os.walk(experiment_logdir)))
