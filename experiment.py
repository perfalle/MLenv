import os
import time as pytime
import pickle, json
import pandas as pd

# import like: from MLenv import Experiment

MODEL_FILE = 'model.mdl'
EVAUATION_DIR = 'evaluation'
RUN_META_FILE = 'run_meta.json'
CHECKPOINT_META_FILE = 'checkpoint_meta.json'
EVALUATION_FILE_EXT = '.ev'

class Experiment():
    # call in sub class constructor
    def __init__(self, logdir=None):
        self.logdir = logdir or f'log_{self.__class__.__name__}'

    # override these methods ##############################
    def get_metaparamspace(self):
        return {} # ...

    def build_model(self, metaparams): # metaparams={'metaparam': value}
        pass # -> model(s), e.g. dict: {'encoder': enc, 'decoder': dec}

    def save_model(self, model, path):
        pass

    def load_model(self, path, metaparams):
        pass

    def train(self, model, metaparams):
        pass # -> model(s)

    def evaluate(self, model, epoch, time): # return value of train
        pass # -> evaluation(s), dict: {'metric name': {'type': scalar, 'value': value}}

    def get_metaparams(self, metaparamspace, experience):
        """experience: pandas dataframe with 'epoch', 'time' as well as all metaparameters, all 'scalar' evaluations"""
        pass # -> training conditions: metaparams[, epochs=1]

    # end ################################################


    def run(self):
        # get conditions for this run from implementation
        metaparamspace = self.get_metaparamspace()
        experience = self._get_experience_scalars()
        metaparams_and_epochs = self.get_metaparams(metaparamspace, experience)
        if metaparams_and_epochs is None:
            raise ValueError(f'Metaparams, returned by "get_metaparams" cannot be None.')
        epochs = 1
        if type(metaparams_and_epochs) == tuple:
            metaparams, epochs = metaparams_and_epochs

        # create or open run
        run_name = self._file_name_from_metaparams(metaparams)
        print(f'Run: {run_name}')
        run_path = os.path.join(self.logdir, run_name)
        run_meta_path = os.path.join(run_path, RUN_META_FILE)

        # write or read run meta file
        if not os.path.exists(run_path):
            os.makedirs(run_path)
            with open(run_meta_path, 'w') as file:
                json.dump({'metaparams': metaparams}, file)
        else:
            with open(run_meta_path, 'r') as file:
                meta = json.load(file)
                if 'metaparams' not in meta or metaparams != meta['metaparams']:
                    raise KeyError('Metaparams of this run did not match expectation (from name) ' +
                                   f'{metaparams} but meta file was {meta}.')

        # load checkpoints of the run, it's gonna be just the folder names
        checkpoints = self._get_checkpoints(run_path)

        if len(checkpoints) == 0:
            checkpoint_path = os.path.join(run_path, '0')
            os.makedirs(checkpoint_path)

            # build model
            model = self.build_model(metaparams)

            # save model
            model_path = os.path.join(checkpoint_path, MODEL_FILE)
            self.save_model(model, model_path)

            # save initial checkpoint meta
            checkpoint_meta = {'time': 0, 'epoch': 0}
            checkpoint_meta_path = os.path.join(checkpoint_path, CHECKPOINT_META_FILE)
            with open(checkpoint_meta_path, 'w') as file:
                json.dump(checkpoint_meta, file)

            # initial evaluation
            evaluation = self.evaluate(model, 0, 0)

            # save evaluation
            self._save_evaluation(evaluation, checkpoint_path)

            # instead of re-reading the checkpoints from file, just adjust the variable
            # to have the newly created one included
            checkpoints = ['0']

        # find latest checkpoint, TODO: find latest checkpoint with a valid model file in it
        latest_epoch = int(checkpoints[-1])
        print(f'Training upon checkpoint {latest_epoch}')
        for epoch in range(latest_epoch, latest_epoch + epochs):
            checkpoint_path = os.path.join(run_path, str(epoch))

            # open checkpoint meta file
            checkpoint_meta_path = os.path.join(checkpoint_path, CHECKPOINT_META_FILE)
            with open(checkpoint_meta_path, 'r') as file:
                checkpoint_meta = json.load(file)

            # load model
            model_path = os.path.join(checkpoint_path, MODEL_FILE)
            model = self.load_model(model_path, metaparams)

            # train model
            time_start = pytime.time()
            train_result = self.train(model, metaparams)
            time_end = pytime.time()
            time = time_end - time_start

            if type(train_result) == tuple:
                model, train_evaluation = train_result
            else:
                model, train_evaluation = train_result, {}

            # save new checkpoint
            new_checkpoint_path = os.path.join(run_path, str(epoch+1))
            if not os.path.exists(new_checkpoint_path):
                os.makedirs(new_checkpoint_path)

            # save model
            new_model_path = os.path.join(new_checkpoint_path, MODEL_FILE)
            self.save_model(model, new_model_path)

            # save checkpoint meta
            checkpoint_meta = {'time': checkpoint_meta['time'] + time, 'epoch': epoch+1}
            new_checkpoint_meta_path = os.path.join(new_checkpoint_path, CHECKPOINT_META_FILE)
            with open(new_checkpoint_meta_path, 'w') as file:
                json.dump(checkpoint_meta, file)

            # evaluate
            evaluation = self.evaluate(model, epoch, time)

            # save evaluation
            self._save_evaluation(evaluation, new_checkpoint_path)

    def _save_evaluation(self, evaluation, checkpoint_path):
        self._validate_evaluation(evaluation)
        # a file for each key in en extra evaluation directory and pickeled content
        evaluation_path = os.path.join(checkpoint_path, EVAUATION_DIR)
        if not os.path.exists(evaluation_path):
            os.makedirs(evaluation_path)
        for key in evaluation or {}:
            with open(os.path.join(evaluation_path, key + EVALUATION_FILE_EXT), 'wb') as file:
                pickle.dump(evaluation[key], file)

    def _load_evaluation(self, checkpoint_path, filter_fn=None):
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
        self._validate_evaluation(evaluation)
        return evaluation

    def _validate_evaluation(self, evaluation):
        for key in evaluation or {}:
            if not 'type' in evaluation[key] or not 'value' in evaluation[key]:
                raise KeyError(f'Evaluation should have a type and a value, but f{key} had keys: {evaluation[key].keys()}.')

    def _get_experience_scalars(self):
        # gather scalars over all past runs and checkpoints together with the metadata
        experience_list = []
        # loop through all runs
        for run_path in self._get_runs():
            # read metaparams for the run
            run_meta_path = os.path.join(run_path, RUN_META_FILE)
            with open(run_meta_path, 'r') as file:
                run_meta = json.load(file)
            metaparams = run_meta['metaparams']

            # loop through all checkpoints in the run
            for checkpoint in self._get_checkpoints(run_path):
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
                evaluation = self._load_evaluation(checkpoint_path, lambda k,v: v['type'] == 'scalar')

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

    def _file_name_from_metaparams(self, metaparams):
        if metaparams is None:
            raise ValueError(f'Metaparams cannot be None.')
        def cvt(p):
            d = str(p)
            if type(p) == float:
                d = f'{p:.5e}'
                d_parts = d.split('e')
                d_base = d_parts[0]
                d_exp = int(d_parts[1])
                d = d_base
                if d_exp != 0:
                    d = f'{d_base}e{d_exp}'
            return d

        plist = list(map(lambda mp: f'{mp}={cvt(metaparams[mp])}', metaparams))
        name = ','.join(plist)
        return name

    def _get_checkpoints(self, run_path):
        # return a (as int) sorted list of all subdirectories of run_path, that names are integers (referring to its epoch)
        return list(map(str, sorted(map(int, filter(lambda s: s.isdigit(), next(os.walk(run_path))[1])))))

    def _get_runs(self):
        # return all subdirectories of the experiments self.logdir, that contain a run meta file
        return map(lambda rdf: rdf[0], filter(lambda rdf: RUN_META_FILE in rdf[2], os.walk(self.logdir)))
