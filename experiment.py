import os
import time as pytime
import json
from . import logging
import torch
import gc
import numpy as np


# import like: from MLenv import Experiment



class Experiment():
    # call in sub class constructor
    def __init__(self, logdir=None):
        self.logdir = logdir or f'log_{self.__class__.__name__}'
        self.tb_logdir = os.path.join(self.logdir, 'tb')
        self.experience_fill_evaluations = True

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

    def get_training_conditions(self, metaparamspace, experience):
        """experience: pandas dataframe with 'epoch', 'time' as well as all metaparameters and all 'scalar' evaluations"""
        pass # -> training conditions:
        # {
        #   'metaparams': mp,
        #   'epochs': 1,
        #   'abs_epochs': 42,
        #   'time': 60,
        #   'abs_time': 3600
        # }

    # end ################################################


    def run(self):
        # get conditions for this run from implementation
        metaparamspace = self.get_metaparamspace()
        experience = logging._get_experience_scalars(self.logdir, self.experience_fill_evaluations)
        conditions = self.get_training_conditions(metaparamspace, experience)

        if conditions is None:
            raise ValueError(f'conditions, returned by "get_training_conditions" cannot be None.')
        print(conditions)
        metaparams = conditions.get('metaparams')
        if metaparams is None:
            raise ValueError(f'metaparams in training conditions cannot be None.')

        # create or open run
        run_name = logging._file_name_from_metaparams(metaparams)
        print(f'Run: {run_name}')
        if torch.cuda.is_available():
            print(f'Currently {torch.cuda.memory_allocated()} of GPU memory allocated.')
        run_path = os.path.join(self.logdir, run_name)
        run_meta_path = os.path.join(run_path, logging.RUN_META_FILE)

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
        checkpoints = logging._get_checkpoints(run_path)

        if len(checkpoints) == 0:
            checkpoint_path = os.path.join(run_path, '0')
            os.makedirs(checkpoint_path)

            # build model
            model = self.build_model(metaparams)

            # save model
            model_path = os.path.join(checkpoint_path, logging.MODEL_FILE)
            self.save_model(model, model_path)

            # save initial checkpoint meta
            checkpoint_meta = {'time': 0, 'epoch': 0}
            checkpoint_meta_path = os.path.join(checkpoint_path, logging.CHECKPOINT_META_FILE)
            with open(checkpoint_meta_path, 'w') as file:
                json.dump(checkpoint_meta, file)

            # initial evaluation
            evaluation = self.evaluate(model, 0, 0) or {}

            # save evaluation
            logging._save_evaluation(evaluation, checkpoint_path)
            logging._write_evaluation_to_experiment_tensorboard(self.tb_logdir, metaparams, evaluation, 0, 0)

            # instead of re-reading the checkpoints from file, just adjust the variable
            # to have the newly created one included
            checkpoints = ['0']

            del model

        # find latest checkpoint, TODO: find latest checkpoint with a valid model file in it
        latest_epoch = int(checkpoints[-1])
        print(f'Training upon checkpoint {latest_epoch}')
        epoch_run_start = latest_epoch
        time_run_start = None
        epoch = latest_epoch
        while True:
        #for epoch in range(latest_epoch, latest_epoch + epochs):
            checkpoint_path = os.path.join(run_path, str(epoch))

            # open checkpoint meta file
            checkpoint_meta_path = os.path.join(checkpoint_path, logging.CHECKPOINT_META_FILE)
            with open(checkpoint_meta_path, 'r') as file:
                checkpoint_meta = json.load(file)
            
            if time_run_start == None:
                time_run_start = checkpoint_meta['time']

            # load model
            model_path = os.path.join(checkpoint_path, logging.MODEL_FILE)
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
            new_model_path = os.path.join(new_checkpoint_path, logging.MODEL_FILE)
            self.save_model(model, new_model_path)

            # save checkpoint meta
            checkpoint_meta = {'time': checkpoint_meta['time'] + time, 'epoch': epoch+1}
            new_checkpoint_meta_path = os.path.join(new_checkpoint_path, logging.CHECKPOINT_META_FILE)
            with open(new_checkpoint_meta_path, 'w') as file:
                json.dump(checkpoint_meta, file)

            # evaluate
            test_evaluation = self.evaluate(model, checkpoint_meta['epoch'], checkpoint_meta['time']) or {}


            # save evaluation
            logging._save_evaluation(train_evaluation, new_checkpoint_path)
            logging._write_evaluation_to_experiment_tensorboard(self.tb_logdir, metaparams, train_evaluation,
                                                                checkpoint_meta['time'], checkpoint_meta['epoch'])
            logging._save_evaluation(test_evaluation, new_checkpoint_path)
            logging._write_evaluation_to_experiment_tensorboard(self.tb_logdir, metaparams, test_evaluation,
                                                                checkpoint_meta['time'], checkpoint_meta['epoch'])

            del model
            gc.collect()
            torch.cuda.empty_cache()

            epoch += 1

            # train another epoch, if the conditions are not met yet
            target_abs_epoch = conditions.get('abs_epochs', -np.inf)
            if checkpoint_meta['epoch'] < target_abs_epoch:
                print(f'Finished epoch {checkpoint_meta["epoch"]}, of desired {target_abs_epoch} (abs)')
                continue

            target_abs_epoch = epoch_run_start + conditions.get('epochs', -np.inf)
            if checkpoint_meta['epoch'] < target_abs_epoch:
                print(f'Finished epoch {checkpoint_meta["epoch"]}, of desired {target_abs_epoch} (rel)')
                continue

            target_abs_time = conditions.get('abs_time', -np.inf)
            if checkpoint_meta['time'] < target_abs_time:
                print(f'Trained {checkpoint_meta["time"]}, of desired {target_abs_time} seconds (abs)')
                continue

            target_abs_time = time_run_start + conditions.get('time', -np.inf)
            if checkpoint_meta['time'] < target_abs_time:
                print(f'Trained {checkpoint_meta["time"]}, of desired {target_abs_time} seconds (rel)')
                continue
            
            # otherwise finish training loop
            break
        pass