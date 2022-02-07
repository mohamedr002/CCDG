# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Run sweeps
"""

import argparse
import copy
import getpass
import hashlib
import json
import os
import random
import shutil
import time
import uuid

import numpy as np
import torch

from domainbed import TSdatasets as datasets
from domainbed import TShparams_registry as hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed import command_launchers

import tqdm
import shlex
import ipdb

class Job:
    NOT_LAUNCHED = 'Not launched'
    INCOMPLETE = 'Incomplete'
    DONE = 'Done'

    def __init__(self, train_args):
        args_str = json.dumps(train_args, sort_keys=True)
        args_hash = hashlib.md5(args_str.encode('utf-8')).hexdigest()
        self.jobname = '-'.join((train_args['jobname'], args_hash))

        self.train_args = copy.deepcopy(train_args)
        self.train_args['jobname'] = self.jobname
        command = ['python', '-m', 'domainbed.scripts.TStrain']
        for k, v in sorted(self.train_args.items()):
            if isinstance(v, list):
                v = ' '.join([str(v_) for v_ in v])
            elif isinstance(v, str):
                v = shlex.quote(v)
            command.append(f'--{k} {v}')
        self.command_str = ' '.join(command)

        self.output_dir = train_args['output_dir']
        if os.path.exists(os.path.join(self.output_dir, 'done')):
            self.state = Job.DONE
        elif os.path.exists(self.output_dir):
            self.state = Job.INCOMPLETE
        else:
            self.state = Job.NOT_LAUNCHED

    def __str__(self):
        job_info = (self.train_args['dataset'],
            self.train_args['algorithm'],
            self.train_args['test_envs'],
            self.train_args['hparams_seed'])
        return '{}: {} {}'.format(
            self.state,
            self.output_dir,
            job_info)

    @staticmethod
    def launch(jobs, launcher_fn):
        print('Launching...')
        jobs = jobs.copy()
        np.random.shuffle(jobs)
        print('Making job directories:')
        for job in tqdm.tqdm(jobs, leave=False):
            os.makedirs(job.output_dir, exist_ok=True)
        commands = [job.command_str for job in jobs]
        launcher_fn(commands)
        print(f'Launched {len(jobs)} jobs!')

    @staticmethod
    def delete(jobs):
        print('Deleting...')
        for job in jobs:
            shutil.rmtree(job.output_dir)
        print(f'Deleted {len(jobs)} jobs!')

# def all_test_env_combinations(n):
#     for i in range(n):
#         yield [i]
def all_test_env_combinations(n):
    # for i in range(n):
    for i in [4,5]:
        yield [i]

def make_args_list(n_trials, dataset_names, algorithms, n_hparams, steps,
    data_dir, hparams, jobname):
    args_list = []
    for trial_seed in range(n_trials):
        for dataset in dataset_names:
            for algorithm in algorithms:
                all_test_envs = all_test_env_combinations(
                    datasets.num_environments(dataset))
                for test_envs in all_test_envs:
                    for hparams_seed in range(n_hparams):
                        train_args = {}
                        train_args['dataset'] = dataset
                        train_args['algorithm'] = algorithm
                        train_args['test_envs'] = test_envs
                        train_args['hparams_seed'] = hparams_seed
                        train_args['data_dir'] = data_dir
                        train_args['trial_seed'] = trial_seed
                        train_args['seed'] = trial_seed
                        if steps is not None:
                            train_args['steps'] = steps
                        if hparams is not None:
                            train_args['hparams'] = hparams
                        # make output_dir and log_dir for job
                        test_envs_str = [str(i) for i in test_envs]
                        if jobname is not None:
                            output_dir = os.path.join('./'+jobname+'_train_output',
                                dataset, algorithm, '_'.join(test_envs_str), 
                                's'+str(trial_seed)+'_h'+str(hparams_seed)+'_t'+str(trial_seed))
                        else:            
                            output_dir = os.path.join('./train_output',  
                                dataset, algorithm, '_'.join(test_envs_str), 
                                's'+str(trial_seed)+'_h'+str(hparams_seed)+'_t'+str(trial_seed))

                        if jobname is not None:
                            log_dir = os.path.join('./'+jobname+'_runs',  
                                dataset, algorithm, '_'.join(test_envs_str), 
                                's'+str(trial_seed)+'_h'+str(hparams_seed)+'_t'+str(trial_seed))            
                        else:
                            log_dir = os.path.join('./runs',  
                                dataset, algorithm, '_'.join(test_envs_str), 
                                's'+str(trial_seed)+'_h'+str(hparams_seed)+'_t'+str(trial_seed))

                        train_args['output_dir'] = output_dir
                        train_args['log_dir'] = log_dir
                        train_args['jobname'] = jobname
                        args_list.append(train_args)
    return args_list

def ask_for_confirmation():
    response = input('Are you sure? (y/n) ')
    if not response.lower().strip()[:1] == "y":
        print('Nevermind!')
        exit(0)

DATASETS = [d for d in datasets.DATASETS if "Debug" not in d]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a sweep')
    parser.add_argument('command', choices=['launch', 'delete_incomplete'])
    parser.add_argument('--datasets', nargs='+', type=str, default=DATASETS)
    parser.add_argument('--algorithms', nargs='+', type=str, default=algorithms.ALGORITHMS)
    parser.add_argument('--n_hparams', type=int, default=20)
    parser.add_argument('--data_dir', type=str, default="../DG_Datasets")
    parser.add_argument('--output_dir', type=str, default=None)    
    parser.add_argument('--log_dir', type=str, default=None,
        help="Directory to save SummaryWriter logs")
    parser.add_argument('--jobname', type=str, default='sweep')
    parser.add_argument('--n_trials', type=int, default=3)
    parser.add_argument('--command_launcher', type=str, required=True)
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--hparams', type=str, default=None)
    parser.add_argument('--skip_confirmation', action='store_true')
    args = parser.parse_args()

    # individual output_dir and log_dir will be assigned for each job
    assert args.output_dir is None
    assert args.log_dir is None

    args_list = make_args_list(
        n_trials=args.n_trials,
        dataset_names=args.datasets,
        algorithms=args.algorithms,
        n_hparams=args.n_hparams,
        steps=args.steps,
        data_dir=args.data_dir,
        hparams=args.hparams,
        jobname=args.jobname
    )

    jobs = [Job(train_args) for train_args in args_list]

    for job in jobs:
        print(job)
    print("{} jobs: {} done, {} incomplete, {} not launched.".format(
        len(jobs),
        len([j for j in jobs if j.state == Job.DONE]),
        len([j for j in jobs if j.state == Job.INCOMPLETE]),
        len([j for j in jobs if j.state == Job.NOT_LAUNCHED]))
    )

    if args.command == 'launch':
        to_launch = [j for j in jobs if j.state == Job.NOT_LAUNCHED]
        print(f'About to launch {len(to_launch)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        launcher_fn = command_launchers.REGISTRY[args.command_launcher]
        Job.launch(to_launch, launcher_fn)

    elif args.command == 'delete_incomplete':
        to_delete = [j for j in jobs if j.state == Job.INCOMPLETE]
        print(f'About to delete {len(to_delete)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        Job.delete(to_delete)