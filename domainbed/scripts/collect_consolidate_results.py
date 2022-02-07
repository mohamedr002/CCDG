import argparse
import collections
import json
import os
import random
import sys
import time
import uuid
import pickle

import numpy as np


def collect_res(consolidate_dir, best_model_type):
    if best_model_type == 'seenval':
        collect_res_seenval(consolidate_dir)
    elif best_model_type == 'last':
        collect_res_last(consolidate_dir)
    elif best_model_type == 'unseentest':
        collect_res_unseentest(consolidate_dir)

# collect results for best_model_type = seenval
def collect_res_seenval(consolidate_dir):
    # find target domains
    test_envs = consolidate_dir.split('/')[-1].split('_')
    test_envs_key = ['env{}_in_acc'.format(i) for i in test_envs]
    test_envs_key_out = ['env{}_out_acc'.format(i) for i in test_envs] # this acc is not used

    # group runs (hyperparameter) by seed
    subdir = [d for d in os.listdir(consolidate_dir) if os.path.isdir(os.path.join(consolidate_dir, d))]
    subdir_split = [d.split('_') for d in subdir]
    subdir_prefix = np.unique([d[0] for d in subdir_split])
    subdir_group = [['_'.join(d) for d in subdir_split if p in d] for p in subdir_prefix]

    # go through each group of runs
    runs_dict = {} # save best result for each run
    for grp in subdir_group:
        grp_split = grp[0].split('_')
        grp_name = '_'.join([grp_split[0], grp_split[2]])
        grp_dict = {}

        savename = 'brief_test_accuracy-' + grp_name + '.txt'
        save_path = os.path.join(consolidate_dir, savename)
        acc_list = []
        record_best_list = []
        # go through each run (hyperparameter)
        for d in grp:
            results_path = os.path.join(consolidate_dir, d, 'results.jsonl')

            # access results file
            records = []
            try:
                with open(results_path, "r") as f:
                    for line in f:
                        records.append(json.loads(line[:-1]))
            except IOError:
                pass

            # extract entry corresponding to best model
            keys = records[0].keys()
            train_envs_key = [k for k in keys if ('out' in k) and not (k in test_envs_key_out)]
            num_envs = len(train_envs_key) + len(test_envs_key)
            
            acc_seenval = []
            for record in records:
                acc_seenval.append(sum([record[k] for k in train_envs_key]))
            idx_best_d = np.argmax(acc_seenval)
            acc_best_d = acc_seenval[idx_best_d]
            record_best_d = records[idx_best_d]

            grp_dict[d] = {'target_acc':sum([record_best_d[k] for k in test_envs_key]),
                'seed': record_best_d['args']['seed'], 'hparams_seed': record_best_d['args']['hparams_seed'],
                'trial_seed': record_best_d['args']['trial_seed'], 'hparams': record_best_d['hparams'],
                'record': record_best_d}

            acc_list.append(acc_best_d)
            record_best_list.append(record_best_d)

        # save consolidate results
        idx_best = np.argmax(acc_list)
        record_best = record_best_list[idx_best]
        f = open(save_path, mode='w')
        f.write(','.join(['overall_acc', 'domain', 'best_model_type', 'best_hyperparam']))
        f.write('\n')
        for d in range(num_envs):
            if d in test_envs:
                acc = record_best['env{}_in_acc'.format(d)]
            else:
                acc = record_best['env{}_out_acc'.format(d)]
            f.write(','.join([str(acc), str(d), 'seenval', str(record_best['args']['hparams_seed'])]))
            f.write('\n')                
        f.close()

        runs_dict[grp_name] = grp_dict

    # save runs_dict
    runs_savename = 'runs_seenval.p'
    runs_save_path = os.path.join(consolidate_dir, runs_savename)
    with open(runs_save_path, 'wb') as handle:
        pickle.dump(runs_dict, handle)

# collect results for best_model_type = last
def collect_res_last(consolidate_dir):
    # find target domains
    test_envs = consolidate_dir.split('/')[-1].split('_')
    test_envs_key = ['env{}_in_acc'.format(i) for i in test_envs]
    test_envs_key_out = ['env{}_out_acc'.format(i) for i in test_envs] # this acc is not used

    # group runs (hyperparameter) by seed
    subdir = [d for d in os.listdir(consolidate_dir) if os.path.isdir(os.path.join(consolidate_dir, d))]
    subdir_split = [d.split('_') for d in subdir]
    subdir_prefix = np.unique([d[0] for d in subdir_split])
    subdir_group = [['_'.join(d) for d in subdir_split if p in d] for p in subdir_prefix]

    # go through each group of runs
    runs_dict = {} # save best result for each run
    for grp in subdir_group:
        grp_split = grp[0].split('_')
        grp_name = '_'.join([grp_split[0], grp_split[2]])
        grp_dict = {}

        savename = 'brief_test_accuracy-' + grp_name + '.txt'
        save_path = os.path.join(consolidate_dir, savename)
        acc_list = []
        record_best_list = []
        # go through each run (hyperparameter)
        for d in grp:
            results_path = os.path.join(consolidate_dir, d, 'results.jsonl')

            # access results file
            records = []
            try:
                with open(results_path, "r") as f:
                    for line in f:
                        records.append(json.loads(line[:-1]))
            except IOError:
                pass

            # extract entry corresponding to best model
            keys = records[0].keys()
            train_envs_key = [k for k in keys if ('out' in k) and not (k in test_envs_key_out)]
            num_envs = len(train_envs_key) + len(test_envs_key)

            record_best_d = records[-1]
            acc_best_d = sum([record_best_d[k] for k in test_envs_key])

            grp_dict[d] = {'target_acc':sum([record_best_d[k] for k in test_envs_key]),
                'seed': record_best_d['args']['seed'], 'hparams_seed': record_best_d['args']['hparams_seed'],
                'trial_seed': record_best_d['args']['trial_seed'], 'hparams': record_best_d['hparams'],
                'record': record_best_d}

            acc_list.append(acc_best_d)
            record_best_list.append(record_best_d)            

        # save consolidate results
        idx_best = np.argmax(acc_list)
        record_best = record_best_list[idx_best]
        if not os.path.exists(save_path):
            f = open(save_path, mode='a+')
            f.write(','.join(['overall_acc', 'domain', 'best_model_type', 'best_hyperparam']))
            f.write('\n')
        else:
            f = open(save_path, mode='a+')
        for d in range(num_envs):
            if d in test_envs:
                acc = record_best['env{}_in_acc'.format(d)]
            else:
                acc = record_best['env{}_out_acc'.format(d)]
            f.write(','.join([str(acc), str(d), 'last', str(record_best['args']['hparams_seed'])]))
            f.write('\n')                
        f.close()

        runs_dict[grp_name] = grp_dict

    # save runs_dict
    runs_savename = 'runs_last.p'
    runs_save_path = os.path.join(consolidate_dir, runs_savename)
    with open(runs_save_path, 'wb') as handle:
        pickle.dump(runs_dict, handle)

# collect results for best_model_type = unseentest
def collect_res_unseentest(consolidate_dir):
    # find target domains
    test_envs = consolidate_dir.split('/')[-1].split('_')
    test_envs_key = ['env{}_in_acc'.format(i) for i in test_envs]
    test_envs_key_out = ['env{}_out_acc'.format(i) for i in test_envs] # this acc is not used

    # group runs (hyperparameter) by seed
    subdir = [d for d in os.listdir(consolidate_dir) if os.path.isdir(os.path.join(consolidate_dir, d))]
    subdir_split = [d.split('_') for d in subdir]
    subdir_prefix = np.unique([d[0] for d in subdir_split])
    subdir_group = [['_'.join(d) for d in subdir_split if p in d] for p in subdir_prefix]

    # go through each group of runs
    runs_dict = {} # save best result for each run
    for grp in subdir_group:
        grp_split = grp[0].split('_')
        grp_name = '_'.join([grp_split[0], grp_split[2]])
        grp_dict = {}

        savename = 'brief_test_accuracy-' + grp_name + '.txt'
        save_path = os.path.join(consolidate_dir, savename)
        acc_list = []
        record_best_list = []
        # go through each run (hyperparameter)
        for d in grp:
            results_path = os.path.join(consolidate_dir, d, 'results.jsonl')

            # access results file
            records = []
            try:
                with open(results_path, "r") as f:
                    for line in f:
                        records.append(json.loads(line[:-1]))
            except IOError:
                pass

            # extract entry corresponding to best model
            keys = records[0].keys()
            train_envs_key = [k for k in keys if ('out' in k) and not (k in test_envs_key_out)]
            num_envs = len(train_envs_key) + len(test_envs_key)
            
            acc_unseentest = []
            for record in records:
                acc_unseentest.append(sum([record[k] for k in test_envs_key]))
            idx_best_d = np.argmax(acc_unseentest)
            acc_best_d = acc_unseentest[idx_best_d]            
            record_best_d = records[idx_best_d]

            grp_dict[d] = {'target_acc':sum([record_best_d[k] for k in test_envs_key]),
                'seed': record_best_d['args']['seed'], 'hparams_seed': record_best_d['args']['hparams_seed'],
                'trial_seed': record_best_d['args']['trial_seed'], 'hparams': record_best_d['hparams'],
                'record': record_best_d}

            acc_list.append(acc_best_d)
            record_best_list.append(record_best_d)    

        # save consolidate results
        idx_best = np.argmax(acc_list)
        record_best = record_best_list[idx_best]
        if not os.path.exists(save_path):
            f = open(save_path, mode='a+')
            f.write(','.join(['overall_acc', 'domain', 'best_model_type', 'best_hyperparam']))
            f.write('\n')
        else:
            f = open(save_path, mode='a+')
        for d in range(num_envs):
            if d in test_envs:
                acc = record_best['env{}_in_acc'.format(d)]
            else:
                acc = record_best['env{}_out_acc'.format(d)]
            f.write(','.join([str(acc), str(d), 'unseentest', str(record_best['args']['hparams_seed'])]))
            f.write('\n')                
        f.close()

        runs_dict[grp_name] = grp_dict

    # save runs_dict
    runs_savename = 'runs_unseentest.p'
    runs_save_path = os.path.join(consolidate_dir, runs_savename)
    with open(runs_save_path, 'wb') as handle:
        pickle.dump(runs_dict, handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization collect results')
    parser.add_argument('--dataset', type=str, default="bearing")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--consolidate_dir', type=str, default=None)
    parser.add_argument('--jobname', type=str, default=None)
    args = parser.parse_args()
    for j in [0,1,2,3,4,5,6,7]:
        args.test_envs = [j]
        if args.consolidate_dir is None:
            test_envs_str = [str(i) for i in args.test_envs]
            if args.jobname is not None:
                args.consolidate_dir = os.path.join('./'+args.jobname+'_train_output',
                    args.dataset, args.algorithm, '_'.join(test_envs_str))
            else:
                args.consolidate_dir = os.path.join('./train_output',
                    args.dataset, args.algorithm, '_'.join(test_envs_str))

        print('Processing algorithm {} for dataset {} env {}...'.format(args.algorithm, args.dataset, test_envs_str))

        # collect results for best_model_type = seenval
        collect_res(args.consolidate_dir, best_model_type = 'seenval')

        # collect results for best_model_type = last
        collect_res(args.consolidate_dir, best_model_type = 'last')

        # collect results for best_model_type = unseentest
        collect_res(args.consolidate_dir, best_model_type = 'unseentest')