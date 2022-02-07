import argparse
import collections
import json
import os
import random
import sys
import time
import uuid

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from domainbed import TSdatasets as datasets
from domainbed import TShparams_registry as hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str, default="../DG_Datasets")
    parser.add_argument('--dataset', type=str, default="paderborn")
    parser.add_argument('--algorithm', type=str, default="ERM_Contrastive")
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')   
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default=None,
        help="Directory to save SummaryWriter logs")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--jobname', type=str, default=None)
    parser.add_argument('--additional_log', default=False, const=True, action='store_const')
    args = parser.parse_args()

    if args.output_dir is None:
        test_envs_str = [str(i) for i in args.test_envs]
        if args.jobname is not None:
            args.output_dir = os.path.join('./'+args.jobname+'_train_output',  
                args.dataset, args.algorithm, '_'.join(test_envs_str), 
                's'+str(args.seed)+'_h'+str(args.hparams_seed)+'_t'+str(args.trial_seed))
            consolidate_dir = os.path.join('./'+args.jobname+'_train_output', 
                args.dataset, args.algorithm, '_'.join(test_envs_str))
        else:            
            args.output_dir = os.path.join('./train_output',  
                args.dataset, args.algorithm, '_'.join(test_envs_str), 
                's'+str(args.seed)+'_h'+str(args.hparams_seed)+'_t'+str(args.trial_seed))
            consolidate_dir = os.path.join('./train_output',  
                args.dataset, args.algorithm, '_'.join(test_envs_str))
    configs_tracked = '-s'+str(args.seed)+'_h'+str(args.hparams_seed)+'_t'+str(args.trial_seed)

    if args.log_dir is None:
        test_envs_str = [str(i) for i in args.test_envs]
        if args.jobname is not None:
            args.log_dir = os.path.join('./runs_'+args.jobname,  
                args.dataset, args.algorithm, '_'.join(test_envs_str), 
                's'+str(args.seed)+'_h'+str(args.hparams_seed)+'_t'+str(args.trial_seed))            
        else:
            args.log_dir = os.path.join('./runs',  
                args.dataset, args.algorithm, '_'.join(test_envs_str), 
                's'+str(args.seed)+'_h'+str(args.hparams_seed)+'_t'+str(args.trial_seed))

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    writer = SummaryWriter(log_dir=args.log_dir)

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.
    in_splits = []
    out_splits = []
    for env_i, env in enumerate(dataset):
        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))
        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
        else:
            in_weights, out_weights = None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))

    # for tensorboard logs
    # note test_target_in is the final test accuracy reported 
    train_source = ['env{}_in_acc'.format(i) for i in range(len(in_splits)) 
        if (i not in args.test_envs)]
    test_source = ['env{}_out_acc'.format(i) for i in range(len(out_splits)) 
        if (i not in args.test_envs)]
    test_target_in = ['env{}_in_acc'.format(i) for i in range(len(in_splits)) 
        if (i in args.test_envs)] 
    test_target_out = ['env{}_out_acc'.format(i) for i in range(len(out_splits)) 
        if (i in args.test_envs)] 

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams, n_steps)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    train_minibatches_iterator = zip(*train_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])

    last_results_keys = None
    for step in range(start_step, n_steps):
        step_start_time = time.time()
        minibatches_device = [(x.to(device), y.to(device))
            for x,y in next(train_minibatches_iterator)]
        step_vals = algorithm.update(minibatches_device)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if step % checkpoint_freq == 0:
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for name, loader, weights in evals:
                acc = misc.accuracy(algorithm, loader, weights, device)
                results[name+'_acc'] = acc

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                colwidth=12)

            results.update({
                'hparams': hparams,
                'args': vars(args)    
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")
            # logs for tensorboard
            writer.add_scalar('Loss/train', results['loss'], step)
            train_source_acc = np.mean([results[i] for i in train_source])
            test_source_acc = np.mean([results[i] for i in test_source])
            test_target_in_acc = np.mean([results[i] for i in test_target_in])
            test_target_out_acc = np.mean([results[i] for i in test_target_out])
            writer.add_scalar('Accuracy/train_source', train_source_acc, step)
            writer.add_scalar('Accuracy/test_source', test_source_acc, step)
            writer.add_scalar('Accuracy/test_target_in', test_target_in_acc, step)
            writer.add_scalar('Accuracy/test_target_out', test_target_out_acc, step)

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

    if not args.skip_model_save:
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.cpu().state_dict()
        }

        torch.save(save_dict, os.path.join(args.output_dir, "model.pkl"))

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')

    if args.additional_log:
        # save consolidate results
        best_model_type = 'last'
        save_path = os.path.join(consolidate_dir, 
            'test_accuracy_' + best_model_type + configs_tracked + '.txt')
        if not os.path.exists(save_path):
            f = open(save_path, mode='a+')
            col_num_samp_per_cls = ','.join(['num_samp_cls' + str(c) for c in range(dataset.num_classes)])
            col_accuracy_per_cls = ','.join(['acc_cls' + str(c) for c in range(dataset.num_classes)])
            f.write(
                ','.join([col_num_samp_per_cls, col_accuracy_per_cls, 'overall_acc', 'domain']))
            f.write('\n')
        else:
            f = open(save_path, mode='a+')    
        # number of samples per class, accuracy per class
        accuracy_list = []
        accuracy_per_class_list = []
        num_samp_per_class_list = []
        evals = zip(eval_loader_names, eval_loaders, eval_weights)
        for name, loader, weights in evals:
            env = int(name.split('_')[0].replace('env', ''))
            train_bool = (not (env in args.test_envs)) and ('out' in name)        
            test_bool = (env in args.test_envs) and ('in' in name)
            if train_bool or test_bool:
                acc = misc.accuracy(algorithm, loader, weights, device)
                acc_per_class, num_samp_per_cls = misc.accuracy_per_class(algorithm, 
                    loader, weights, device, dataset.num_classes)
                accuracy_list.append(acc)
                accuracy_per_class_list.append(acc_per_class)
                num_samp_per_class_list.append(num_samp_per_cls)
        # writing results
        for j in range(len(accuracy_list)):
            num_samp_per_class_j = [str(n) for n in num_samp_per_class_list[j]]
            accuracy_per_class_j = [str(a) for a in accuracy_per_class_list[j]]
            accuracy_j = str(accuracy_list[j])
            f.write(','.join(
                num_samp_per_class_j + accuracy_per_class_j + [accuracy_j] + [str(j)]))
            f.write('\n')
        f.close()
