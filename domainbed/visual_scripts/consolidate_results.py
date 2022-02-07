import sys, os
import argparse
import json

from domainbed.visual_utils.consolidate import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Domain generalization consolidate results')
    parser.add_argument('--dataset', type=str, default="bearing")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--alg_results_dir', type=str, default=None)
    parser.add_argument('--jobname', type=str, default=None)
    # type of consolidation
    parser.add_argument('--best_model_type', type=str, default="seenval", 
        choices=("seenval","last","unseentest"))
    parser.add_argument('--aggregate', default=False, const=True, action='store_const',
        help='If true, return average accuracy across all target domains.')    
    parser.add_argument('--num_seed', type=int, default=3)    
    args = parser.parse_args()

    if args.alg_results_dir is None:
        if args.jobname is not None:
            args.alg_results_dir = os.path.join('./'+args.jobname+'_train_output',    
                args.dataset, args.algorithm)
        else:            
            args.alg_results_dir = os.path.join('./train_output',  
                args.dataset, args.algorithm)

    if args.dataset == 'bearing':
        num_envs = 8

    ###  domain generalization performance
    
    print('--------- dataset: {}, algorithm: {}, best model type: {} ---------'.format(
        args.dataset, args.algorithm, args.best_model_type))

    if not args.aggregate:
        # performance for single target domain        
        for d in range(num_envs):
            print("##### target domain {} #####".format(d))
            acc_d = get_perf_gen(args.alg_results_dir, [d], args.best_model_type, num_seed=args.num_seed)
    else:
        # average performance across target domains
        print("##### average across target domains #####")   
        acc_agg = get_perf_gen(args.alg_results_dir, range(num_envs), args.best_model_type, num_seed=args.num_seed)
