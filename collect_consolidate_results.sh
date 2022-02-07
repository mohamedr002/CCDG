#!/bin/bash

declare -a alg_notune_list=("ERM_Contrastive" );
declare -i env;

#bearing dataset
# Collect  
for env in {0..7}; do
	python -m domainbed.scripts.collect_consolidate_results --dataset bearing --algorithm ERM_Contrastive--test_envs $env --jobname sweep
done
# Consolidate 
python -m domainbed.visual_scripts.consolidate_results --dataset bearing --algorithm ERM_Contrastive --num_seed 3  --jobname sweeep --best_model_type last

# Paderborn dataset
# Collect  
for env in {0..5}; do
	python -m domainbed.scripts.collect_consolidate_results --dataset paderborn --algorithm ERM_Contrastive--test_envs $env --jobname sweep
done
# Consolidate 
python -m domainbed.visual_scripts.consolidate_results --dataset paderborn --algorithm $alg --num_seed 3 --aggregate --jobname sweeep --best_model_type last
