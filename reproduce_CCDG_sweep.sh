#!/bin/bash

# cd /TSDomainBed/

declare -a dataset_list=("bearing" "paderborn");

for data in "${dataset_list[@]}"; do 
	CUDA_VISIBLE_DEVICES=0	python -m domainbed.scripts.TSsweep_batch delete_incomplete --dataset data --command_launcher local --algorithms ERM_Contrastive --n_hparams 20 --jobname Sweep_CCDG --n_trials 3 --skip_confirmation
	CUDA_VISIBLE_DEVICES=0	python -m domainbed.scripts.TSsweep_batch launch --dataset data  --command_launcher local --algorithms ERM_Contrastive --n_hparams 20 --jobname Sweep_CCDG --n_trials 3 --skip_confirmation
done
# done 
