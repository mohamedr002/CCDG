#!/bin/bash

declare -a alg_list=( "ERM_Contrastive" );
declare -i seed;
declare -i env;

# bearing
for env in {0..2}; do
	for seed in {0..7}; do
		python -m domainbed.scripts.TStrain --dataset bearing --algorithm $alg --test_envs $env --seed $seed
	done
done

# paderborn
for env in {0..2}; do
	for seed in {0..5}; do
		python -m domainbed.scripts.TStrain --dataset paderborn --algorithm $alg --test_envs $env --seed $seed
	done
done

