#!/bin/bash

env_name=$1
model=$2
script=scripts/train_mpi.sh
models=([0]="sac" [1]="ddpg" [2]="ppo" [3]="mppi" [4]="rand")

if [ -z "$env_name" ]
then
	env_name="LunarLanderContinuous-v2"
fi

export env_name=$env_name
export nsteps=500000

if [ -z "$model" ]
then
	for i in `seq 0 3`
	do
		export model=${models[$i]}
		sbatch $script
		sleep 1
	done
else
	export model=$model
	sbatch $script
fi
