#!/bin/bash

base=$1
env_name="CarRacing-v1"
nsteps=1000000
models=([0]="sac" [1]="ddpg" [2]="ppo" [3]="rand")

for i in 0 1 2 3
do
	model=${models[$i]}
	port=$(($base+1000+$(($i*2000))))
	bash scripts/train_tcp.sh $env_name $model $nsteps $port
done

