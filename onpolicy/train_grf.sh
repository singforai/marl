
#! /bin/bash

# command="python train_grf.py --seed 0 --experiment_name MAT --group_name MAT --algorithm_name mat --use_wandb --use_linear_lr_decay --n_rollout_threads 10 --num_gpu 1"


# $command 

command="python train_grf.py --seed {0} --experiment_name tizero --group_name self_play_Tizero --algorithm_name tizero --use_wandb --use_linear_lr_decay --n_rollout_threads 10"

args_gpu0="--num_gpu 0"
args_gpu1="--num_gpu 1"


parallel -j 4 $command $args_gpu0 ::: 0 2 & 

parallel -j 4 $command $args_gpu1 ::: 1 3 

wait