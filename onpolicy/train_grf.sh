
#! /bin/bash

command="python train_grf.py --seed 0 --experiment_name MAT --group_name MAT --algorithm_name mat --use_wandb --use_linear_lr_decay --n_rollout_threads 10 --use_available_actions --num_gpu 1"


$command 

# command="python train_grf.py --seed 0 --experiment_name tizero --group_name tizero_mask_action --algorithm_name tizero --use_wandb --use_linear_lr_decay --n_rollout_threads 10 --use_available_actions --num_gpu 6"


# $command 