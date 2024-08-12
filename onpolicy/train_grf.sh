
#! /bin/bash

# command="python train_grf.py --seed {0} --experiment_name mappo --group_name mappo --algorithm_name mappo --use_wandb --use_linear_lr_decay --n_rollout_threads 10 --layer_N 1"

# args_gpu0="--num_gpu 0"
# args_gpu1="--num_gpu 1"
# args_gpu2="--num_gpu 2"
# args_gpu3="--num_gpu 3"

# parallel -j 5 $command $args_gpu0 ::: 0    

# command="python train_grf.py --seed 0 --experiment_name tizero --group_name tizero_non_mask_action --algorithm_name tizero --use_wandb --use_linear_lr_decay --n_rollout_threads 10 --num_gpu 6"


# $command 

command="python train_grf.py --seed 0 --experiment_name tizero --group_name tizero_mask_action --algorithm_name tizero --use_wandb --use_linear_lr_decay --n_rollout_threads 10 --use_available_actions --num_gpu 6"


$command 