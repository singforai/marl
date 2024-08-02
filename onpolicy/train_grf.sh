
#! /bin/bash

# command="python train_grf.py --seed {0} --experiment_name Xt_TiZero --group_name Xt_TiZero_2 --algorithm_name tizero --use_wandb --use_xt --use_linear_lr_decay --n_rollout_threads 10"

# args_gpu0="--num_gpu 0"
# # args_gpu1="--num_gpu 1"
# # args_gpu2="--num_gpu 2"
# # args_gpu3="--num_gpu 3"

# parallel -j 5 $command $args_gpu0 ::: 0  


command="python train_grf.py --seed {0} --experiment_name base_TiZero --group_name base_TiZero_2 --algorithm_name tizero --use_wandb --use_linear_lr_decay --n_rollout_threads 10"

args_gpu0="--num_gpu 1"
args_gpu1="--num_gpu 1"
args_gpu2="--num_gpu 2"
args_gpu3="--num_gpu 3"

parallel -j 5 $command $args_gpu0 ::: 0  

# command="python train_grf.py --seed {0} --experiment_name base_JRPO --group_name base_JRPO --algorithm_name jrpo --use_wandb"

# args_gpu0="--num_gpu 0"
# args_gpu1="--num_gpu 1"

# parallel -j 5 $command $args_gpu0 ::: 0  

# command="python train_grf.py --seed {0} --experiment_name base_MAPPO --group_name base_MAPPO --algorithm_name mappo --use_wandb --use_joint_action_loss"

# args_gpu0="--num_gpu 0"
# #args_gpu1="--num_gpu 1"

# parallel -j 5 $command $args_gpu0 ::: 0  