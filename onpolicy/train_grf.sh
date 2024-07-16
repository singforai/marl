
#! /bin/bash


#=============================== base TiZero ==========================
command="python train_grf.py --seed {0} --experiment_name base_TiZero --group_name base_TiZero --algorithm_name tizero --use_wandb  --n_rollout_threads 10"

args_gpu0="--num_gpu 0"
args_gpu1="--num_gpu 1"
args_gpu2="--num_gpu 2"
args_gpu3="--num_gpu 3"

parallel -j 5 $command $args_gpu0 ::: 0  &
parallel -j 5 $command $args_gpu1 ::: 1 2&
parallel -j 5 $command $args_gpu2 ::: 3  &
parallel -j 5 $command $args_gpu3 ::: 4 
wait

#=============================== XT TiZero ==========================
# command="python train_grf.py --seed {0} --experiment_name XT_TiZero --group_name XT_TiZero --algorithm_name tizero --use_wandb  --n_rollout_threads 10 --use_xt"

# args_gpu0="--num_gpu 0"
# args_gpu1="--num_gpu 1"
# args_gpu2="--num_gpu 2"
# args_gpu3="--num_gpu 3"

# parallel -j 5 $command $args_gpu0 ::: 0  &
# parallel -j 5 $command $args_gpu1 ::: 1 2&
# parallel -j 5 $command $args_gpu2 ::: 3  &
# parallel -j 5 $command $args_gpu3 ::: 4 
# wait