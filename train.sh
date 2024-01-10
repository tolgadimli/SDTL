#!/bin/bash 
# -------- Global Experiment Variables --------
DATE=`date +%Y-%m-%d`
model="resnet20"; dataset="CIFAR10"
num_nodes=1
node_rank=0
print_node_rank=$(($node_rank+1))

# -------- Other Experiment Variables --------
bs=128; mom=0.9; lr=1e-1; wd=1e-4
epochs=210
comm=4; c=0.1; p=0.1
seed=32

dist_opt="LSGD";
export_dir="$dist_opt-$dataset-$model-node-$print_node_rank-of-$num_nodes-$DATE"
exp_name="$dist_opt-comm=$comm-c=$c-p=$p-seed=$seed"
echo "STARTING $exp_name"
python codes/main.py --ip_address master.ip.address --node_gpus 4 --num_nodes $num_nodes --node_rank $node_rank \
    --dist_optim_name $dist_opt  --export_dir $export_dir  --experiment_name $exp_name  \
    --dataset $dataset --model $model  --bs $bs --lr $lr --wd $wd --mom $mom \
    --scheduler_gamma 0.1 --scheduler_milestones '120,180' --epochs $epochs --seed $seed \
    --comm_period $comm --c $c --p $p 
echo "FINISHING $exp_name"

dist_opt="DataParallel"
export_dir="$dist_opt-$dataset-$model-node-$print_node_rank-of-$num_nodes-$DATE"
exp_name="$dist_opt-$dataset-$model-seed=$seed"
python codes/main.py --ip_address master.ip.address  --node_gpus 4 --num_nodes $num_nodes --node_rank $node_rank \
    --dist_optim_name $dist_opt  --export_dir $export_dir  --experiment_name $exp_name  \
    --dataset $dataset --model $model --epochs $epochs --bs $bs --lr $lr --wd $wd --mom $mom \
    --scheduler_gamma 0.1 --scheduler_milestones '120,180' --epochs $epochs --seed $seed 
echo "FINISHING $exp_name"