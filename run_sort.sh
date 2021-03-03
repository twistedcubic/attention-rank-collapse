#!/bin/bash

#export PYTHONPATH=/data/users/yihdong/transformers/src:$PYTHONPATH
#export PYTHONPATH=/usr/local/google/home/yihed/transformers/src:$PYTHONPATH
cur_path=$(realpath $0)
cur_dir=$(dirname $cur_path)
export PYTHONPATH=$cur_dir:$PYTHONPATH

python sort.py --width 8 --depth 1 --hidden_dim 32 --seed 2 --num_labels 15 --n_epochs 8
python sort.py --width 4 --depth 2 --hidden_dim 32 --seed 2 --num_labels 15 --n_epochs 8
python sort.py --width 2 --depth 4 --hidden_dim 32 --seed 2 --num_labels 15 --n_epochs 8
python sort.py --width 1 --depth 8 --hidden_dim 32 --seed 2 --num_labels 15 --n_epochs 8

#python sort.py --width 1 --depth 6 --hidden_dim 36 --seed 2 --num_labels 10 --n_epochs 8
#python sort.py --width 2 --depth 3 --hidden_dim 36 --seed 2 --num_labels 5

#### sort 
#python sort.py --width 1 --depth 8 --hidden_dim 32 --seed 2 --num_labels 2 --n_epochs 1 --bert
#python sort.py --width 2 --depth 6 --hidden_dim 32 --seed 2 --num_labels 5 --n_epochs 5 --bert --path_len 1 --n_paths 2
#python sort.py --width 2 --depth 6 --hidden_dim 48 --seed 2 --num_labels 4 --n_epochs 15 --bert --path_len 1 --n_paths 7
python sort.py --width 2 --depth 6 --hidden_dim 48 --seed 2 --num_labels 4 --n_epochs 15 --bert --path_len 1 --n_paths 7 --all_heads

python sort.py --width 2 --depth 6 --hidden_dim 32 --seed 2 --num_labels 25 --n_epochs 65 --bert --path_len 1 --n_paths 10 --n_repeat 3 --n_train_data 500
python sort.py --width 2 --depth 6 --hidden_dim 32 --seed 2 --num_labels 25 --n_epochs 65 --bert --path_len 0 --n_paths 10 --n_repeat 3 --n_train_data 500
python sort.py --width 2 --depth 6 --hidden_dim 32 --seed 2 --num_labels 32 --seq_len 8 --n_epochs 65 --bert --path_len 0 --n_paths 10 --n_repeat 3 --n_train_data 500
#--no_repeat

#Run convex hull
python convex_hull.py --width 2 --depth 6 --bert --seq_len 10 --hidden_dim 32 --seed 2 --num_labels 2 --n_epochs 15 --path_len 1 --n_paths 10 --n_train_data 10000 --n_repeat 2
python convex_hull.py --width 4 --depth 12 --bert --seq_len 10 --hidden_dim 128 --seed 2 --num_labels 2 --n_epochs 15 --path_len 0 --n_paths 20 --n_train_data 500 --n_repeat 2 --n_eval_data 100 --all_heads
#plot
python convex_hull.py --width 3 --depth 6 --bert --seq_len 10 --hidden_dim 66 --seed 2 --num_labels 2 --n_epochs 65 --path_len 0 --n_paths 15 --n_train_data 10000 --n_repeat 2 --n_eval_data 100 --no_sub_path
python convex_hull.py --width 2 --depth 6 --bert --seq_len 10 --hidden_dim 80 --seed 2 --num_labels 2 --n_epochs 65 --path_len 0 --n_paths 5 --n_train_data 10001 --n_repeat 2 --n_eval_data 150 --no_sub_path
python convex_hull.py --width 3 --depth 6 --bert --seq_len 8 --seed 2 --num_labels 2 --n_epochs 70 --path_len 0 --n_train_data 10001 --n_repeat 5 --n_eval_data 250 --no_sub_path --hidden_dim 84 --n_paths 5
python convex_hull.py --width 3 --depth 6 --bert --seq_len 8 --seed 2 --num_labels 2 --n_e
pochs 70 --path_len 0 --n_train_data 10001 --n_repeat 5 --n_eval_data 250 --no_sub_path --hidden_dim 84 --n_paths 5
python convex_hull.py --width 3 --depth 6 --bert --seq_len 8 --hidden_dim 84 --seed 2 --num_labels 2 --n_epochs 70 --path_len 0 --n_paths 20 --n_train_data 10001 --n_repeat 2 --ffn2 --n_eval_data 200 --no_sub_path

python sort.py --width 2 --depth 6 --bert --hidden_dim 32 --seed 2 --num_labels 25 --n_epochs 65 --path_len 1 --n_paths 15 --n_train_data 500 --n_repeat 3 --n_eval_data 100 --no_sub_path
python sort.py --width 2 --depth 6 --bert --hidden_dim 32 --seed 2 --num_labels 32 --seq_len 8 --n_epochs 65 --path_len 0 --n_paths 5 --n_train_data 1000 --n_repeat 5 --n_eval_data 150 --no_sub_path
python sort.py --width 2 --depth 6 --bert --hidden_dim 48 --seed 2 --num_labels 10 --seq_len 8 --n_epochs 105 --path_len 0 --n_paths 20 --n_train_data 10000 --n_repeat 5 --n_eval_data 150 --no_sub_path

--train_as_eval
--no_sub_path
non monotonic python convex_hull.py --width 4 --depth 6 --bert --seq_len 10 --hidden_dim 64 --seed 2 --num_labels 2 --n_epochs 65 --path_len 1 --n_paths 20 --n_train_data 10000 --n_repeat 2

#no overfitting
python convex_hull.py --width 1 --depth 8 --bert --seq_len 10 --hidden_dim 48 --seed 2 --num_labels 2 --n_epochs 65 --path_len 1 --n_paths 15 --n_train_data 500 --n_repeat 2 --n_eval_data 50
#auc
python convex_hull.py --width 3 --depth 6 --bert --seq_len 10 --hidden_dim 120 --seed 2 --num_labels 2 --n_epochs 65 --path_len 0 --n_paths 20 --n_train_data 10000 --n_repeat 2
#numpy.linalg.norm(aa.cpu().numpy(), float('inf'))*numpy.linalg.norm(aa.cpu().numpy(), 1)

#Circle
python circle.py --width 2 --depth 1 --bert --seed 2 --num_labels 2 --seq_len 10 --n_epochs 45 --n_train_data 1000 --n_repeat 2  --n_eval_data 150 --hidden_dim 24
python circle.py --width 2 --depth 1 --bert --seed 2 --num_labels 2 --seq_len 10 --n_epochs 45 --n_train_data 1000 --n_repeat 2 --n_eval_data 150 --hidden_dim 84 

--circle_skip
--compute_alpha
--do_mlp
--do_train

python circle.py --width 2 --depth 1 --bert --seed 2 --num_labels 2 --seq_len 10 --n_epochs 105 --n_train_data 1000 --n_repeat 2  --n_eval_data 150 --hidden_dim 32 --do_mlp  #or circle_skip n_epochs can be 65 without these
python circle.py --width 2 --depth 1 --bert --seed 2 --num_labels 2 --seq_len 10 --n_epochs 205 --n_train_data 1000 --n_repeat 2 --hidden_dim 128 --n_eval_data 100 --circle_skip --do_train
