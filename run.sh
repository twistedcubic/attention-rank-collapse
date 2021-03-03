#!/bin/bash

#Some sample commands, usage as outlined in sort.py, or can be viewed with `python sort.py --h`.

cur_path=$(realpath $0)
cur_dir=$(dirname $cur_path)
export PYTHONPATH=$cur_dir:$PYTHONPATH

python convex_hull.py --width 3 --depth 6 --bert --seq_len 10 --hidden_dim 66 --seed 2 --num_labels 2 --n_epochs 65 --path_len 0 --n_paths 15 --n_train_data 10000 --n_repeat 2 --n_eval_data 100 --no_sub_path

python sort.py --width 2 --depth 6 --bert --hidden_dim 32 --seed 2 --num_labels 32 --seq_len 8 --n_epochs 65 --path_len 0 --n_paths 5 --n_train_data 1000 --n_repeat 5 --n_eval_data 150 --no_sub_path

#--train_as_eval
#--no_sub_path
python circle.py --width 2 --depth 1 --bert --seed 2 --num_labels 2 --seq_len 10 --n_epochs 45 --n_train_data 1000 --n_repeat 2 --n_eval_data 150 --hidden_dim 64

#Examples of additional options:
#--circle_skip
#--compute_alpha
#--do_mlp
#--do_train
