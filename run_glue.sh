#!/bin/bash

#export GLUE_DIR=/data/users/yihdong/data/glue_data
#export GLUE_DIR=/home/yihdong/glue_data
#put GLUE data for a task under data dir to run that task.
export GLUE_DIR=data
cur_path=$(realpath $0)
cur_dir=$(dirname $cur_path)
export PYTHONPATH=$cur_dir:$PYTHONPATH
echo $cur_dir
export GLUE_DIR=/usr/local/google/home/yihed/glue_data
#export GLUE_DIR=/TEE2-lustre/users/yihdong/data/glue_data
export TASK_NAME=MRPC
export TASK_NAME=CoLA
export TASK_NAME=RTE
#export PYTHONPATH=/data/users/yihdong/transformers/src:/data/users/yihdong/transformers:$PYTHONPATH
#export PYTHONPATH=/usr/local/google/home/yihed/transformers:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

python ./examples/text-classification/run_glue.py \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --do_train \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_device_eval_batch_size=32   \
    --per_device_train_batch_size=8   \
    --learning_rate 3e-4 \
    --num_train_epochs 50 \
    --output_dir snap/$TASK_NAME \
    --overwrite_output_dir \
    --seed 2 \
    --width 2 --depth 6 --hidden_dim 250 --seed 2 --n_repeat 5 --n_paths 5 $1 --n_train_data 500 --path_len 0  --no_sub_path

# --compute_alpha --all_heads  
#--n_paths 15 --no_sub_path
#--width 4 --depth 12 --bert --seq_len 10 --hidden_dim 128 --seed 2 --num_labels 2 --n_epochs 15 --path_len 3 --n_paths 20 --n_train_data 500 --n_repeat 2 --n_eval_data 100 --all_heads
#    --n-gpu 1
#    --per_device_train_batch_size=128   \
#    --per_device_eval_batch_size=1   \
#    --per_device_train_batch_size=1   \
#    --do_eval \
