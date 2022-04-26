#!/usr/bin/env bash

# the data_root direcotry structure is consistent with the preprocessing.py and the S3 bucket used by inference_job.py
data_root=/home/ec2-user/SageMaker/almrc-sol/data/Final_June_18_Data
current_dir=/home/ec2-user/SageMaker/almrc-sol
python inference.py --model_dir ${current_dir} \
    --ppm_model_fn aro_ppm_train_model.joblib \
    --data_dir ${data_root}/processed \
    --data_fn lmc_route_full_1637316909.parquet \
    --output_dir ${data_root}/model_apply_outputs \
    --dist_matrix_dir ${data_root}/distance_matrix \
    --zone_list_dir ${data_root}/zone_list \
    --model_score_input_dir ${data_root}/model_score_inputs \
    --actual_seq_fn eval_actual_sequences.json \
    --invalid_seq_fn eval_invalid_sequence_scores.json \
    --model_apply_input_dir ${data_root}/model_apply_inputs \
    --travel_time_fn eval_travel_times.json \
    --output_score_dir ${data_root}/model_score_outputs