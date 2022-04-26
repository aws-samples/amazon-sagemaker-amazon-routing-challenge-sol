# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

from sagemaker.pytorch import PyTorchProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

"""
Running inferene using the SageMaker processing job

python inference_job.py
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="inference job")
    parser.add_argument(
        "--model_uri", required=True, 
            type=str, help="S3 model prefix"
    )
    parser.add_argument(
        "--mode", default='val', type=str, help="only supports two modes: train, val"
    )

    parser.add_argument(
        "--sm_role", required=True, 
                     type=str, 
                     help="SageMaker role id, e.g.AmazonSageMaker-ExecutionRole-2021xxxx"
    )

    parser.add_argument(
        "--ppm_model_fn", 
        default="aro_ppm_train_model.joblib", 
        type=str, 
        help="File name of the PPM model"
    )

    parser.add_argument(
        "--s3_prefix",
        required=True,
        type=str,
        help="S3 prefix of the data uri, including the bucket name"
    )

    parser.add_argument(
        "--processed_dfn",
        required=True,
        type=str,
        help="name of the pre-processed file, e.g. lmc_route_full_1637316909.parquet"
    )

    args = parser.parse_args()
    my_base_job_name = 'ppm-rollout'
    my_inst_type = 'ml.m5.4xlarge'
    processed_dfn = args.processed_dfn
    if args.mode == 'val':
        prefix_dir = 'Final_June_18_Data'
        #processed_dfn = 'lmc_route_full_1637316909.parquet'
    elif args.mode == 'train':
        prefix_dir = 'Final_March_15_Data'
        #processed_dfn = 'lmc_route_full_1627030750.parquet'
    else:
        raise Exception(f'unknown mode {args.mode}')

    processor = PyTorchProcessor(
        framework_version='1.7.1',
        py_version='py3',
        role=args.sm_role,
        instance_count=1,
        instance_type=my_inst_type,
        base_job_name=my_base_job_name,
        max_runtime_in_seconds=3600 * 24 * 4
    )
    sm_local_input = '/opt/ml/processing/input'
    sm_local_output = '/opt/ml/processing/output'

    model_dir_local = f'{sm_local_input}/models'
    pi_model = ProcessingInput(
        source=args.model_uri,
        destination=model_dir_local)
    
    data_dir_local = f'{sm_local_input}/data'
    pi_data = ProcessingInput(
        source=f'{args.s3_prefix}/{prefix_dir}/processed/',
        destination=data_dir_local
        )
    
    dist_matrix_dir_local = f'{sm_local_input}/distance_matrix'
    pi_dist_matrix = ProcessingInput(
        source=f'{args.s3_prefix}/{prefix_dir}/distance_matrix/',
        destination=dist_matrix_dir_local
        )
    
    zone_list_dir_local = f'{sm_local_input}/zone_list'
    pi_zone_list = ProcessingInput(
        source=f'{args.s3_prefix}/{prefix_dir}/zone_list/',
        destination=zone_list_dir_local
        )
    
    model_score_dir_local = f'{sm_local_input}/model_score'
    pi_model_score = ProcessingInput(
        source=f'{args.s3_prefix}/{prefix_dir}/model_score_inputs/',
        destination=model_score_dir_local
    )
    
    model_apply_dir_local = f'{sm_local_input}/model_apply_input'
    pi_model_apply = ProcessingInput(
        source=f'{args.s3_prefix}/{prefix_dir}/model_apply_inputs/',
        destination=model_apply_dir_local
    )

    output_dir_local = f'{sm_local_output}/submission'
    po_data = ProcessingOutput(
        source=output_dir_local,
        destination=f'{args.s3_prefix}/{prefix_dir}/model_apply_outputs'
    )

    output_score_local = f'{sm_local_output}/score'
    po_score = ProcessingOutput(
        source=output_score_local,
        destination=f'{args.s3_prefix}/{prefix_dir}/model_score_outputs'
    )
    
    pargs = []
    pargs.append(f"--model_dir {model_dir_local}")
    pargs.append(f"--ppm_model_fn {args.ppm_model_fn}")
    pargs.append(f"--data_dir {data_dir_local}")
    pargs.append(f"--data_fn {processed_dfn}")
    pargs.append(f"--dist_matrix_dir {dist_matrix_dir_local}")
    pargs.append(f"--zone_list_dir {zone_list_dir_local}")

    pargs.append(f"--model_score_input_dir {model_score_dir_local}")
    pargs.append("--actual_seq_fn eval_actual_sequences.json")
    pargs.append("--invalid_seq_fn eval_invalid_sequence_scores.json")

    pargs.append(f"--model_apply_input_dir {model_apply_dir_local}")
    pargs.append("--travel_time_fn eval_travel_times.json")

    pargs.append(f"--output_dir {output_dir_local}")
    pargs.append(f"--output_score_dir {output_score_local}")

    parguments = " ".join(pargs).split()
    deps = ["aro"]
    processor.run(code="inference.py",
                    source_dir='.',
                    dependencies=deps,
                    inputs=[pi_model, pi_data, pi_dist_matrix, pi_zone_list, 
                            pi_model_score, pi_model_apply], 
                    outputs=[po_data, po_score], 
                    wait=False, logs=False, 
                    arguments=parguments)

    