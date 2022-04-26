#!/usr/bin/env bash

# note that the `s3_prefix` argument does NOT include the {train_or_eval_data_dir} directory, e.g.

# └── {s3_prefix}                       # e.g. s3://my-bucket/data
#    ├── {train_or_eval_data_dir}       # e.g. `Final_March_15_Data` OR `Final_June_18_Data`
#       ├── distance_matrix/            # distance matrix file produced by preprocessing.py 
#       ├── ...                         # challenge input/output directories
#       ├── processed/                  # distance matrix file
#       └── zone_list/                  # zone files produced by preprocessing.py 

bucket_name=my-bucket-name # please change this
sagemaker_exec_role=AmazonSageMaker-ExecutionRole-2022xxxxx # please change this
s3_model_prefix=almrc # optional change this
s3_data_prefix=lmc # optional change this

# please change this as per your preprocessing output
# under the folder: ${data_root}/processed
# e.g. data_root=/home/ec2-user/SageMaker/almrc-sol/data/Final_June_18_Data
preprocessed_filename=lmc_route_full_1637316909.parquet 

python inference_job.py \
    --model_uri s3://${bucket_name}/models/${s3_model_prefix}/ \
    --sm_role ${sagemaker_exec_role} \
    --s3_prefix s3://${bucket_name}/data/${s3_data_prefix} \
    --processed_dfn ${preprocessed_filename} \
    --mode val
