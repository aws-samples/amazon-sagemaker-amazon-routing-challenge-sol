# AWS Last Mile Route Optimization

This Git repository showcases our solution to the [Amazon Last Mile Routing Research Challenge](https://routingchallenge.mit.edu/). The solution runs as an Amazon SageMaker Processing Job, and is based on our paper **[Learning from Drivers to Tackle the Amazon Last Mile Routing Research Challenge](https://arxiv.org/)**. The diagram below shows an overview of our method. Our solution hierarchically integrates Markov model training, online policy search (Rollout), and a conventional Traveling Salesperson Problem (TSP) solver to produce driver friendly routes during last mile planning. The choice of the underlying TSP solver is flexible. For example, our paper reported the evaluation score of 0.0374 using [LKH](http://akira.ruc.dk/~keld/research/LKH/). This Git repository uses [OR-tools](https://github.com/google/or-tools) for simplicity, and obtains a nearly identical score of 0.0372. These results are comparable to what the top three teams have achieved on the public [leaderboard](https://routingchallenge.mit.edu/last-mile-routing-challenge-team-performance-and-leaderboard/).

<img src="method.png" alt="An overview of our method" width="800"/>

## Install
```bash
# Setup Python Environment
conda create --name aro python=3.8
conda activate aro

# Install the current version of the package
pip install -r requirements_dev.txt
pip install -e .

```

## Obtain data
Visit https://registry.opendata.aws/ to register and download the Challenge data. Specific instructions are comging soon.
Assume you download two datasets in two directories `Final_March_15_Data` and `Final_June_18_Data` respectively.

## Preprocess data
```bash
train_data_dir=Final_March_15_Data
eval_data_dir=Final_June_18_Data
cd ${repository_dir}
mkdir data
# please replace the source directory `/tmp` with the directory where data files are located
mv /tmp/${train_data_dir} data/
mv /tmp/${eval_data_dir} data/

python preprocessing.py --act gen_route --data_dir  data/${train_data_dir}
python preprocessing.py --act gen_dist_mat --data_dir  data/${train_data_dir}
python preprocessing.py --act gen_zone_list --data_dir  data/${train_data_dir}
python preprocessing.py --act gen_actual_zone --data_dir  data/${train_data_dir}

python preprocessing.py --act gen_route --data_dir  data/${eval_data_dir}
python preprocessing.py --act gen_dist_mat --data_dir  data/${eval_data_dir}
python preprocessing.py --act gen_zone_list --data_dir  data/${eval_data_dir}
```

## Run the example
```bash
# train the PPM model
python train.py --train_zdf_fn data/${train_data_dir}/zone_list/actual_zone-train.csv

# test inference locally
bash example_inference.sh

# upload data to S3
# optional - set `${s3_data_prefix}` with your own S3 data prefix
s3_data_prefix=lmc
aws s3 sync data/${train_data_dir}/ s3://${bucket_name}/data/${s3_data_prefix}/${train_data_dir}/
aws s3 sync data/${eval_data_dir}/ s3://${bucket_name}/data/${s3_data_prefix}/${eval_data_dir}/

# upload trained model to S3
s3_model_prefix=almrc # optional - set `${s3_model_prefix}` with your own S3 model prefix
aws s3 cp aro_ppm_train_model.joblib s3://${bucket_name}/models/${s3_model_prefix}/

# submit the SageMaker processing job for model inference
# please set environment variables (e.g. ${bucket_name}) in `example_inference_job.sh` before running it
bash example_inference_job.sh

# Log onto Amazon SageMaker Processing jobs console to check if a job named `ppm-rollout-2022-xxx` is running
# It should take less than 60 minutes to complete the processing job.

# Check and download the latest submission file from S3
aws s3 ls s3://${bucket_name}/data/${s3_data_prefix}/${eval_data_dir}/model_apply_outputs/eval-ppm-rollout

# Once the submission file is downloaded, follow the evaluation instructions at https://github.com/MIT-CAVE/rc-cli
# to calculate the evaluation score, which should be around 0.0372 ~ 0.0376
```

## The directory structure of the dataset
``` bash
└── {train_or_eval_data_dir}                    # e.g. Final_March_15_Data OR Final_June_18_Data
    ├── distance_matrix/                        # a directory with all distance matrix files
    │   ├── {route_id_0}_raw_w_st.npy           # distance matrix file produced by preprocessing.py 
    │   ├── ...                                 # more distance matrix files
    │   └── {route_id_N}_raw_w_st.npy           # distance matrix file
    ├── model_apply_inputs/                     # challenge input files
    ├── model_apply_outputs/                    # output json results here
    ├── model_build_inputs/                     # challenge input files
    ├── model_score_inputs/                     # challenge input files
    ├── model_score_outputs/                    # output json score here
    ├── processed/                              # output processed parquet file here
    └── zone_list                               # A directory with all zone files
        ├── {route_id_0}_zone_w_st.joblib       # zone file produced by preprocessing.py            
        ├── ...                                 # more zone files
        ├── {route_id_N}_zone_w_st.joblib       # the last zone file
        └── actual_zone-{mode}.csv              # ground-truth zone sequence file produced by preprocessing.py
```
