# AWS Last Mile Route Optimization

This Git repository showcases our solution to the [Amazon Last Mile Routing Research Challenge](https://routingchallenge.mit.edu/). The solution runs as an Amazon SageMaker Processing Job, and is based on our paper **[Learning from Drivers to Tackle the Amazon Last Mile Routing Research Challenge](https://arxiv.org/)**. The diagram below shows an overview of our method. Our solution hierarchically integrates Markov model training, online policy search (Rollout), and a conventional Traveling Salesperson Problem (TSP) solver to produce driver friendly routes during last mile planning. The choice of the underlying TSP solver is flexible. For example, our paper reported the evaluation score of 0.0374 using [LKH](http://akira.ruc.dk/~keld/research/LKH/). This Git repository uses [OR-tools](https://github.com/google/or-tools) for simplicity, and obtains a nearly identical score of 0.0372. These results are comparable to what the top three teams have achieved on the public [leaderboard](https://routingchallenge.mit.edu/last-mile-routing-challenge-team-performance-and-leaderboard/).

<img src="method.png" alt="An overview of our method" width="800"/>

# Quick Start

## 1. Install
```bash
# Get the source code
git clone https://github.com/aws-samples/amazon-sagemaker-amazon-routing-challenge-sol.git
cd amazon-sagemaker-amazon-routing-challenge-sol

# Setup Python Environment
conda create --name aro python=3.8
conda activate aro # `aro` is the name of the python virtual environment, feel free to change it
# if on a SageMaker notebook instance, uncomment and execute the following command instead
# source activate aro

# Install the current version of the package
pip install -r requirements_dev.txt
pip install -e .
```

## 2. Obtain data
Register at https://registry.opendata.aws/ in order to download the Challenge data. More specific instructions are comging soon.
The following code snippets assume you have downloaded the `train` and `evaluation` datasets to your local machine at `/tmp/Final_March_15_Data` and `/tmp/Final_June_18_Data` respectively.

## 3. Preprocess data
```bash
train_data_dir=Final_March_15_Data
eval_data_dir=Final_June_18_Data
mkdir data
# please replace the example source directory `/tmp` with actual path for downloaded datasets
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

## 4. Upload pre-processed data to S3
```bash
export bucket_name=my-route-bucket # set `${bucket_name}` to your own S3 bucket name
export s3_data_prefix=lmc # set `${s3_data_prefix}` to your own S3 data prefix
aws s3 sync data/${train_data_dir}/ s3://${bucket_name}/data/${s3_data_prefix}/${train_data_dir}/
aws s3 sync data/${eval_data_dir}/ s3://${bucket_name}/data/${s3_data_prefix}/${eval_data_dir}/
```
After completing this step, the prefix structure in your S3 bucket appears as follows:
``` bash
└── ${train_or_eval_data_dir}                   # e.g. `Final_March_15_Data` OR `Final_June_18_Data`
    ├── distance_matrix/                        # a directory with all distance matrix files
    │   ├── ${route_id_0}_raw_w_st.npy          # distance matrix file produced by preprocessing.py 
    │   ├── ...                                 # more distance matrix files
    │   └── ${route_id_N}_raw_w_st.npy          # distance matrix file
    ├── model_apply_inputs/                     # challenge input files
    ├── model_apply_outputs/                    # output json results here
    ├── model_build_inputs/                     # challenge input files
    ├── model_score_inputs/                     # challenge input files
    ├── model_score_outputs/                    # output json score here
    ├── processed/                              # output processed parquet file here
    └── zone_list                               # A directory with all zone files
        ├── ${route_id_0}_zone_w_st.joblib      # zone file produced by preprocessing.py            
        ├── ...                                 # more zone files
        ├── ${route_id_N}_zone_w_st.joblib      # the last zone file
        └── actual_zone-{mode}.csv              # ground-truth zone sequence file produced by preprocessing.py
```

## 5. Train the PPM model
This step trains the [Prediction by Partial Matching (PPM for short)](https://en.wikipedia.org/wiki/Prediction_by_partial_matching) model. In our work, the PPM model is used as a sequential probability model for generating synthetic zone sequences.
```bash
# train the PPM model
python train.py --train_zdf_fn data/${train_data_dir}/zone_list/actual_zone-train.csv
```

## 6. Upload the trained model to S3
We upload the PPM model to S3 so that the subsequent SageMake processing job can access the model for zone sequence generation.
```bash
export s3_model_prefix=almrc # optional - set `${s3_model_prefix}` with your own S3 model prefix
aws s3 cp aro_ppm_train_model.joblib s3://${bucket_name}/models/${s3_model_prefix}/
```

## 7. Run route generation locally
We run the inferernce for route generation locally (e.g. on your laptop or desktop) for the purpose of debugging or understanding the inner workings of our approach.
```bash
# test inference locally
./example_inference.sh
```
Change your script accordingly as per any debugging information revealed in the output or error statements.

## 8. Run route generation as a SageMaker processing job
Now we are ready to generate routes by submititng an Amazon SageMaker processing job running on a `ml.m5.4xlarge` instance.
```bash
# please set environment variables (e.g. ${bucket_name}) in `example_inference_job.sh` before running it
./example_inference_job.sh
```
Once submission is successful, we can open the Amazon SageMaker Processing jobs console to check if a job named `ppm-rollout-2022-xxx` is indeed running.

## 9. Check submission file
It should take less than 60 minutes to complete the processing job. Once the job status becomes 'completed', we can check the generated sequences for all routes by running the following command,
```bash
aws s3 ls s3://${bucket_name}/data/${s3_data_prefix}/${eval_data_dir}/model_apply_outputs/eval-ppm-rollout
```

## 10. Get evaluation scores
Once the submission file is downloaded, follow the evaluation instructions at https://github.com/MIT-CAVE/rc-cli
to calculate the evaluation score, which should be around `0.0372` ~ `0.0376`

## 11. Integrate this example into your last mile routing applications
If you are intereted in this example and its applications, please feel free to [contact us](https://github.com/chenwuperth).