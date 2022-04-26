# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import joblib
import pandas as pd
import os
import time
import json
import numpy as np

from aro.model.zone_utils import zone_based_tsp

def get_route_with_station(df, route_id):
    dfr = df[(df.route_id == route_id)].sort_values(
        ["is_station", "stop"], ascending=(False, True)
    )  # make sure depot is always the first
    return dfr

if __name__ == "__main__":
    """
    check example_inference.sh for usage
    """
    parser = argparse.ArgumentParser(description="inference")
    parser.add_argument(
        "--model_dir", required=True, type=str, 
        help="Directory where the PPM model file is"
    )
    parser.add_argument(
        "--ppm_model_fn", required=True, type=str, 
        help="File name of the PPM model"
    )
    parser.add_argument(
        "--data_dir", required=True, type=str, 
        help="Directory where the parquet file lmc_route_full_16xxxxx.parquet is"
    )
    parser.add_argument(
        "--dist_matrix_dir", required=True, type=str, 
        help="Directory of distance_matrix, where files like RouteID_*_raw_w_st.npy are"
    )
    parser.add_argument(
        "--zone_list_dir", required=True, type=str, 
        help="Directory of zone_list, where files like RouteID_*_zone_w_st.joblib are"
    )
    parser.add_argument(
        "--output_dir", required=True, type=str, help="Directory of output"
    )

    parser.add_argument(
        "--output_score_dir", required=True, type=str, help="Directory of output"
    )

    parser.add_argument(
        "--data_fn", required=True, type=str, 
        help="data file name, e.g. lmc_route_full_16xxxxx.parquet"
    )

    parser.add_argument(
        "--model_score_input_dir", required=True, type=str, help="Directory of model score input"
    )
    parser.add_argument(
        "--actual_seq_fn", required=True, type=str, help="data file name for actual sequences"
    )
    parser.add_argument(
        "--invalid_seq_fn", required=True, type=str, help="data file name for invalid sequences"
    )

    parser.add_argument(
        "--model_apply_input_dir", required=True, type=str, help="Directory of model apply input"
    )
    parser.add_argument(
        "--travel_time_fn", required=True, type=str, help="data file name for travel time"
    )

    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        default=False,
        help="debug or not",
    )

    args = parser.parse_args()
    df_val = pd.read_parquet(os.path.join(args.data_dir, args.data_fn))

    json_dict = dict()
    stt = time.time()
    
    mfn = f"{args.model_dir}/{args.ppm_model_fn}"
    if os.path.exists(mfn):
        zone_prob_model = joblib.load(mfn)
    else:
        raise Exception(f'Cannot find {mfn}')
    for idx, route_id in enumerate(df_val.route_id.unique()):
        df_route = get_route_with_station(
        df_val,
        route_id,
        )
        nb_nodes = df_route.shape[0]
        
        fn = f"{args.dist_matrix_dir}/{route_id}_raw_w_st.npy"
        if os.path.exists(fn):
            pre_dist_matrix = np.load(fn)

        zfn = f"{args.zone_list_dir}/{route_id}_zone_w_st.joblib"
        if (os.path.exists(zfn)):
            zone_list = joblib.load(zfn)
            cw = [0.25, 0.25, 0.25, 0.25]
            zs_algo = 'ppm'
            sol_list = zone_based_tsp(
                        pre_dist_matrix,
                        zone_list,
                        zone_prob_model,
                        route_id,
                        cluster_weights=cw,
                        zone_sort_algo=zs_algo
                    )
        else:
            print(f'zone files missing: {zfn}')
        rank_list = [-1] * len(sol_list)  # df_route.shape[0]
        for i, rank in enumerate(sol_list):
            try:
                rank_list[rank] = i
            except:
                print(sol_list)
                raise Exception(f"{i}, {rank}")
        
        df_route["myrank"] = rank_list
        df_route = df_route.sort_values(["stop"])
        sequence_dict = {}
        for _, stop_item in df_route.iterrows():
            rank_id = stop_item.myrank
            stop_id = stop_item.stop
            sequence_dict[stop_id] = rank_id
        proposal_dict = {"proposed": sequence_dict}
        json_dict[route_id] = proposal_dict
        if (args.debug):
            break
        if (idx + 1) % 10 == 0:
            avg_speed = (time.time() - stt) / (idx + 1)
            print(f'Number of routes done: {idx + 1}, avg speed = {avg_speed:.3f} per route')
    ttt = int(time.time())
    args_input_perf_file = f"eval-ppm-rollout-{ttt}.json"
    output_json_file = os.path.join(args.output_dir, args_input_perf_file)
    with open(output_json_file, "w") as fout:
        json.dump(json_dict, fout)