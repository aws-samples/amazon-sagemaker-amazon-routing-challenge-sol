# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import time
from collections import defaultdict
from datetime import datetime
import os

import numpy as np
import pandas as pd
import utm
from scipy.spatial import distance
from sklearn import manifold
from sklearn.metrics.pairwise import haversine_distances
from tqdm import tqdm
import json

"""
preprocess routes data for training the TSP model
"""

def _dist2coord(dist_matrix):
    seed = np.random.RandomState(seed=3)
    mds = manifold.MDS(
        n_components=2,
        max_iter=300,
        eps=1e-3,
        random_state=seed,
        dissimilarity="precomputed",
        n_jobs=1,
        verbose=0,
        n_init=4,
    )
    fit_re = mds.fit(dist_matrix)
    pos, stress = fit_re.embedding_, fit_re.stress_
    return pos[:, 0], pos[:, 1]

def _scale_coordinates(df, space=True, exclude_station=True):
    col_nm_suffix = "_norm"
    if exclude_station:
        col_nm_suffix += "_wo_st"  # without station
    else:
        col_nm_suffix += "_w_st"
    if space:
        x_col = "x_sp"
        y_col = "y_sp"
    else:
        x_col = "x_tm"
        y_col = "y_tm"

    df_len = df.shape[0]
    new_x_col = f"{x_col}{col_nm_suffix}"
    new_y_col = f"{y_col}{col_nm_suffix}"
    df[new_x_col] = [-1.0] * df_len
    df[new_y_col] = [-1.0] * df_len

    for i, col in enumerate(df.columns):
        if col == new_x_col:
            x_col_ind = i
        elif col == new_y_col:
            y_col_ind = i

    if exclude_station:
        ind = df.index[df.is_station == 0].tolist()
        X = df[df.is_station == 0][x_col]
        Y = df[df.is_station == 0][y_col]
    else:
        ind = list(range(df_len))
        X = df[x_col]
        Y = df[y_col]

    max_value = max(np.max(X - np.min(X)), np.max(Y - np.min(Y)))

    # scale by the same magnitude to keep the correlation
    X1 = (X - np.min(X)) / max_value
    Y1 = (Y - np.min(Y)) / max_value
    df.iloc[ind, x_col_ind] = X1
    df.iloc[ind, y_col_ind] = Y1

def normalise_route(df_r, route_id, df_p, row=None, project_2d=True):
    """
    if df_t (travel time table) is not None, use travel_time to "recover" temporal node coordinates
    "project_2d" is checked only of df_t == None
    """
    lats, lons = [], []
    X_sp = []
    Y_sp = []
    is_station = []
    stops = []
    for idx, (k, v) in enumerate(df_r.loc[route_id].stops.items()):
        stops.append(k)
        lat = v["lat"]
        lon = v["lng"]
        lats.append(lat)
        lons.append(lon)
        if project_2d:
            x, y, _, _ = utm.from_latlon(v["lat"], v["lng"])  # 3D project to 2D
        else:
            x, y = lon + 360 if lon < 0 else lon, v["lat"]
        X_sp.append(x)
        Y_sp.append(y)
        if v["type"] == "Station":
            is_station.append(1)
        else:
            is_station.append(0)
    df = pd.DataFrame(
        {
            "stop": stops,
            "latitude": lats,
            "longitude": lons,
            "is_station": is_station,
            "x_sp": X_sp,
            "y_sp": Y_sp,
        }
    )
    df = df.sort_values(["is_station"], ascending=False)
    df.reset_index(drop=True, inplace=True)
    _scale_coordinates(df, space=True, exclude_station=True)
    _scale_coordinates(df, space=True, exclude_station=False)
    if row is not None:
        # TODO need to check "exclude_station" first
        dist_matrix = _get_raw_time_matrix(row, df.stop.values)
        X_tm, Y_tm = _dist2coord(dist_matrix)
        df["x_tm"] = X_tm
        df["y_tm"] = Y_tm
        _scale_coordinates(df, space=False, exclude_station=False)
        _scale_coordinates(df, space=False, exclude_station=True)
    return df


def normalise_time_window(df_p, route_id, df_r, df, exclude_station=True):
    if exclude_station:
        norm_coords = df[df.is_station == 0].loc[:, ["x_tm_norm_wo_st", "y_tm_norm_wo_st"]].values
        origin_coords = df[df.is_station == 0].loc[:, ["x_tm", "y_tm"]].values
    else:
        norm_coords = df.loc[:, ["x_tm_norm_w_st", "y_tm_norm_w_st"]].values
        origin_coords = df.loc[:, ["x_tm", "y_tm"]].values

    station_node = df[df.is_station == 1].stop[0]
    norm_dist_matrix = distance.cdist(norm_coords, norm_coords, "euclidean")
    origin_dist_matrix = distance.cdist(origin_coords, origin_coords, "euclidean")

    # this produces a matrix where all elements are the same (i.e. scaling factor)
    # except the diagonal, so we skip divisioning on the entire matrix
    # scf = np.divide(origin_dist_matrix, norm_dist_matrix)
    scale_factor = origin_dist_matrix[2, 4] / norm_dist_matrix[2, 4]

    r_route = df_r.loc[route_id]
    depart_dtstr = r_route.date_YYYY_MM_DD + " " + r_route.departure_time_utc
    depart_dto = datetime.strptime(depart_dtstr, "%Y-%m-%d %H:%M:%S")
    # assuming the Courier works up to 12 hours a day
    scaled_one_day_in_seconds = 12 * 3600 / scale_factor

    p_route = df_p.loc[route_id]
    st_times = defaultdict(list)
    ed_times = defaultdict(list)
    st_times_orig = defaultdict(list)
    ed_times_orig = defaultdict(list)
    svc_times = defaultdict(list)
    for col in df_p.columns:
        it = p_route[col]
        if type(it) is not dict:  # TODO - better ways to check NaN value?
            continue
        for pkg_id, pkg_info in it.items():
            svc_time = pkg_info["planned_service_time_seconds"]
            if svc_time is not None:
                svc_times[col].append(svc_time)
            tw = pkg_info["time_window"]
            stu = tw["start_time_utc"]
            etu = tw["end_time_utc"]
            if stu is None:
                st_times[col].append(0)
            else:
                sto = datetime.strptime(stu, "%Y-%m-%d %H:%M:%S")
                ttt = (sto - depart_dto).total_seconds() / scale_factor
                st_times[col].append(max(ttt, 0))
                st_times_orig[col].append(sto)
            if etu is None:
                ed_times[col].append(scaled_one_day_in_seconds)
            else:
                edo = datetime.strptime(etu, "%Y-%m-%d %H:%M:%S")
                ed_times[col].append((edo - depart_dto).total_seconds() / scale_factor)
                ed_times_orig[col].append(edo)

    node_to_index_map = {stop: i for i, stop in enumerate(df["stop"])}
    # merge window using a strict rule (get the smallest window)
    df_len = df.shape[0]
    st_times_rt = [0] * df_len
    ed_times_rt = [0] * df_len
    none_dto = datetime.strptime("1970-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
    st_times_orig_rt = [none_dto] * df_len
    ed_times_orig_rt = [none_dto] * df_len
    svc_time_list = [0] * df_len

    for k, v in svc_times.items():
        svc_time_list[node_to_index_map[k]] = np.sum(v) / scale_factor

    for k, v in st_times.items():
        st_times_rt[node_to_index_map[k]] = np.max(v)

    for k, v in ed_times.items():
        ed_times_rt[node_to_index_map[k]] = np.min(v)

    for k, v in st_times_orig.items():
        if len(v) > 0:
            st_times_orig_rt[node_to_index_map[k]] = np.max(v)
            # datetime.strftime(np.max(v), '%Y-%m-%d %H:%M:%S')

    for k, v in ed_times_orig.items():
        if len(v) > 0:
            ed_times_orig_rt[node_to_index_map[k]] = np.min(v)

    # st_times_rt[node_to_index_map[station_node]] = 0
    st_times_orig_rt[node_to_index_map[station_node]] = depart_dto
    ed_times_rt[node_to_index_map[station_node]] = scaled_one_day_in_seconds
    if exclude_station:
        df["stt_wo_st"] = st_times_rt
        df["edt_wo_st"] = ed_times_rt
        df["svc_tm_wo_st"] = svc_time_list
        df["tm_sc_wo_st"] = scale_factor
    else:
        df["stt_w_st"] = st_times_rt
        df["edt_w_st"] = ed_times_rt
        df["stt_orig"] = st_times_orig_rt
        df["edt_orig"] = ed_times_orig_rt
        df["svc_tm_w_st"] = svc_time_list
        df["tm_sc_w_st"] = scale_factor
    # return st_times_rt, ed_times_rt, scale_factor


def _get_raw_time_matrix(row, stops, symmetric=True):
    dim_mat = len(stops)
    raw_time_matrix = np.zeros((dim_mat, dim_mat))
    for i, col1 in enumerate(stops):
        for j, col2 in enumerate(stops):
            a = row[col1][0][col2]  # + svc_time_dict.get(col2, 0)
            b = row[col2][0][col1]  # + svc_time_dict.get(col1, 0)
            if symmetric:
                c = np.mean([a, b])
                raw_time_matrix[i][j] = c
                raw_time_matrix[j][i] = c
            else:
                raw_time_matrix[i][j] = a
                raw_time_matrix[j][i] = b

    return raw_time_matrix


def gen_all_routes(df_r, df_p, df_t, output_root_dir):
    df_list = []
    tt_len = df_r.shape[0]
    for ind, route_id in tqdm(enumerate(list(df_r.index)), position=0, leave=True):
        row = (df_t.loc[[route_id]]).dropna(axis=1)
        df = normalise_route(df_r, route_id, df_p, row)
        normalise_time_window(df_p, route_id, df_r, df)
        normalise_time_window(df_p, route_id, df_r, df, exclude_station=False)
        df.drop(columns=["x_sp", "y_sp", "x_tm", "y_tm"], inplace=True)
        df["route_id"] = route_id
        new_col_list = [x.replace("_norm", "") for x in list(df.columns)]
        df.columns = new_col_list
        df_list.append(df)
        # if (ind > 10):
        #     break
    print("Concating now")
    dfpq = pd.concat(df_list)
    tm = int(time.time())
    try:
        dfpq.to_parquet(os.path.join(output_root_dir, f"lmc_route_full_{tm}.parquet"), index=False)
    except:
        dfpq.to_csv(os.path.join(output_root_dir, f"lmc_route_full_{tm}.csv"), index=False)


def gen_distance_matrix(
    df_r, df_t, out_dir, include_station=False, add_softmax=True
):
    """
    Attention distance matrix
    exclude the station
    """
    for route_id in tqdm(list(df_r.index)):
        stops = [k for k, v in df_r.loc[route_id].stops.items() if (v["type"] != "Station")]
        if include_station:
            station_list = [
                k for k, v in df_r.loc[route_id].stops.items() if (v["type"] == "Station")
            ]
        stops = station_list + stops
        row = (df_t.loc[[route_id]]).dropna(axis=1)
        dist_matrix = _get_raw_time_matrix(row, stops, symmetric=False)
        if include_station:
            np.save(f"{out_dir}/{route_id}_raw_w_st.npy", dist_matrix)
        else:
            np.save(f"{out_dir}/{route_id}_raw.npy", dist_matrix)

def get_actual_route_by_route_id(df_act_seq, df_val, route_id, x_col, y_col):
    stop_seq_dict = df_act_seq.loc[route_id].values[0]
    ranks, idxs = [], []
    # df_route = df_val[(df_val.route_id == route_id) & (df_val.is_station == 0)]
    df_route = df_val[(df_val.route_id == route_id)].sort_values(
        ["is_station", "stop"], ascending=(False, True)
    )
    for idx, k in enumerate(df_route.stop):
        ranks.append(stop_seq_dict[k])
        idxs.append(idx)
    df_route["rank"] = ranks
    df_route["ind"] = idxs
    X = df_route[[x_col, y_col, "stt_wo_st", "edt_wo_st"]].values
    X = X.reshape([1, X.shape[0], X.shape[1]])
    return X, df_route

def sort_df_into_sequence(df_route):
    df_sequence = df_route.sort_values(["rank"])
    target_lat = list(df_sequence.latitude.values[1:])
    target_lon = list(df_sequence.longitude.values[1:])
    target_lat.append(df_sequence.latitude.values[0])
    target_lon.append(df_sequence.longitude.values[0])
    df_sequence["tlat"] = target_lat
    df_sequence["tlon"] = target_lon
    return df_sequence

def get_actual_zone(df_val, df_act_seq, df_r, data_dir, mode):
    he_set = set()
    # only pick sequences with Hith or Medium quality
    for idx, item in df_r.iterrows():
        if (item.route_score != 'Low'):
            he_set.add(idx)
    lines = []
    fline = "route_id,zone_penalty,zone_seq,full_zone_seq"
    lines.append(fline)
    for route_id in df_val.route_id.unique():
        if route_id not in he_set:
            continue
        _, df_route = get_actual_route_by_route_id(
            df_act_seq, df_val, route_id, "x_sp_wo_st", "y_sp_wo_st"
        )
        
        df_sequence = sort_df_into_sequence(df_route)
        so_list = df_sequence.ind.values
        zfn = f"{data_dir}/zone_list/{route_id}_zone_w_st.joblib"
        zfn = f"{data_dir}/zone_list/{route_id}_zone_w_st.json"
        with open(zfn, "r") as f:
            zone_list = json.load(f)
        actual_zone = [zone_list[x] for x in so_list]
        zone_seq = []
        full_zone_seq = []
        last_zone = None
        for zone in actual_zone:
            if (zone != last_zone):
                zone_seq.append(zone)
                last_zone = zone
            full_zone_seq.append(zone)
        zone_seq_str = '|'.join(zone_seq)
        full_zone_seq_str = '|'.join(full_zone_seq)
        fline = ','.join([route_id, zone_seq_str, full_zone_seq_str])
        lines.append(fline)
    lines_str = os.linesep.join(lines)
    output_fn = os.path.join(data_dir, 'zone_list', f'actual_zone-{mode}.csv')
    with open(output_fn, "w") as fout:
        fout.write(lines_str)
    print(f'File saved to {output_fn}')

def gen_zone_list(df_r, out_dir="data/zone_list"):
    os.makedirs(out_dir, exist_ok=True)
    for route_id in list(df_r.index):
        stops = []
        coords = []
        zones = []
        none_zones = []
        # In Python 3 - df_r.loc[route_id].stops.items() will keep stop sorted alphabetically 
        # which happens to be its insertion order, this sorting is inline with `get_route_with_station()`
        # in the file ro/demo/first_call.py, this is quite important (and rather brittle)
        for idx, (stop, v) in enumerate(df_r.loc[route_id].stops.items()):
            coord = [v['lat'], v['lng']]
            zone = v['zone_id']

            if v["type"] == "Station":
                stops.insert(0, stop)
                coords.insert(0, coord)
                zones.insert(0, 'stz')
            else:
                stops.append(stop)
                coords.append(coord)
                zones.append(zone)
        
        for idx, z in enumerate(zones):
            if z is None:
                none_zones.append(idx)
        
        for idx in none_zones:
            x = np.array(coords[idx]).reshape([1, 2])
            y = np.array(coords)
            result = haversine_distances(np.radians(x), np.radians(y))[0]
            
            sort_idx = np.argsort(result)
            for nn_idx in sort_idx: # nn -> nearest neighbour
                nn_zone = zones[nn_idx]
                if nn_zone is None or nn_zone == 'stz':
                    continue
                
                zones[idx] = nn_zone
                break
        with open(f"{out_dir}/{route_id}_zone_w_st.json", "w") as f:
            json.dump(zones, f)
        #break # debug break

if __name__ == "__main__":
    """
    The structure of the data folder and how each action is related to some sub-folders

    └── {train_or_eval_data_dir}                # e.g. almrrc2021-data-training OR almrrc2021-data-evaluation
        ├── distance_matrix/                        # a directory with all distance matrix files 
        │   ├── {route_id_0}_raw_w_st.npy           # distance matrix file produced by action `gen_dist_mat`
        │   ├── ...                                  
        │   └── {route_id_N}_raw_w_st.npy            
        ├── model_apply_inputs/                     
        ├── model_apply_outputs/                    
        ├── model_build_inputs/                     
        ├── model_score_inputs/                     
        ├── model_score_outputs/                    
        ├── processed/                              # output processed parquet file produced by action `gen_route`
        └── zone_list                               # A directory with all zone files
            ├── {route_id_0}_zone_w_st.joblib       # zone file produced produced by action `gen_zone_list`            
            ├── ...                                 
            ├── {route_id_N}_zone_w_st.joblib        
            └── actual_zone-{mode}.csv              # ground-truth zone sequence file produced by action `gen_actual_zone`
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--act", default="gen_route", help="actions i.e. `gen_route`, `gen_dist_mat`, `gen_zone_list`, `gen_actual_zone`")
    parser.add_argument("--mode", default="train", help="train or eval")
    parser.add_argument("--pkfn", default=None, help="Parquet file name generated by the `gen_route` action")
    parser.add_argument("--data_dir", default="data/almrrc2021-data-training", 
                        help="`almrrc2021-data-training` is for training, and `almrrc2021-data-evaluation` is for evaluation")

    args = parser.parse_args()

    DATA_DIR = args.data_dir
    if not os.path.exists(DATA_DIR):
        raise Exception(f'{DATA_DIR} not found')

    if args.mode.lower() == 'train':
        LMC_ACTUAL_FN = f"{DATA_DIR}/model_build_inputs/actual_sequences.json"
        LMC_PACKAGE_FN = f"{DATA_DIR}/model_build_inputs/package_data.json"
        LMC_ROUTE_FN = f"{DATA_DIR}/model_build_inputs/route_data.json"
        LMC_TRAVEL_TIME_FN = f"{DATA_DIR}/model_build_inputs/travel_times.json"
    else:
        LMC_ACTUAL_FN = f"{DATA_DIR}/model_score_inputs/eval_actual_sequences.json"
        LMC_PACKAGE_FN = f"{DATA_DIR}/model_apply_inputs/eval_package_data.json"
        LMC_ROUTE_FN = f"{DATA_DIR}/model_apply_inputs/eval_route_data.json"
        LMC_TRAVEL_TIME_FN = f"{DATA_DIR}/model_apply_inputs/eval_travel_times.json"
    
    print(LMC_ROUTE_FN)
    print("Reading Route file...", end='', flush=True)
    df_r = pd.read_json(LMC_ROUTE_FN).T
    print('done')
    if "gen_route" == args.act:
        print("Reading travel time file...", end='', flush=True)
        df_t = pd.read_json(LMC_TRAVEL_TIME_FN).T
        print('done')
        print("Reading package file...", end='', flush=True)
        df_p = pd.read_json(LMC_PACKAGE_FN).T
        print('done')
        gen_all_routes(df_r, df_p, df_t, DATA_DIR)
    elif "gen_dist_mat" == args.act:
        print("Reading travel time file...", end='', flush=True)
        df_t = pd.read_json(LMC_TRAVEL_TIME_FN).T
        print("done")
        dist_mat_dir = os.path.join(DATA_DIR, "distance_matrix")
        if (not os.path.exists(dist_mat_dir)):
            os.mkdir(dist_mat_dir)
        gen_distance_matrix(
            df_r, df_t, out_dir=dist_mat_dir, include_station=True, add_softmax=False
        )
    elif "gen_zone_list" == args.act:
        zone_dir = os.path.join(DATA_DIR, "zone_list")
        if (not os.path.exists(zone_dir)):
            os.mkdir(zone_dir)
        gen_zone_list(df_r, out_dir=zone_dir)
    elif "gen_actual_zone" == args.act:
        if args.pkfn is None:
            raise Exception("Please use argument '--pkfn' to specify the name of the parquet file (e.g. lmc_route_full_1637316909.parquet) generated by the gen_route action")
        zdir = os.path.join(DATA_DIR, "zone_list")
        if not os.path.exists(zdir):
            raise Exception('Missing zone directory, please run the gen_zone_list action first')
        df_val = pd.read_parquet(f'{DATA_DIR}/{args.pkfn}')
        df_act_seq = pd.read_json(LMC_ACTUAL_FN).T
        get_actual_zone(df_val, df_act_seq, df_r, DATA_DIR, args.mode)
