# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict

import numpy as np

from aro.model.ortools_helper import run_ortools


def sort_zones(orig_zone_list, prob_model, route_id, cluster_weights=[0.25, 0.25, 0.25, 0.25], algo='hzm'):
    """
    zone_list is a list (sequence) of zones sorted in a random (i.e. original order, like a set)
        Simply pass in the `zone_list` from file: {zone_dir}/{route_id}_zone_w_st.joblib

    return:
        1. an array of sorted zones as a new most "probable" sequence
        2. non-zero standard deviation of the zone "probability" distance matrix
    """
    zone_list = list(dict.fromkeys(orig_zone_list))
    pred_zone_seq, _ = ppm_rollout(zone_list, prob_model, 1.0, cluster_weights=cluster_weights)
    return pred_zone_seq

def zone_based_tsp(
    matrix, zone_list, prob_model, route_id, cluster_weights=[0.25, 0.25, 0.25, 0.25], zone_sort_algo='hzm'
):
    """
    zone_list is a list (sequence) of zones sorted in a random (i.e. original order, like a set)
        Simply pass in the `zone_list` from file: {zone_dir}/{route_id}_zone_w_st.joblib

    Halo effect - use the mean distance from the current zone to other zones to "simulate" some psudo
    nodes (i.e. one psudo node per zone)

    zone_sort_algo:  hzm (hierarchical zone model) - default
             :  ppm (ppm-rollout model)
             :  z2v (zone to vec model)

    """
    zone_sequence = sort_zones(zone_list, prob_model, route_id, cluster_weights, zone_sort_algo)

    # build zone to node dict
    z2n_dict = defaultdict(list)
    for idx, zone in enumerate(zone_list):  # exclude station zone ?
        z2n_dict[zone].append(idx)

    # get representitive node for each zone
    # TODO vectorise this (by stacking multiple identical distance matrix)
    rep_node_index = []
    for idx, zone in enumerate(zone_sequence):
        if zone == "stz":
            rep_node_index.append(0)
            continue
        zone_node_idx = z2n_dict[zone]
        node_vals = []
        node_vals_idx = []
        for single_node_idx in zone_node_idx:
            row_m = np.median(matrix[single_node_idx, :])
            col_m = np.median(matrix[:, single_node_idx])
            node_vals.append(np.mean([row_m, col_m]))
            node_vals_idx.append(single_node_idx)
        qidx = np.argsort(node_vals)[len(node_vals) // 2]
        rep_node_index.append(node_vals_idx[qidx])

    tour = [0]
    for idx, zone in enumerate(zone_sequence):
        if zone == "stz":
            continue
        zone_node_idx = z2n_dict[zone]
        # add subsequence nodes (each of which is a zone here)
        if len(zone_node_idx) == 1:
            tour += zone_node_idx
            continue
        zone_node_idx.insert(0, tour[-1])
        if idx < len(zone_sequence) - 1:
            extra_rep_nodes = rep_node_index[idx + 1 :]
        else:
            extra_rep_nodes = []
        # if zone_node_idx[0] != 0:
        extra_rep_nodes.append(0)  # add the first node as the last node to form a virtual circle
        zone_node_idx += extra_rep_nodes
        part_whole_dict = {k: v for k, v in enumerate(zone_node_idx)}
        my_matrix = matrix[np.ix_(zone_node_idx, zone_node_idx)]
        # print('my_matrix.shape = ', my_matrix.shape)
        my_list = run_ortools(my_matrix)
        added_tour = [part_whole_dict[x] for x in my_list][
            1:
        ]  # exclude the first one, which was done in "last" iter
        tour += [x for x in added_tour if x not in set(extra_rep_nodes)]
    return np.array(tour, dtype=np.int32)

def ppm_rollout(zone_list, ppm_model, no_context_panelty, 
                explore_budget=1, cluster_weights=[0.25, 0.25, 0.25, 0.25]):
    orig_sol, orig_reward = ppm_base(zone_list, "stz", ppm_model, no_context_panelty, 
                                     cluster_weights=cluster_weights)
    nb_nodes = len(zone_list)
    tm_reward = orig_reward
    part_sol = ["stz"]  # use list to maintain the selection sequence too
    part_sol_tt_reward = []
    nb_explorations = 0

    tmp_good_tours = orig_sol
    zone_set = set(zone_list)
    while len(part_sol) < nb_nodes:
        start_list = list(zone_set - set(part_sol))
        # TODO filter start_list --> selective depth-wise exploration --> MCTS
        # TODO two-step lookahead
        # TODO parallelise this with multi-threads
        ret_list = [
            ppm_base(start_list, start_zone, ppm_model, no_context_panelty, 
                     cluster_weights=cluster_weights, opt_cycle=False)
            for start_zone in start_list
        ]
        seq_rewards = [seq_reward for (seq, seq_reward) in ret_list]
        max_seq_reward = np.max(seq_rewards)
        best_seq = part_sol + ret_list[np.argmax(seq_rewards)][0]
        #rollout_reward = ppm_model.query(best_seq[:-1], best_seq[-1], cluster_weights=cluster_weights)
        if (len(part_sol_tt_reward) == 0):
            rollout_reward = ppm_model.query(part_sol, best_seq[0], cluster_weights=cluster_weights) + max_seq_reward
        else:
            rollout_reward = np.sum(part_sol_tt_reward) + max_seq_reward
        if rollout_reward > tm_reward:
            # print(f'found a better rollout_reward: {rollout_reward}')
            tmp_good_tours = best_seq
            part_sol.append(tmp_good_tours[len(part_sol)])
            part_sol_tt_reward.append(ppm_model.query(part_sol[:-1], part_sol[-1], cluster_weights=cluster_weights))
            tm_reward = rollout_reward
            nb_explorations = 0
        else:
            # print('not as good as tm_reward')
            if nb_explorations < explore_budget:
                part_sol.append(tmp_good_tours[len(part_sol)])
                nb_explorations += 1
            else:
                part_sol = tmp_good_tours
                break
    return part_sol, tm_reward

def ppm_base(zone_list, start_zone, ppm_model, no_context_panelty, 
             cluster_weights=[0.25, 0.25, 0.25, 0.25], opt_cycle=True):
    """
    basic heuristics used repeatedly by rollout
    opt_cycle:  optimise for circle (tail to head is considered)
    """
    len_z = len(zone_list)
    if len_z == 1:
        return zone_list, ppm_model.query([], zone_list[0], cluster_weights=cluster_weights)
    if len_z == 2:
        if zone_list[0] != start_zone:
            zone_list[0], zone_list[1] = zone_list[1], zone_list[0]
        return zone_list, ppm_model.query([zone_list[0]], zone_list[1], cluster_weights=cluster_weights)
    if (opt_cycle):
        init_scores = []
        init_pairs = []
        init_tails = []
        # zone_list.remove(start_zone)
        for zone_tail in zone_list:
            if zone_tail == start_zone:
                continue
            for zone_head in zone_list:
                if zone_head == start_zone or zone_tail == zone_head:
                    continue
                init_ctx = [zone_tail, start_zone]
                symbol = zone_head
                init_pair = [start_zone, zone_head]
                init_score = ppm_model.query(init_ctx, symbol, 
                                            no_context_panelty=no_context_panelty,
                                            cluster_weights=cluster_weights)
                init_pairs.append(init_pair)
                init_scores.append(init_score)
                init_tails.append(zone_tail)
        if len(init_scores) == 0:
            print(zone_list)
            print("start_zone = ", start_zone)
            raise Exception("len(init_scores) == 0")
        init_idx = np.argmax(init_scores)
        pred_zone_seq = init_pairs[init_idx]
        pred_seq_prob = [init_scores[init_idx]]
        #final_zone_tail = init_tails[init_idx]
        # assert len(pred_zone_seq) == 2
        # rem_list = list(set(zone_list) - set(pred_zone_seq) - set([final_zone_tail]))
    else:
        pred_zone_seq = [start_zone]
        pred_seq_prob = []
    rem_list = list(set(zone_list) - set(pred_zone_seq))

    for _ in range(len_z - len(pred_zone_seq)):
        # print(pred_zone_seq)
        rem_scores = []
        for rem in rem_list:
            sc = ppm_model.query(pred_zone_seq, rem, cluster_weights=cluster_weights)
            rem_scores.append(sc)
        cand_idx = np.argmax(rem_scores)
        pred_zone_seq.append(rem_list[cand_idx])
        pred_seq_prob.append(rem_scores[cand_idx])
        del rem_list[cand_idx]
    # pred_zone_seq.append(final_zone_tail)
    return pred_zone_seq, np.sum(pred_seq_prob)
