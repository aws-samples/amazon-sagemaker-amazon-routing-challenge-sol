# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
import numpy as np

ZERO_KEY = "000"

class PPM(object):
    """
    PPM - Prediction by Partial Matching
    """

    def __init__(self, nb_order, vocab_size=8704):
        """
        an n-order Markov chain has n + 1 orders, starting from 0 to n.
        so if nb_order = 4, then we have order 0, 1, 2, 3, 4
        """
        self.nb_order = nb_order
        self.tables = []
        for i in range(nb_order + 1):
            self.tables.append(dict())
        tbl = self.tables[0]
        tbl[ZERO_KEY] = Ctx(ZERO_KEY)
        self.vocab_size = vocab_size
        self.neg_order_prob = np.log(1 / vocab_size)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            "("
            f"nb_order={self.nb_order}, "
            f"vocab_size={self.vocab_size}, "
            f"neg_order_prob={self.neg_order_prob}"
            ")"
        )

    def add_sequence(self, zone_list):
        """
        e.g.
        seq:  ['stz', 'C-17.3D', 'C-17.2D', 'C-17.1D', 'C-17.1E', 'C-17.2E', ..., 'C-18.1H', ..., 'C-18.2E']
        """
        zone_list += zone_list[:1]  # append the station (the first) as the last to form a cycle
        for order in range(self.nb_order + 1):
            tbl = self.tables[order]
            if order == 0:
                ctx = tbl[ZERO_KEY]
                for zone in zone_list:
                    ctx.add_entry(zone)
            else:
                for idx, _ in enumerate(zone_list[:-order]):
                    ctx_key = "|".join(zone_list[idx : idx + order])
                    entry = zone_list[idx + order]
                    if ctx_key in tbl:
                        ctx = tbl[ctx_key]
                    else:
                        ctx = Ctx(ctx_key)
                        tbl[ctx_key] = ctx
                    ctx.add_entry(entry)

    def query_zone_sequence(self, zone_list):
        """
        check the prob of the entire zone sequence
        """
        relist = []
        for i in range(len(zone_list) - 1):
            preced_zl = zone_list[:i + 1]
            following_zone = zone_list[i + 1]
            relist.append(self.query(preced_zl, following_zone))
        return np.sum(relist)
    
    def query(
        self,
        preceding_zone_list,
        following_zone,
        no_context_panelty=1.0,
        consider_hierarchy=True,
        cluster_weights=[0.25, 0.25, 0.25, 0.25],
    ):
        #C-17.3D
        if consider_hierarchy:
            # C
            pre_c_seq = [x[0] for x in preceding_zone_list]
            fol_c_z = following_zone[0]

            # 17 (or should it be C-17)
            pre_sc_seq = [x.split(".")[0].split("-")[-1] for x in preceding_zone_list]
            fol_sc_z = following_zone.split(".")[0].split("-")[-1]

            # 3D
            pre_ssc_seq = [x.split('.')[-1] for x in preceding_zone_list]
            fol_ssc_seq = following_zone.split('.')[-1]

            sc00 = self._query(preceding_zone_list, following_zone, no_context_panelty)
            sc01 = self._query(pre_c_seq, fol_c_z, no_context_panelty)
            sc02 = self._query(pre_sc_seq, fol_sc_z, no_context_panelty)
            sc03 = self._query(pre_ssc_seq, fol_ssc_seq, no_context_panelty)

            # return np.mean([sc00, sc01, sc02, sc03])
            cw = cluster_weights
            return sc00 * cw[0] + sc01 * cw[1] + sc02 * cw[2] + sc03 * cw[3]

        else:
            return self._query(preceding_zone_list, following_zone, no_context_panelty)

    def _query(self, preceding_zone_list, following_zone, no_context_panelty=1.0):
        """
        preceding_zone_list:    a list of strings (zones)
        following_zone:             string
        """
        lens = len(preceding_zone_list)
        no_ctx_pen = np.log(no_context_panelty)
        if lens > self.nb_order:
            preceding_zone_list = preceding_zone_list[lens - self.nb_order :]
            lens = self.nb_order
        penalty_probs = []
        # print('lens =', lens)
        for i in range(lens + 1):
            cur_order = lens - i
            # print('cur_order = ', cur_order)
            if cur_order > 0:
                ctx_key = "|".join(preceding_zone_list[i:])
            else:
                ctx_key = ZERO_KEY
            tbl = self.tables[cur_order]
            if ctx_key in tbl:
                ctx = tbl[ctx_key]
                if following_zone in ctx.entries:
                    return ctx.pc_prob(following_zone) + np.sum(penalty_probs)
                else:
                    # print(f'Symbol {following_zone} escapes from order {cur_order} w/ context - {ctx_key}')
                    penalty_probs.append(ctx.escape_prob)
            else:
                # no context found in this order,
                penalty_probs.append(no_ctx_pen)

        # deal witth order -1
        return self.neg_order_prob + np.sum(penalty_probs)


class Ctx(object):
    """ """

    def __init__(self, key):
        """
        special context is the order-0 context that has the "ZERO" key
        """
        self.key = key
        self.n = 0  # the number of times of this context has appeared
        self.entries = defaultdict(int)  # k - entry, v - count
        self._escape_prob = None

    def add_entry(self, entry):
        self.entries[entry] += 1
        self.n += 1

    @property
    def escape_prob(self):
        if self._escape_prob is None:
            self._escape_prob = np.log(self.d / (2 * self.n))
        return self._escape_prob

    def pc_prob(self, symbol):
        count = self.entries[symbol]
        return np.log((2 * count - 1) / (2 * self.n))

    @property
    def d(self):
        return len(self.entries)

def zone_sequence_gtset(gt_zone_seq, gt_zone_full_seq):
    nb_stops_seq = [] # record how many stops for each zone in zone_seq
    curr_nb_stops = 0
    last_zone = None
    for zone in gt_zone_full_seq:
        if (zone != last_zone):
            if (last_zone is not None):
                nb_stops_seq.append(curr_nb_stops) # end last
                curr_nb_stops = 0 # clear last
            last_zone = zone
        curr_nb_stops += 1
    nb_stops_seq.append(curr_nb_stops)
    #print(nb_stops_seq)
    #assert len(gt_zone_seq) == len(nb_stops_seq), f"{len(gt_zone_seq)} != {len(nb_stops_seq)}"
    
    zone_seq_gtset = []
    index_dict = dict()
    for idx, zone in enumerate(gt_zone_seq):
        if zone not in index_dict:
            index_dict[zone] = idx
        else:
            its_idx = index_dict[zone]
            if (nb_stops_seq[idx] > nb_stops_seq[its_idx]):
                index_dict[zone] = idx
    for idx, zone in enumerate(gt_zone_seq):
        if (idx == index_dict[zone]):
            zone_seq_gtset.append(zone)

    return zone_seq_gtset

def build_ppm_model(zdf, nb_orders, consider_hierarchy=True, gt_strictly_set=False, pass_in_model=None):
    """
    zdf = pd.read_csv(f'{zone_list_dir}/actual_zone-train.csv')

    gt_strictly_set: whether gt sequence is strictly a set
    """
    zone_collections = zdf
    if (pass_in_model):
        my_ppm = pass_in_model
    else:
        my_ppm = PPM(nb_orders)
    for idx, item in zone_collections.iterrows():
        gt_zone_seq = item.zone_seq.split("|")
        if (gt_strictly_set):
            #gt_zone_seq = list(dict.fromkeys(gt_zone_seq))
            gt_zone_seq = zone_sequence_gtset(gt_zone_seq, item.full_zone_seq.split("|"))
        # route_id = item.route_id
        # C-17.3D
        my_ppm.add_sequence(gt_zone_seq)
        if consider_hierarchy:
            # C
            gt_c_seq = [x[0] for x in gt_zone_seq]
            my_ppm.add_sequence(gt_c_seq)

            # 17
            gt_sc_seq = [x.split(".")[0].split("-")[-1] for x in gt_zone_seq]
            my_ppm.add_sequence(gt_sc_seq)

            # 3D
            gt_ssc_seq = [x.split('.')[-1] for x in gt_zone_seq]
            my_ppm.add_sequence(gt_ssc_seq)
    return my_ppm