import pickle
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import multiprocessing as mp
from itertools import repeat
import warnings

warnings.simplefilter("ignore")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def load_global_interactions(interactions_file, field_size, max_rank, prune_subsets, top_k):

    with open(interactions_file, "rb") as handle:
        interaction_results = pickle.load(handle, encoding="latin1")

    global_interactions = {}
    mlp_losses = []
    inters = []
    for result in interaction_results:
        if result is None:
            continue
        for inter in result["inters"][:top_k]:
            if len(inter[0]) == field_size:
                continue
            if inter[0] not in global_interactions:
                global_interactions[inter[0]] = 1
            else:
                global_interactions[inter[0]] += 1

    global_interactions = sorted(
        global_interactions.items(), key=lambda x: x[1], reverse=True
    )

    if prune_subsets:
        pruned_global_interactions = []
        index = 0
        while len(pruned_global_interactions) < max_rank:
            inter = global_interactions[index]
            if any(
                set(inter[0]) < set(new_inter[0])
                for new_inter in pruned_global_interactions
            ):
                pass
            else:
                pruned_global_interactions.append(inter)
                pruned_global_interactions = [
                    t
                    for t in pruned_global_interactions
                    if not (set(t[0]) < set(inter[0]))
                ]
            index += 1
    else:
        pruned_global_interactions = global_interactions[:max_rank]

    top_K_inters, _ = zip(*pruned_global_interactions)
    return top_K_inters


def load_data_autoint(dataset, data_path):

    path_prefix = data_path + "/"

    if dataset == "criteo":
        file_name = ["train_i.npy", "train_x2.npy", "train_y.npy"]
    elif dataset == "avazu":
        file_name = ["train_i.npy", "train_x.npy", "train_y.npy"]
    else:
        raise ValueError("Invalid dataset")

    data = []
    for j in tqdm(range(1, 11)):
        folder_path = path_prefix + "/part" + str(j) + "/"
        Xi = np.load(folder_path + file_name[0])
        Xv = np.load(folder_path + file_name[1])
        y = np.load(folder_path + file_name[2])
        data.append({"Xi": Xi, "Xv": Xv, "y": y})

    return data


def merge_data(data):
    Xi = []
    Xv = []
    y = []
    lens = []
    for d in data:
        Xi.append(d["Xi"])
        Xv.append(d["Xv"])
        y.append(d["y"])
        lens.append(len(d["y"]))
    Xi = np.concatenate(Xi)
    Xv = np.concatenate(Xv)
    y = np.concatenate(y)
    return Xi, Xv, y, lens


def get_training_batch(data, size=1000000):
    Xi_batch = data[2]["Xi"][:size]
    Xv_batch = data[2]["Xv"][:size]
    y_batch = data[2]["y"][:size]
    return Xi_batch, Xv_batch, y_batch


def get_dense_sparse_feat_indices(Xi_batch, dataset):

    dense_feat_indices = []
    sparse_feat_indices = []
    for i in tqdm(range(Xi_batch.shape[1])):
        uniq = np.unique(Xi_batch[:, i])
        if len(uniq) == 1 and "avazu" not in dataset:
            dense_feat_indices.append(i)
        else:
            sparse_feat_indices.append(i)

    return dense_feat_indices, sparse_feat_indices


def discretize_dense(Xv_feat, Xv_feat_batch, num_bins):
    est = KBinsDiscretizer(n_bins=num_bins, encode="ordinal", strategy="quantile")
    est.fit(Xv_feat_batch)
    disc = est.transform(Xv_feat)
    cardinality = len(est.bin_edges_[0]) - 1
    return disc, est, cardinality


def _par_discretize(f_idx, Xv_feat, Xv_feat_batch, num_bins):
    #     print("start", f_idx)
    disc, est, cardinality = discretize_dense(Xv_feat, Xv_feat_batch, num_bins)
    return f_idx, disc, est, cardinality


def discretize_dense_features(
    Xi,
    Xv,
    Xv_batch,
    dense_feat_indices,
    sparse_feat_indices,
    num_feats,
    num_bins,
    num_processes=20,
):

    discretizers = {}
    new_Xv_dense = []
    cardinalities = {}

    Xv_feats = []
    Xv_feats_batch = []
    for i in dense_feat_indices:
        Xv_feats.append(Xv[:, i].reshape(-1, 1))
        Xv_feats_batch.append(Xv_batch[:, i].reshape(-1, 1))

    pool = mp.Pool(processes=num_processes)
    disc_collect = pool.starmap(
        _par_discretize,
        zip(dense_feat_indices, Xv_feats, Xv_feats_batch, repeat(num_bins)),
    )
    cardinalities = {}
    discretizers = {}
    disc_summary = [x[1:] for x in sorted(disc_collect, key=lambda x: x[0])]
    new_Xv_dense = []
    for i, disc in enumerate(disc_summary):
        new_Xv_dense.append(disc[0])
        discretizers[i] = disc[1]
        cardinalities[i] = disc[2]

    if dense_feat_indices:
        new_Xv_dense = np.concatenate(new_Xv_dense, 1)
        den = True
    else:
        den = False

    pool.close()

    if den:
        sparsified_data = np.zeros((Xi.shape[0], num_feats))
        sparsified_data[:, dense_feat_indices] = new_Xv_dense
        sparsified_data[:, sparse_feat_indices] = Xi[:, sparse_feat_indices]
    else:
        sparsified_data = Xi

    return sparsified_data


def zero_index_sp_feats(combo_map, feat_combo):
    new_i = []
    new_v = []
    for c in feat_combo:
        if tuple(c) not in combo_map:
            new_i.append(0)
            new_v.append(0)
        else:
            new_i.append(combo_map[tuple(c)])
            new_v.append(1)

    return new_i, new_v


def _par_zero_sp(combo_idx, combo_map, feat_combo):
    #     print(combo_idx, feat_combo.shape)
    new_i, new_v = zero_index_sp_feats(combo_map, feat_combo)
    return combo_idx, new_i, new_v


def cross_sparse_features(
    top_K_inters,
    sparsified_data,
    sparsified_batch,
    Xi_batch,
    threshold,
    num_processes=20,
):

    # collect combo frequency
    inter_feats = []
    inter_combo_maps = {}

    for inter in tqdm(top_K_inters):

        inter_list = list(inter)
        inter_counts = {}
        for d, data_inst in enumerate(sparsified_batch):
            combo = tuple(data_inst[inter_list])
            if combo not in inter_counts:
                inter_counts[combo] = 1
            else:
                inter_counts[combo] += 1
        combo_map = {}
        for combo in inter_counts:
            if inter_counts[combo] <= Xi_batch.shape[0] * threshold:
                pass
            else:
                orig_len = len(combo_map)
                combo_map[combo] = orig_len + 1  # shift by 1 (0 value means missing)
        inter_combo_maps[inter] = combo_map

    #  f = open("b" + str(nbins),"w")
    #  for cm in inter_combo_maps:
    #      f.write(str(cm) + "\t" + str(len(inter_combo_maps[cm])) + "\n")
    #      print(len(inter_combo_maps[cm]))

    inters = []
    combo_maps = []
    feat_combos = []
    for inter in tqdm(inter_combo_maps):
        inters.append(inter)
        combo_maps.append(inter_combo_maps[inter])
        feat_combos.append(sparsified_data[:, list(inter)])

    pool = mp.Pool(processes=num_processes)
    cross_feats = pool.starmap(
        _par_zero_sp, zip(list(range(len(combo_maps))), combo_maps, feat_combos)
    )
    del feat_combos
    pool.close()
    cross_feats = sorted(cross_feats, key=lambda x: x[0])

    ##   the serial way of obtaining cross_feats (in case the parallel code doesnt work..)
    #     cross_feats = []
    #     i = 0
    #     for inter in tqdm(inter_combo_maps):
    #         cross_feats.append(_par_zero_sp(i, inter_combo_maps[inter], sparsified_data[:,list(inter)]))
    #         i += 1

    return cross_feats


def get_X_cross(inters, cross_feats, Xi, Xv, sparse_feat_indices):
    # get cross feats in autoint data format

    # if a feature value is 0 , e.g. missing data, then any interaction with this feature will also be deemed missing with value 0
    n = 0
    Xv_cross = []
    for inter in tqdm(inters):
        mask = np.ones(Xv.shape[0])
        for i, idx in enumerate(inter):
            if idx in sparse_feat_indices:
                mask = mask * Xv[:, idx]

        Xv_cross.append(cross_feats[n][2] * mask)
        n += 1

    cross_start = Xi.max() + 1

    # shift all the cross feature values so they can be packed later into a single embedding matrix (autoint format)
    for i in tqdm(range(len(cross_feats))):
        cur_cross_feat = np.array(cross_feats[i][1])
        max_val = cur_cross_feat.max()
        cross_feats[i] = (
            cur_cross_feat + cross_start
        )  # in-place modification to save memory
        cross_start = max_val + cross_start + 1

    Xi_cross = cross_feats

    Xi_cross = np.stack(Xi_cross, 1)
    Xv_cross = np.stack(Xv_cross, 1)

    return Xi, Xv, Xi_cross, Xv_cross
