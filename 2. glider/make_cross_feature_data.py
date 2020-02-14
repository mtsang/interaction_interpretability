import pickle
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
import os
import pandas as pd
import argparse

warnings.simplefilter("ignore")

from utils.cross_feature_utils import *


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_file",
    type=str,
    help="the path where global interaction results are saved",
    default="experiments/detected_interactions_criteo_repr2.pickle",
)
parser.add_argument("--exp", type=str, help="an experiment id", default="cross2_K20")
parser.add_argument(
    "--K", type=int, help="the top-K threshold for global interactions", default=20
)
parser.add_argument("--data", type=str, help="data name", default="criteo")
parser.add_argument(
    "--autoint_save_path",
    type=str,
    help="folder where cross features for autoint are saved",
    default="data/autoint/criteo",
)
parser.add_argument(
    "--deepctr_save_path",
    type=str,
    help="folder where cross features for deepctr are saved",
    default="data/deepctr/criteo",
)
parser.add_argument(
    "--bs", type=int, help="batch size of training data", default=1000000
)
parser.add_argument("--nbins", type=int, help="num bins", default=100)
parser.add_argument(
    "--nprocs", type=int, help="number of parallel processes", default=20
)
parser.add_argument(
    "--thresh",
    type=float,
    help="min pct of training batch to require cross feature ids to appear",
    default=0.0001,
)
parser.add_argument(
    "--top_k",
    type=int,
    help="k threshold for madex interactions",
    default=100,
)
parser.add_argument(
    "--save_base_data",
    type=str2bool,
    help="y/n: save (baseline) data without cross features for deepctr",
    nargs="?",
    const=True,
    default=True,
)
parser.add_argument(
    "--prune",
    type=str2bool,
    help="prune interaction subsets",
    nargs="?",
    const=True,
    default=True,
)

args = parser.parse_args()

interactions_file = args.data_file
experiment = args.exp
max_rank = args.K
dataset = (args.data).lower()
data_path_autoint = args.autoint_save_path
data_path_deepctr = args.deepctr_save_path

training_batch_size = args.bs
num_bins = args.nbins
num_processes = args.nprocs
threshold_pct = args.thresh
deepctr_save_baseline_data = args.save_base_data
prune_interaction_subsets = args.prune
top_k = args.top_k


def make_cross_feature_data(
    interactions_file,
    max_rank,
    dataset,
    training_batch_size,
    data_path,
    num_bins,
    threshold,
    top_k,
    prune_subsets,
    num_processes,
):

    print("loading autoint data")
    data = load_data_autoint(dataset, data_path)
    Xi, Xv, y, lens = merge_data(data)
    Xi_batch, Xv_batch, y_batch = get_training_batch(data, size=training_batch_size)
    dense_feat_indices, sparse_feat_indices = get_dense_sparse_feat_indices(
        Xi_batch, dataset
    )

    #     print("dense feature indices", dense_feat_indices)

    if dataset == "avazu":
        num_sparse, num_dense = 23, 0
    elif dataset == "criteo":
        num_sparse, num_dense = 26, 13
    else:
        raise ValueError("Invalid dataset")

    assert num_dense == len(dense_feat_indices)
    assert num_sparse == len(sparse_feat_indices)

    num_feats = len(sparse_feat_indices) + len(dense_feat_indices)

    print("loading interactions")
    inters = load_global_interactions(
        interactions_file, num_feats, max_rank, prune_subsets, top_k
    )

    print("discretizing dense features")
    sparsified_data = discretize_dense_features(
        Xi,
        Xv,
        Xv_batch,
        dense_feat_indices,
        sparse_feat_indices,
        num_feats,
        num_bins,
        num_processes=num_processes,
    )

    train_start = sum(lens[0:2])
    sparsified_batch = sparsified_data[train_start : train_start + training_batch_size]

    print("crossing sparse features")
    cross_feats = cross_sparse_features(
        inters,
        sparsified_data,
        sparsified_batch,
        Xi_batch,
        threshold,
        num_processes=num_processes,
    )
    Xi, Xv, Xi_cross, Xv_cross = get_X_cross(
        inters, cross_feats, Xi, Xv, sparse_feat_indices
    )

    return (
        Xi,
        Xv,
        y,
        Xi_cross,
        Xv_cross,
        lens,
        max_rank,
        num_feats,
        dense_feat_indices,
        sparse_feat_indices,
    )


def save_cross_feats_autoint(Xi_cross, Xv_cross, lens, experiment, data_path):
    print("saving data for autoint")

    cross_name = ["i_cross.npy", "x_cross.npy"]

    prev_len = 0
    for i in tqdm(range(1, 11)):
        cur_len = prev_len + lens[i - 1]
        Xi_seg = Xi_cross[prev_len:cur_len]
        Xv_seg = Xv_cross[prev_len:cur_len]

        folder_path = data_path + "/part" + str(i) + "/" + experiment
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + "/" + cross_name[0], Xi_seg)
        np.save(folder_path + "/" + cross_name[1], Xv_seg)

        prev_len = cur_len

    feature_size = int(Xi_cross.max() + 1)
    #     print("feature_size = %d" % feature_size)

    folder_path2 = data_path + "/" + experiment

    if not os.path.exists(folder_path2):
        os.makedirs(folder_path2)
    np.save(folder_path2 + "/feature_size.npy", np.array([feature_size]))


def save_cross_feats_deepctr(
    Xi,
    Xv,
    y,
    Xi_cross,
    Xv_cross,
    dense_feat_indices,
    sparse_feat_indices,
    lens,
    experiment,
    data_path,
    deepctr_save_baseline_data,
):
    print("saving data for deepctr")
    num_dense = len(dense_feat_indices)
    num_sparse = len(sparse_feat_indices)

    sparse_features = ["C" + str(i) for i in range(1, num_sparse + 1)]
    dense_features = ["I" + str(i) for i in range(1, num_dense + 1)]
    cross_features = ["G" + str(i) for i in range(1, max_rank + 1)]
    target = ["label"]

    settings = ["cross"]
    if deepctr_save_baseline_data:
        settings.append("baseline")

    n_unique_dict = dict()

    for setting in settings:

        if setting == "baseline":
            sparse_indices = sparse_feat_indices
            Xi_sparse_na = np.where(
                Xv[:, sparse_indices] == 1, Xi[:, sparse_indices], -1
            )
            data_np = np.concatenate(
                [np.expand_dims(y, axis=1), Xv[:, dense_feat_indices], Xi_sparse_na],
                axis=1,
            )
            df = pd.DataFrame(
                data_np, columns=target + dense_features + sparse_features
            )
            temp_feats = sparse_features
            save_path = data_path
            postfix = ""
        else:
            sparse_indices = list(range(max_rank))
            Xi_sparse_na = np.where(
                Xv_cross[:, sparse_indices] == 1, Xi_cross[:, sparse_indices], -1
            )
            data_np = Xi_sparse_na
            df = pd.DataFrame(data_np, columns=cross_features)
            temp_feats = cross_features
            save_path = data_path + "/" + experiment
            postfix = "_" + str(max_rank)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for feat in tqdm(temp_feats):
            lbe = LabelEncoder()
            df[feat] = lbe.fit_transform(
                df[feat]
            )  # global label encoding consistent with autoint's data preprocessing

        train = df[sum(lens[0:2]) :]
        valid = df[lens[0] : sum(lens[0:2])]
        test = df[0 : lens[0]]

        train.to_hdf(
            save_path + "/" + setting + postfix + ".h5",
            key="train",
            format="table",
            model="w",
        )
        valid.to_hdf(
            save_path + "/" + setting + postfix + ".h5", key="valid", format="table"
        )
        test.to_hdf(
            save_path + "/" + setting + postfix + ".h5", key="test", format="table"
        )

        for feat in tqdm(temp_feats):
            n_unique_dict[feat] = df[feat].nunique()

    n_unique_cross_dict = dict()
    for feat in cross_features:
        n_unique_cross_dict[feat] = n_unique_dict[feat]

    if deepctr_save_baseline_data:
        n_unique_baseline_dict = dict()
        for feat in sparse_features:
            n_unique_baseline_dict[feat] = n_unique_dict[feat]
        with open(data_path + "/n_unique_dict_baseline.pickle", "wb") as handle:
            pickle.dump(
                n_unique_baseline_dict, handle, protocol=pickle.HIGHEST_PROTOCOL
            )

    with open(
        data_path + "/" + experiment + "/n_unique_dict_cross.pickle", "wb"
    ) as handle:
        pickle.dump(n_unique_cross_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    print("warning: this process may take several hours and significant RAM (>150GB)")

    Xi, Xv, y, Xi_cross, Xv_cross, lens, max_rank, num_feats, dense_feat_indices, sparse_feat_indices = make_cross_feature_data(
        interactions_file,
        max_rank,
        dataset,
        training_batch_size,
        data_path_autoint,
        num_bins,
        threshold_pct,
        top_k,
        prune_interaction_subsets,
        num_processes,
    )

    save_cross_feats_autoint(Xi_cross, Xv_cross, lens, experiment, data_path_autoint)
    save_cross_feats_deepctr(
        Xi,
        Xv,
        y,
        Xi_cross,
        Xv_cross,
        dense_feat_indices,
        sparse_feat_indices,
        lens,
        experiment,
        data_path_deepctr,
        deepctr_save_baseline_data,
    )
