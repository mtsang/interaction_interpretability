import sys

sys.path.append("models/autoint")
from model import AutoInt
import numpy as np
import os


class get_args:
    # the original parameter configuration of AutoInt
    blocks = 3
    block_shape = [64, 64, 64]
    heads = 2
    embedding_size = 16
    dropout_keep_prob = [1, 1, 1]
    epoch = 3
    batch_size = 1024
    learning_rate = 0.001
    learning_rate_wide = 0.001
    optimizer_type = "adam"
    l2_reg = 0.0
    random_seed = 2018  # used in the official autoint code
    loss_type = "logloss"
    verbose = 1
    run_times = 1
    is_save = False
    greater_is_better = False
    has_residual = True
    has_wide = False
    deep_layers = [400, 400]
    batch_norm = 0
    batch_norm_decay = 0.995

    def __init__(self, save_path, field_size, dataset, data_path):
        self.save_path = save_path
        self.field_size = field_size
        self.data = dataset
        self.data_path = data_path


def parse_args(dataset, data_path, save_path):
    dataset = dataset.lower()
    if "avazu" in dataset:
        field_size = 23
    elif "criteo" in dataset:
        field_size = 39
    else:
        raise ValueError("Invalid dataset")

    return get_args(save_path, field_size, dataset, data_path)


def get_data_info(args):
    data = args.data.split("/")[-1].lower()
    if any([data.startswith(d) for d in ["avazu"]]):
        file_name = ["train_i.npy", "train_x.npy", "train_y.npy"]
    elif any([data.startswith(d) for d in ["criteo"]]):
        file_name = ["train_i.npy", "train_x2.npy", "train_y.npy"]
    else:
        raise ValueError("invalid data arg")

    path_prefix = os.path.join(args.data_path, args.data)
    return file_name, path_prefix


def get_autoint_and_data(
    dataset="Criteo",
    data_path="/workspace/AutoInt",
    save_path="/test_code/Criteo/b3h2_dnn_dropkeep1_400x2/1/",
):
    args = parse_args(dataset, data_path, save_path)

    file_name = []

    file_name, path_prefix = get_data_info(args)
    feature_size = np.load(path_prefix + "/feature_size.npy")[0]

    run_cnt = 0
    model = AutoInt(args=args, feature_size=feature_size, run_cnt=run_cnt)

    Xi_valid = np.load(path_prefix + "/part2/" + file_name[0])
    Xv_valid = np.load(path_prefix + "/part2/" + file_name[1])
    y_valid = np.load(path_prefix + "/part2/" + file_name[2])

    feature_indices = list(range(Xi_valid.shape[1]))
    means_dict = {}
    for i in feature_indices:
        means_dict[i] = np.mean(Xv_valid[:, i])

    model.restore(args.save_path)
    return model, {"Xi": Xi_valid, "Xv": Xv_valid, "y": y_valid, "means": means_dict}
