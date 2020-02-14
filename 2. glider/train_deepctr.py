import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="model", default="WDL")
parser.add_argument("--runs", type=int, help="num trials", default=5)
parser.add_argument("--exp", type=str, help="experiment", default="baseline")
parser.add_argument("--ds", type=str, help="dataset", default="criteo")
parser.add_argument("--bs", type=int, help="batchsize", default=1024)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--lr", type=float, help="learning rate", default=0.01)
parser.add_argument("--opt", type=str, help="optimizer", default="adagrad")
parser.add_argument("--epochs", type=int, help="epochs", default=50)
parser.add_argument("--test_id", type=str, help="test_id", default="test1")
parser.add_argument("--emb_dim", type=int, help="size of embedding table", default=16)
parser.add_argument("--patience", type=int, help="patience", default=1)
parser.add_argument("--d_base", type=str, help="base data id", default="baseline")
parser.add_argument("--d_cross", type=str, help="cross data id", default="cross")
parser.add_argument("--d_cross_exp", type=str, help="cross exp", default="cross1")
parser.add_argument("--n_cross", type=int, help="num cross features", default=40)
parser.add_argument(
    "--epochs_skip_es",
    type=int,
    help="num of epochs to skip for checking early stopping",
    default=0,
)


args = parser.parse_args()

model_type = args.model  # ["WDL", "DeepFM", "DCN", "xDeepFM"]
num_trials = args.runs
experiment = args.exp  # ["baseline", "cross"]
dataset = args.ds  # ["criteo", "avazu"]
batch_size = args.bs
learning_rate = args.lr
opt = args.opt
gpu_device = args.gpu
epochs = args.epochs
test_id = args.test_id
emb_dim = args.emb_dim
patience = args.patience
base_data_id = args.d_base
cross_data_id = args.d_cross
cross_experiment = args.d_cross_exp
n_cross = args.n_cross
epochs_skip_es = args.epochs_skip_es

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)


import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.models import xDeepFM, DeepFM, WDL, DCN
from deepctr.inputs import SparseFeat, DenseFeat, get_fixlen_feature_names
from tensorflow.python.keras.models import save_model, load_model

from deepctr.layers import custom_objects
import pickle

import keras

import numpy as np
from tqdm import tqdm
import math

from os.path import join
import tensorflow as tf
from tensorflow.python.keras.optimizers import Adam, Adagrad

from tensorflow.keras.callbacks import EarlyStopping
from keras.backend.tensorflow_backend import set_session


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = (
    True
) 
sess = tf.Session(config=config)
set_session(sess)  

assert model_type in ["WDL", "DeepFM", "DCN", "xDeepFM"]
assert experiment in ["baseline", "cross"]
assert dataset in ["criteo", "avazu"]

if dataset == "criteo":
    src_datapath = "data/deepctr/criteo/"
    n_sparse = 26
    n_dense = 13

elif dataset == "avazu":
    src_datapath = "data/deepctr/avazu"
    n_sparse = 23
    n_dense = 0


sparse_features = ["C" + str(i) for i in range(1, n_sparse + 1)]
dense_features = ["I" + str(i) for i in range(1, n_dense + 1)]
target = ["label"]

if experiment == "cross":
    cross_features = ["G" + str(i) for i in range(1, n_cross + 1)]
else:
    cross_features = []


def get_labels(input_path, batch_size, target="label"):

    labels = {}
    for mode in ["valid", "test"]:
        label_batches = []
        for data_batch in tqdm(pd.read_hdf(input_path, key=mode, chunksize=batch_size)):
            label_batches.append(data_batch[target].values)
        labels[mode] = np.concatenate(label_batches)
    return labels


labels = get_labels(src_datapath + "/" + base_data_id + ".h5", int(1e5))


with open(
    join(src_datapath, "n_unique_dict_" + base_data_id + ".pickle"), "rb"
) as handle:
    unique_dict_baseline = pickle.load(handle)

unique_dict = dict()
for key in sparse_features:
    unique_dict[key] = unique_dict_baseline[key]

if experiment == "cross":
    with open(
        join(
            src_datapath, cross_experiment, "n_unique_dict_" + cross_data_id + ".pickle"
        ),
        "rb",
    ) as handle:
        unique_dict_cross = pickle.load(handle)
    for key in cross_features:
        unique_dict[key] = unique_dict_cross[key]


fixlen_feature_columns = [
    SparseFeat(feat, unique_dict[feat]) for feat in sparse_features + cross_features
] + [DenseFeat(feat, 1) for feat in dense_features]

dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns

fixlen_feature_names = get_fixlen_feature_names(
    linear_feature_columns + dnn_feature_columns
)


def get_data_generator(
    base_path,
    cross_path,
    model_inputs,
    batch_size,
    mode="train",
    target="label",
    keras=False,
):
    while True:
        i = 0
        while True:
            data_batch_baseline = pd.read_hdf(
                base_path, key=mode, start=i * batch_size, stop=(i + 1) * batch_size
            )
            if cross_path:
                data_batch_cross = pd.read_hdf(
                    cross_path,
                    key=mode,
                    start=i * batch_size,
                    stop=(i + 1) * batch_size,
                )
            i += 1
            if data_batch_baseline.shape[0] == 0:
                break
            data_batch = (
                pd.concat([data_batch_baseline, data_batch_cross], axis=1)
                if cross_path
                else data_batch_baseline
            )
            X = [data_batch[name] for name in model_inputs]
            Y = data_batch[target].values
            yield (X, Y)
        if not keras:
            break


base_path = join(src_datapath, base_data_id + ".h5")
cross_path = (
    join(src_datapath, cross_experiment, cross_data_id + "_" + str(n_cross) + ".h5")
    if experiment == "cross"
    else ""
)
cross_experiment = cross_experiment if experiment == "cross" else "baseline"


def shuffle_batch(X, y=None, seed=None):
    if seed is not None:
        np.random.seed(seed)

    indices = np.random.permutation(len(X[0]))

    X_shuff = []
    for i in range(len(X)):
        X_shuff.append(X[i].iloc[indices])

    if y is not None:
        y_shuff = y[indices]
        return X_shuff, y_shuff
    else:
        return X_shuff


exp_folder = join("experiments", "deepctr", test_id, "checkpoints")

if not os.path.exists(exp_folder):
    os.makedirs(exp_folder)


pkl_path = join(
    "experiments",
    "deepctr",
    test_id,
    dataset + "_" + model_type + "_" + cross_experiment + ".pkl",
)

if os.path.exists(pkl_path):
    with open(pkl_path, "rb") as handle:
        results_dict = pickle.load(handle)
    histories = results_dict["histories"]
    histories_val = results_dict["val_loss"]
    test_performances = results_dict["test_performances"]
    checkpoints = results_dict["checkpoints"]
else:
    histories = []
    histories_val = []
    test_performances = []
    checkpoints = []


for i in range(num_trials):
    if i < len(histories):
        continue

    print("Starting trial", i + 1)

    model_checkpoint_file = join(
        "experiments",
        "deepctr",
        test_id,
        "checkpoints",
        dataset + "_" + model_type + "_" + cross_experiment + "_trial" + str(i) + ".h5",
    )

    test_generator = get_data_generator(
        base_path, cross_path, fixlen_feature_names, batch_size, mode="test", keras=True
    )

    if model_type == "DeepFM":
        model = DeepFM(
            linear_feature_columns,
            dnn_feature_columns,
            task="binary",
            embedding_size=emb_dim,
            use_fm=True,
            dnn_hidden_units=[400, 400, 400],
        )

    if model_type == "xDeepFM":
        model = xDeepFM(
            linear_feature_columns,
            dnn_feature_columns,
            task="binary",
            embedding_size=emb_dim,
            dnn_hidden_units=[400, 400],
            cin_layer_size=[200, 200, 200],
        )

    if model_type == "WDL":
        model = WDL(
            linear_feature_columns,
            dnn_feature_columns,
            task="binary",
            embedding_size=emb_dim,
            dnn_hidden_units=[1024, 512, 256],
        )

    if model_type == "DCN":
        model = DCN(
            dnn_feature_columns,
            task="binary",
            embedding_size=emb_dim,
            dnn_hidden_units=[1024, 1024],
            cross_num=6,
        )

    if opt == "adagrad":
        optimizer = Adagrad
    elif opt == "adam":
        optimizer = Adam
    else:
        raise ValueError("Invalid optimizer")

    model.compile(
        optimizer(learning_rate), "binary_crossentropy", metrics=["binary_crossentropy"]
    )

    callbacks = []

    patience_counter = 0
    best_valid_loss = float("Inf")

    history_epoch = {}
    history_val = {}
    for epoch in range(epochs):
        breakout = False
        history_epoch[epoch] = {}
        history_val[epoch] = []
        train_generator = get_data_generator(
            base_path,
            cross_path,
            fixlen_feature_names,
            len(labels["valid"]),
            mode="train",
        )
        for file_count, data_batch in enumerate(train_generator):
            print("epoch", epoch, "filecount", file_count)
            train_model_input, train_model_labels = data_batch

            X_shuffled, Y_shuffled = shuffle_batch(
                train_model_input, train_model_labels
            )  # using AutoInt's convention

            history = model.fit(
                X_shuffled,
                Y_shuffled,
                batch_size=batch_size,
                epochs=1,
                verbose=1,
                callbacks=callbacks,
            )

            history_epoch[epoch][file_count] = [history.history, history.params]

            if epoch < epochs_skip_es:
                continue

            valid_generator = get_data_generator(
                base_path,
                cross_path,
                fixlen_feature_names,
                batch_size,
                mode="valid",
                keras=True,
            )
            valid_pred = model.predict_generator(
                valid_generator, steps=math.ceil(len(labels["valid"]) / batch_size)
            )
            valid_loss = log_loss(labels["valid"], valid_pred, eps=1e-7)
            history_val[epoch].append(valid_loss)

            if valid_loss < best_valid_loss:
                save_model(model, model_checkpoint_file)

                print(
                    "[%d-%d] model saved!. Valid loss improved from %.4f to %.4f"
                    % (epoch, file_count, best_valid_loss, valid_loss)
                )
                best_valid_loss = valid_loss
                patience_counter = 0
            else:
                if patience_counter >= patience:
                    breakout = True
                    print("Early Stopping!")
                    break
                patience_counter += 1

        if breakout:
            break

    best_model = tf.keras.models.load_model(model_checkpoint_file, custom_objects)

    pred_ans = best_model.predict_generator(
        test_generator, steps=math.ceil(len(labels["test"]) / batch_size)
    )

    test_logloss = round(log_loss(labels["test"], pred_ans, eps=1e-7), 7)
    test_auc = round(roc_auc_score(labels["test"], pred_ans), 7)
    print("test LogLoss", test_logloss)
    print("test AUC", test_auc)

    histories.append(history_epoch)
    test_performances.append({"logloss": test_logloss, "auc": test_auc})
    histories_val.append(
        {
            "history": history_val,
            "best_valid_loss": best_valid_loss,
            "patience": patience,
        }
    )
    checkpoints.append(model_checkpoint_file)
    results_dict = {
        "histories": histories,
        "test_performances": test_performances,
        "val_loss": histories_val,
        "params": best_model.count_params(),
        "checkpoints": checkpoints,
    }

    with open(pkl_path, "wb") as handle:
        pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("\n\n")

aucs = [x["auc"] for x in test_performances]
loglosses = [x["logloss"] for x in test_performances]

auc_mean = np.mean(aucs)
auc_std = np.std(aucs)
logloss_mean = np.mean(loglosses)
logloss_std = np.std(loglosses)

print(auc_mean, auc_std, logloss_mean, logloss_std)
