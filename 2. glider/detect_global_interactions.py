import sys

sys.path.append("../1. madex")
from sampling_and_inference import generate_perturbation_dataset_autoint
from neural_interaction_detection import detect_interactions
import os
import logging
from tqdm import tqdm
import warnings
import pickle
import numpy as np
import argparse
import torch
import torch.optim as optim
from utils.global_interaction_utils import *
import torch.multiprocessing as multiprocessing


warnings.simplefilter("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--global_size", type=int, default=1000)
parser.add_argument("--num_perturbation", type=int, default=6000)
parser.add_argument(
    "--save_path",
    type=str,
    default="/meladyfs/newyork/mtsang/AutoInt/test_code/Criteo/b3h2_dnn_dropkeep1_400x2_5trials_v2/1/",
)
parser.add_argument("--data", type=str, help="data name", default="criteo")
parser.add_argument("--save_id", type=str, help="save id", default="testpar2")
parser.add_argument(
    "--data_path", type=str, help="root path for all the data", default="data/autoint"
)
parser.add_argument("--epochs", type=int, help="num epochs", default=100)
parser.add_argument("--es", type=int, help="enable early stopping", default=1)
parser.add_argument("--l1", type=float, help="set l1 reg constant", default=1e-4)
parser.add_argument("--lr", type=float, help="learning rate", default=0.01)
parser.add_argument("--opt", type=str, help="optimizer", default="adam")
parser.add_argument(
    "--par_batch_size",
    type=int,
    help="size of parallel batch (same as num parallel processes)",
    default=32,
)
parser.add_argument("--add_linear", type=int, help="contain main effects in interaction detector via linear regression", default=0)
parser.add_argument("--detector", type=str, help="detector: NID or GradientNID", default="NID")
parser.add_argument("--gpu", type=int, help="gpu number", default=0)

args = parser.parse_args()
par_batch_size = args.par_batch_size
if args.opt == "adagrad":
    opt = optim.Adagrad
elif args.opt == "adam":
    opt = optim.Adam
else:
    raise ValueError("invalid optimizer")

# device = torch.device("cuda:" + str(args.gpu))


def par_experiment(idx, perturbations):
    feats = perturbations["feats"]
    labels = perturbations["targets"]

#     distributes processes across two gpus
    device = torch.device("cuda:" + str(idx%2))

    try:
        inters, mlp_loss = detect_interactions(
            feats,
            labels,
            arch=[256, 128, 64],
            nepochs=args.epochs,
            early_stopping=args.es,
            patience=5,
            l1_const=args.l1,
            learning_rate=args.lr,
            opt_func=opt,
            add_linear=args.add_linear,
            detector=args.detector,
            seed=42,
            verbose=False,
            device=device,
        )
        print("mlp loss", mlp_loss)
        result = {"inters": inters, "mlp_loss": mlp_loss}
    except:
        print("error in learning mlp for interaction detection")
        result = None

    return idx, result


def run():
    multiprocessing.set_start_method("spawn", force=True)

    # this data is shuffled. other datasets must be shuffled for global interaction detection
    model, data = get_autoint_and_data(
        data_path=args.data_path, dataset=args.data, save_path=args.save_path
    )

    dense_feat_indices = []
    sparse_feat_indices = []
    for i in tqdm(range(data["Xi"].shape[1])):
        uniq = np.unique(data["Xi"][:, i])
        if len(uniq) == 1 and not args.data == "avazu":
            dense_feat_indices.append(i)
        else:
            sparse_feat_indices.append(i)

    print("dense feature indices", dense_feat_indices)

    save_postfix = "_" + args.save_id if args.save_id else ""

    base_path = "experiments/detected_interactions/"
    pkl_path = (
        base_path
        + "detected_interactions_"
        + args.data.lower()
        + save_postfix
        + ".pickle"
    )
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as handle:
            interaction_results = pickle.load(handle)
        print("loaded existing results. starting from index", len(interaction_results))
    else:
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        interaction_results = []

    indexes = list(range(len(interaction_results), args.global_size))
    num_par_batches = int(np.ceil(len(indexes) / par_batch_size))

    for b in tqdm(range(num_par_batches)):
        index_batch = indexes[b * par_batch_size : (b + 1) * par_batch_size]
        perturbation_batch = []
        for idx in index_batch:

            data_inst = {
                "Xi": data["Xi"][idx],
                "Xv": data["Xv"][idx],
                "means": data["means"],
            }
            feats, targets = generate_perturbation_dataset_autoint(
                data_inst,
                model,
                dense_feat_indices,
                sparse_feat_indices,
                num_samples=args.num_perturbation,
                valid_size=500,
                test_size=500,
                seed=idx,
            )
            perturbation_batch.append({"feats": feats, "targets": targets})

        with multiprocessing.Pool(processes=par_batch_size) as pool:
            results_batch = pool.starmap(
                par_experiment, zip(index_batch, perturbation_batch)
            )

        results_batch.sort(key=lambda x: x[0])

        for _, result in results_batch:
            interaction_results.append(result)

        with open(pkl_path, "wb") as handle:
            pickle.dump(interaction_results, handle)


if __name__ == "__main__":
    run()
