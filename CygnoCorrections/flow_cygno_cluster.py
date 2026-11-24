import os

# these are to avoid creash in multithread on mac (2 conflicting libomp in the pool of threads for OpenMP)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import hashlib
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import yaml
from yaml import Loader

from data_reading.read_data_2D import *
from training.clusterTraining import (
    build_dataloader,
    forward_test,
    plot_training_history,
    test_training,
    train_model,
)
from training.validation import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onlycache",
        action="store_true",
        help="Run only the creation of the panda dataframes and cache them",
    )
    parser.add_argument(
        "--configuration",
        type=str,
        default="configuration_limerun4_v2",
        help="Key of the configuration in the flow yaml configuration file",
    )
    parser.add_argument(
        "--dscheck",
        action="store_true",
        help="do a sanity check of the conditioned dataset",
    )
    parser.add_argument(
        "--integrity",
        action="store_true",
        help="use cached clusters dataset, and do integrity tests",
    )
    parser.add_argument(
        "--fwdtest",
        action="store_true",
        help="do the fwd test of the CYGNO transport model",
    )
    parser.add_argument(
        "--train", action="store_true", help="train the correction"
    )
    parser.add_argument(
        "--test", action="store_true", help="train the correction"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="validate the correction by pure inference",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="give the path of a checkpoint.pt file containing a trained model (to start the training from a given point)",
    )
    args = parser.parse_args()

    # loop to read over network condigurations from the yaml file: - one way to do hyperparameter optimization
    with open("flow_configuration.yaml", "r") as stream:
        dictionary = yaml.load(stream, Loader)
    conf = args.configuration

    if args.onlycache:
        read_data_and_save(conf)

    inputfile = "data/cache/cygno_clusters_dataset.pkl"
    if args.integrity:
        integrity_tests(inputfile)

    if args.dscheck:
        dataset_sanity(inputfile)

    if args.fwdtest:
        forward_test(inputfile)

    outputmodel = (
        os.getcwd() + "/results/" + str(conf) + "/saved_states/best_model.pt"
    )
    if args.train:
        print("\n\t === TRAIN THE MODEL ===")
        if args.checkpoint:
            print(
                f"---> Starting the training from the pre-trained model saved in {args.checkpoint}"
            )
        model, history = train_model(inputfile, outputmodel, args.checkpoint)
        plot_training_history(history)
        print("\n\t === TEST THE MODEL ===")
        print(f"\nTest the trained model using the saved state in {outputmodel}")
        test_training(model, inputfile)

    if args.test:
        test_training(outputmodel, inputfile)

    if args.validate:

        sweep_vars = ["P", "T", "H"]

        # -----------------------
        # 0. Device setup
        # -----------------------
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        # -----------------------
        # 1. Dataset reading and metadata (singolo caricamento in memoria)
        # -----------------------
        print("Data loading...")
        batch_size = 256
        dataset, loader = build_dataloader(
            inputfile, batch_size=batch_size, is_test=True, num_workers=0
        )

        metadata = getattr(dataset, "metadata", None)
        if not metadata:
            with open(inputfile, "rb") as f:
                dataset_bundle = pickle.load(f)
            metadata = dataset_bundle.get("metadata", {})

        # -----------------------
        # 2. Load model & plot 10 random clusters (riusa dataset già in memoria)
        # -----------------------
        print(f"\nLoading model from:\n{outputmodel}")
        model = test_training(outputmodel, dataset, device=device)
        model.eval()

        unique_z = sorted(
            list(set([k[0] for k in metadata["keys"]["data_keys"]]))
        )
        print(
            f"====> Will make data/MC comparison for these z values: {unique_z}"
        )

        with torch.inference_mode():
            for z_val in unique_z:
                print(f"\n--- Data/MC for z = {z_val} cm... ---")
                central_sim_key = (z_val, 0.0214, 1450.0)  # z, alpha, lambda
                for var in sweep_vars:
                    print(
                        f"\n\t\t--> Testing variable {var} for median values of the other variables"
                    )
                    run_validation_sweep_from_dict(
                        model,
                        metadata,
                        loader,
                        inputfile,
                        central_sim_key,
                        sweep_var=var,
                        device=device,
                    )