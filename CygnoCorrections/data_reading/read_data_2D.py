# Loads and treat the reconstruction tree data and simulation
# test basic plotting with python -m data_reading.read_data

import argparse
from collections import defaultdict
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import pickle
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import uproot
import yaml
from yaml import Loader

from data_reading.cluster import *
from data_reading.clusterDataset import ConditionalClusterDataset

selection_cfg = {
    "integral_min": 5000,
    "integral_max": 100000,
    "x_min": 500,
    "x_max": 2000,
    "y_min": 500,
    "y_max": 2000,
    "min_npix": 500,
    "n_hits": 200,
    # additional (not used for training)
    "length_min": 0,
    "length_max": 1e10,
    "slimness_min": 0.0,
}


def process_single_file_task(
    root_file,
    mapkey,
    all_cluster_variables,
    isdata,
    selection_cfg,
    chunks_dir,
):
    """
    Scarica, processa e salva in cache un singolo file ROOT in modo indipendente.
    """
    file_id = hashlib.md5(f"{mapkey}_{root_file}".encode()).hexdigest()
    chunk_path = os.path.join(chunks_dir, f"chunk_{file_id}.pkl")

    if os.path.exists(chunk_path):
        return mapkey, isdata, chunk_path, root_file

    local_file = root_file
    tmp_dir = None
    force_local_staging = True

    try:
        if force_local_staging:
            tmp_dir = tempfile.mkdtemp()
            local_file = os.path.join(tmp_dir, root_file.split("/")[-1])

            if root_file.startswith("http://") or root_file.startswith(
                "https://"
            ):
                try:
                    urllib.request.urlretrieve(root_file, local_file)
                except urllib.error.URLError as e:
                    raise RuntimeError(
                        f"Impossibile scaricare l'URL S3 {root_file}: {e}"
                    )
            elif ":" in root_file and not root_file.startswith(("C:", "D:")):
                subprocess.run(
                    ["rclone", "copyto", root_file, local_file], check=True
                )
            else:
                shutil.copy2(root_file, local_file)

        clusters = build_clusters_from_root_file(
            local_file,
            scalar_vars=all_cluster_variables,
            conditions=mapkey,
            isdata=isdata,
            selection_cfg=selection_cfg,
        )

        with open(chunk_path, "wb") as f:
            pickle.dump(clusters, f, protocol=4)

    finally:
        if force_local_staging and os.path.exists(local_file):
            os.remove(local_file)
        if force_local_staging and tmp_dir and os.path.exists(tmp_dir):
            os.rmdir(tmp_dir)

    return mapkey, isdata, chunk_path, root_file


def read_data_and_save(conf, reuse_sim_from=None, out_pkl=None):

    with open("flow_configuration.yaml", "r") as stream:
        dictionary = yaml.load(stream, Loader)

    with open("cluster_training_list.json", "r") as file:
        json_data = json.load(file)

    all_cluster_variables = json_data["all_cluster_variables"]
    print(
        "List of all the variables that should be attached to the cluster object:   ",
        all_cluster_variables,
    )
    target_cluster_variables = json_data["target_cluster_variables"]
    print(
        "List of the variables that should be targeted by the flow:   ",
        target_cluster_variables,
    )

    cachedir = "data/cache"
    chunks_dir = os.path.join(cachedir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    if out_pkl is None:
        out_pkl = os.path.join(cachedir, "cygno_clusters_dataset.pkl")

    sim_clusters_dict = defaultdict(list)
    data_clusters_dict = defaultdict(list)

    sim_map = dictionary["data_inputs"]["sim_map"]
    data_map = dictionary["data_inputs"]["data_map"]

    if reuse_sim_from is not None:
        print(f"\nCaricamento dataset SIM esistente da: {reuse_sim_from}")
        with open(reuse_sim_from, "rb") as f:
            old_bundle = pickle.load(f)
        sim_clusters_dict = old_bundle["sim"]
        n_sim_loaded = sum(len(v) for v in sim_clusters_dict.values())
        print(
            f"SIM caricata: {len(sim_clusters_dict)} chiavi, {n_sim_loaded} cluster totali."
        )
        maps = {"data": data_map}
    else:
        print(
            "Now filling the datasets for the simulation and data. It applies the selection and converts them to panda DFs.  Since many files are involved, it takes time..."
        )
        maps = {"sim": sim_map, "data": data_map}

    tasks = []

    for k, m in maps.items():
        with open(m) as f:
            raw_map_dic = yaml.safe_load(f) or {}

            map_dic = {}
            for raw_k, v in raw_map_dic.items():
                if not v:  # gestisce chiavi con tutti i file commentati nel YAML
                    continue
                str_vals = str(raw_k).split(",")
                mapkey = tuple(round(float(x), 4) for x in str_vals)
                if mapkey in map_dic:
                    map_dic[mapkey].extend(v)
                else:
                    map_dic[mapkey] = list(v)

            isdata = k == "data"
            for mapkey, files in map_dic.items():
                for root_file in files:
                    tasks.append(
                        (
                            root_file,
                            mapkey,
                            all_cluster_variables,
                            isdata,
                            selection_cfg,
                            chunks_dir,
                        )
                    )

    print(f"\nAvvio processamento di {len(tasks)} file in parallelo...")

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=8, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        futures = {
            executor.submit(process_single_file_task, *t): t for t in tasks
        }

        for future in concurrent.futures.as_completed(futures):
            try:
                mapkey, isdata, chunk_path, completed_file = future.result()

                with open(chunk_path, "rb") as f:
                    clusters = pickle.load(f)

                if isdata:
                    data_clusters_dict[mapkey].extend(clusters)
                else:
                    sim_clusters_dict[mapkey].extend(clusters)

                print(
                    f"Completato: {completed_file} ({len(clusters)} clusters trovati)"
                )

            except Exception as e:
                failed_file = futures[future][0]
                print(f"ERRORE irreversibile nel file {failed_file}:\n{e}")

    metadata = {
        "version": conf,
        "description": "CYGNO cluster dataset for sim-data shape translation",
    }

    metadata["features"] = {
        "pix": ["x_centered", "y_centered", "charge"],
        "all_cluster_variables": all_cluster_variables,
        "flow_scalar_variables": target_cluster_variables,
    }

    metadata["conditioning"] = {
        "sim": ["z", "alpha", "lambda"],
        "data": ["z", "P", "T", "H"],
        "shared_latent": ["z"],
    }

    metadata["stats"] = {
        "n_clusters_sim": sum(len(v) for v in sim_clusters_dict.values()),
        "n_clusters_data": sum(len(v) for v in data_clusters_dict.values()),
    }

    metadata["keys"] = {
        "sim_keys": list(sim_clusters_dict.keys()),
        "data_keys": list(data_clusters_dict.keys()),
    }

    metadata["dataset_hash"] = hashlib.md5(str(metadata).encode()).hexdigest()

    dataset_bundle = {
        "data": dict(data_clusters_dict),
        "sim": dict(sim_clusters_dict),
        "metadata": metadata,
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_pkl)), exist_ok=True)

    # Se sovrascriviamo il file sorgente da cui abbiamo preso la SIM, facciamo prima un backup
    if reuse_sim_from is not None and os.path.abspath(reuse_sim_from) == os.path.abspath(out_pkl):
        backup_path = out_pkl.replace(".pkl", "_backup_sim.pkl")
        if not os.path.exists(backup_path):
            shutil.copy2(out_pkl, backup_path)
            print(f"Backup di sicurezza salvato in: {backup_path}")

    tmp_out = out_pkl + ".tmp"
    with open(tmp_out, "wb") as f:
        pickle.dump(dataset_bundle, f, protocol=4)
    os.replace(tmp_out, out_pkl)

    print(
        f"\nSalvato dataset aggiornato in {out_pkl}: "
        f"SIM={metadata['stats']['n_clusters_sim']} clusters ({len(sim_clusters_dict)} chiavi), "
        f"DATA={metadata['stats']['n_clusters_data']} clusters ({len(data_clusters_dict)} chiavi)"
    )


def inspect_cluster_set(clusters, name=""):

    print(f"\n{name} clusters: {len(clusters)}")

    c0 = clusters[0]

    print("Cluster fields:")

    for k, v in c0.__dict__.items():

        if isinstance(v, np.ndarray):
            print(f"  {k}: ndarray shape={v.shape} dtype={v.dtype}")

        else:
            print(f"  {k}: {type(v).__name__} value={v}")


def _get_cached_cluster_image(c, img_size=64):
    cache_attr = f"_cached_img_tensor_{img_size}"
    img_t = getattr(c, cache_attr, None)
    if img_t is None:
        img_t = torch.as_tensor(
            np.asarray(c.to_image(img_size), dtype=np.float32)
        ).clone()
        setattr(c, cache_attr, img_t)
    return img_t


def _get_cached_cluster_scalars(c, dataset):
    sca_t = getattr(c, "_cached_scalars_tensor", None)
    if sca_t is None:
        sca_t = dataset.cluster_scalars_to_tensor(c, dataset.target_scalars)
        setattr(c, "_cached_scalars_tensor", sca_t)
    return sca_t


def make_cygno_collate_fn(dataset):

    def cygno_collate_fn(batch):

        out = {}

        # -----------------------------
        # stack simple tensors
        # -----------------------------
        out["z"] = torch.tensor([b["z"] for b in batch], dtype=torch.float32)

        out["sim_cond"] = torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(b["sim_cond"], dtype=torch.float32) for b in batch],
            batch_first=True,
        )

        out["data_cond"] = torch.nn.utils.rnn.pad_sequence(
            [
                torch.as_tensor(b["data_cond"], dtype=torch.float32)
                for b in batch
            ],
            batch_first=True,
        )

        # -----------------------------
        # keep raw clusters
        # -----------------------------
        out["sim_clusters"] = [b["sim_clusters"] for b in batch]
        out["data_clusters"] = [b["data_clusters"] for b in batch]

        # -----------------------------
        # images & scalars (con cache in memoria sui cluster)
        # -----------------------------
        sim_images = []
        data_images = []

        sim_scalars = []
        data_scalars = []

        for b in batch:
            sim_images.append(
                torch.stack(
                    [_get_cached_cluster_image(c, 64) for c in b["sim_clusters"]]
                )
            )
            sim_scalars.append(
                torch.stack(
                    [
                        _get_cached_cluster_scalars(c, dataset)
                        for c in b["sim_clusters"]
                    ]
                )
            )

            data_images.append(
                torch.stack(
                    [
                        _get_cached_cluster_image(c, 64)
                        for c in b["data_clusters"]
                    ]
                )
            )
            data_scalars.append(
                torch.stack(
                    [
                        _get_cached_cluster_scalars(c, dataset)
                        for c in b["data_clusters"]
                    ]
                )
            )

        out["sim_images"] = torch.stack(sim_images)
        out["data_images"] = torch.stack(data_images)

        out["sim_scalars"] = torch.stack(sim_scalars)
        out["data_scalars"] = torch.stack(data_scalars)

        return out

    return cygno_collate_fn


def data_loader_test(dataset):
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=make_cygno_collate_fn(dataset),
    )

    batch = next(iter(loader))

    print("\n===== BATCH TEST =====")
    print("z:", batch["z"])
    print("sim_cond:", batch["sim_cond"].shape)
    print("data_cond:", batch["data_cond"].shape)
    print("flow target scalar variables:", dataset.target_scalars)


def image_test(sample):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(sample["sim_clusters"][0].to_image(64), origin="lower")
    ax[0].set_title("SIM")

    ax[1].imshow(sample["data_clusters"][0].to_image(64), origin="lower")
    ax[1].set_title("DATA")

    plt.show()


def dataset_sanity(inputfile):
    print(
        f"Reading source_data and target_data from pre-selected cluster datasets in {inputfile}"
    )

    dataset = ConditionalClusterDataset(pkl_file=inputfile, n_clusters=32)

    debug_clusters_dataset(
        sim_dataset_dict=dataset.sim_dict,
        whitelist_scalars=["integral"],
        output_dir="plot/debug_fase_1",
    )


def integrity_tests(inputfile):
    print(
        f"Reading source_data and target_data from pre-selected cluster datasets in {inputfile}"
    )

    dataset = ConditionalClusterDataset(pkl_file=inputfile, n_clusters=32)

    print("\n===== DATASET INFO =====")
    print("Shared z values:", dataset.shared_z)
    print(
        "Number of sim keys:",
        sum(len(v) for v in dataset.sim_keys_by_z.values()),
    )
    print(
        "Number of data keys:",
        sum(len(v) for v in dataset.data_keys_by_z.values()),
    )
    print("Dataset length (virtual):", len(dataset))

    sample = dataset[0]

    print("\n\t ***** SINGLE SAMPLE TEST *****")
    print("\n===== SAMPLE KEYS =====")
    print(sample.keys())

    print("\n===== BASIC SHAPES =====")

    print("z:", sample["z"])
    print("sim_cond:", sample["sim_cond"])
    print("data_cond:", sample["data_cond"])

    print("\nSIM CLUSTERS:", len(sample["sim_clusters"]))
    print("DATA CLUSTERS:", len(sample["data_clusters"]))

    print("\n\n\t ***** CLUSTER VARIANCE TEST *****")
    inspect_cluster_set(sample["sim_clusters"], "SIM")
    inspect_cluster_set(sample["data_clusters"], "DATA")

    print("\n\n\t ***** DATA LOADER TEST *****")
    data_loader_test(dataset)

    print("\n\n\t ***** IMAGE CLUSTER TEST *****")
    image_test(sample)


def debug_clusters_dataset(
    sim_dataset_dict, whitelist_scalars=["integral"], output_dir="debug_plots"
):
    """Esegue un sanity check immediato sui cluster appena estratti in memoria.

    Accetta un dizionario dove la chiave identifica il contesto (es. la stringa
    o la tupla della condizione) e il valore è la lista di oggetti Cluster.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n=== [DEBUG] ISPEZIONE DATASET DI SIMULAZIONE IN MEMORIA ===")

    for ctx_key, cluster_list in sim_dataset_dict.items():
        num_clusters = len(cluster_list)
        print(
            f"Chiave Contesto: {ctx_key} | Numero Cluster Estratti: {num_clusters}"
        )

        if num_clusters == 0:
            print(f"  --> ATTENZIONE: Nessun cluster per la chiave {ctx_key}")
            continue

        for var_name in whitelist_scalars:
            try:
                vals = [getattr(c, var_name) for c in cluster_list]
                vals = np.array(vals)

                v_min, v_max = vals.min(), vals.max()
                v_mean, v_std = vals.mean(), vals.std()
                print(
                    f"  -> Var '{var_name}': Media={v_mean:.2f} ± {v_std:.2f} | Range=[{v_min:.2f}, {v_max:.2f}]"
                )

                plt.figure(figsize=(6, 4))
                bins = np.linspace(v_min, v_max, 40) if v_min != v_max else 10

                plt.hist(
                    vals,
                    bins=bins,
                    color="tab:blue",
                    alpha=0.7,
                    edgecolor="black",
                    density=True,
                )

                clean_filename = (
                    str(ctx_key)
                    .replace("(", "")
                    .replace(")", "")
                    .replace(" ", "")
                    .replace(",", "_")
                )

                plt.title(
                    f"DEBUG INGRESSO: {var_name}\nContesto: {ctx_key}\nCluster Totali: {num_clusters}",
                    fontsize=10,
                    fontweight="bold",
                )
                plt.xlabel(var_name)
                plt.ylabel("Densità")
                plt.grid(True, linestyle="--", alpha=0.5)

                plot_path = os.path.join(
                    output_dir, f"debug_{var_name}_{clean_filename}.png"
                )
                plt.savefig(plot_path, dpi=120, bbox_inches="tight")
                plt.close()
                print(f"  --> Grafico salvato in: {plot_path}")

            except AttributeError:
                print(
                    f"  --> ERRORE: Lo scalare '{var_name}' non esiste nell'oggetto Cluster."
                )

    print("===========================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rigenera solo DATA dal YAML aggiornato e fa merge con la SIM esistente."
    )
    parser.add_argument(
        "--configuration",
        type=str,
        default="configuration_limerun4_v2",
        help="Nome configurazione da salvare nei metadata",
    )
    parser.add_argument(
        "--sim-pkl",
        type=str,
        default="data/cache/cygno_clusters_dataset.pkl",
        help="File .pkl esistente da cui estrarre la SIM",
    )
    parser.add_argument(
        "--out-pkl",
        type=str,
        default="data/cache/cygno_clusters_dataset.pkl",
        help="File .pkl di output unito (SIM vecchia + DATA nuova)",
    )
    args = parser.parse_args()

    read_data_and_save(
        conf=args.configuration,
        reuse_sim_from=args.sim_pkl,
        out_pkl=args.out_pkl,
    )