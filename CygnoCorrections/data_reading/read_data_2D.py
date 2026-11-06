# Loads and treat the reconstruction tree data and simulation
# test basic plotting with python -m data_reading.read_data 

import os
import pickle
import uproot
import awkward as ak
import pandas as pd
import numpy as np
import json
from collections import defaultdict
import matplotlib.pyplot as plt

import yaml
from yaml import Loader
import json
import hashlib
import urllib.request
import urllib.error

import concurrent.futures
import tempfile
import subprocess
import hashlib
import shutil

import torch
from torch.utils.data import DataLoader

from data_reading.cluster import *
from data_reading.clusterDataset import ConditionalClusterDataset

selection_cfg = {
    "integral_min": 2000,
    "integral_max": 50000,
    "x_min": 500,
    "x_max": 2000,
    "y_min": 500,
    "y_max": 2000,
    "min_npix": 500,
    "n_hits": 200,
    # additional (not used for training)
    "length_min": 0,
    "length_max": 1e10,
    "slimness_min": 0.,
}

def process_single_file_task(root_file, mapkey, all_cluster_variables, isdata, selection_cfg, chunks_dir):
    """
    Scarica, processa e salva in cache un singolo file ROOT in modo indipendente.
    """
    # Genera un ID univoco per questo specifico file e condizione
    file_id = hashlib.md5(f"{mapkey}_{root_file}".encode()).hexdigest()
    chunk_path = os.path.join(chunks_dir, f"chunk_{file_id}.pkl")
    
    # 3. [SICUREZZA E RESILIENZA]
    # Se il chunk esiste già, significa che questo file ROOT era già stato processato con successo in precedenza.
    if os.path.exists(chunk_path):
        return mapkey, isdata, chunk_path, root_file

    local_file = root_file
    tmp_dir = None
    
    # Riconosce se è un file remoto o montato. 
    # Metti a True se vuoi FORZARE sempre la copia in /tmp (consigliato se leggi da un mount di rete lento)
    force_local_staging = True 

    try:
        # 1. [OTTIMIZZAZIONE LETTURA] Staging locale
        if force_local_staging:
            tmp_dir = tempfile.mkdtemp()
            # Usa il nome del file originale preso dall'URL
            local_file = os.path.join(tmp_dir, root_file.split("/")[-1])
            
            if root_file.startswith("http://") or root_file.startswith("https://"):
                # Download diretto ad altissima velocità (ideale per S3 esposto via HTTP)
                try:
                    urllib.request.urlretrieve(root_file, local_file)
                except urllib.error.URLError as e:
                    raise RuntimeError(f"Impossibile scaricare l'URL S3 {root_file}: {e}")
            elif ":" in root_file and not root_file.startswith(("C:", "D:")):
                # Fallback per sintassi rclone "remote:bucket/file"
                subprocess.run(["rclone", "copyto", root_file, local_file], check=True)
            else:
                # Fallback per percorsi di sistema standard
                shutil.copy2(root_file, local_file)
            
        # 2. Processa il file ROOT (ora è sul disco locale e la lettura sarà istantanea)
        clusters = build_clusters_from_root_file(
            local_file,
            scalar_vars=all_cluster_variables,
            conditions=mapkey,
            isdata=isdata,
            selection_cfg=selection_cfg
        )
    
        # Salva il risultato parziale. Se si interrompe ora, al prossimo run ripartirà da qui.
        with open(chunk_path, "wb") as f:
            pickle.dump(clusters, f, protocol=4)
            
    finally:
        # Pulizia dello spazio disco: elimina file scaricato e cartella temporanea
        if force_local_staging and os.path.exists(local_file):
            os.remove(local_file)
        if force_local_staging and tmp_dir and os.path.exists(tmp_dir):
            os.rmdir(tmp_dir)

    return mapkey, isdata, chunk_path, root_file

def read_data_and_save(conf):

    #loop to read over network condigurations from the yaml file: - one way to do hyperparameter optimization
    stream = open("flow_configuration.yaml", 'r')
    dictionary = yaml.load(stream,Loader)

    with open("cluster_training_list.json", "r") as file:
        json_data = json.load(file)
        
    all_cluster_variables = json_data["all_cluster_variables"]
    print("List of all the variables that should be attached to the cluster object:   ", all_cluster_variables)
    target_cluster_variables = json_data["target_cluster_variables"]
    print("List of the variables that should be targeted by the flow:   ", target_cluster_variables)
    
    print(f"Now filling the datasets for the simulation and data. It applies the selection and converts them to panda DFs.  Since many files are involved, it takes time...")
    
    cachedir = "data/cache"
    chunks_dir = os.path.join(cachedir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    
    sim_clusters_dict = defaultdict(list)
    data_clusters_dict = defaultdict(list)
    
    sim_map  = dictionary["data_inputs"]["sim_map"]
    data_map = dictionary["data_inputs"]["data_map"]
    maps = dict(zip(["sim", "data"], [sim_map, data_map]))

    # Prepara la lista di tutti i task (file singoli)
    tasks = []
    
    for k, m in maps.items():
        with open(m) as f:
            raw_map_dic = yaml.safe_load(f)
            
            map_dic = {}
            for raw_k, v in raw_map_dic.items():
                str_vals = raw_k.split(",")
                mapkey = tuple(round(float(x), 4) for x in str_vals)
                if mapkey in map_dic:
                    map_dic[mapkey].extend(v) 
                else:
                    map_dic[mapkey] = v

            isdata = (k == "data")
            for mapkey, files in map_dic.items():
                for root_file in files:
                    tasks.append((root_file, mapkey, all_cluster_variables, isdata, selection_cfg, chunks_dir))

    print(f"\nAvvio processamento di {len(tasks)} file in parallelo...")
    
    # 2. [PARALLELIZZAZIONE]
    # Modifica max_workers in base al numero di CPU libere e alla RAM.
    # Inizia con 4 o 8. Se la rete non è satura e hai CPU libere, puoi alzarlo.
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        # Sottometti i task
        futures = {executor.submit(process_single_file_task, *t): t for t in tasks}
        
        # Mano a mano che i processi finiscono, uniamo i risultati
        for future in concurrent.futures.as_completed(futures):
            try:
                mapkey, isdata, chunk_path, completed_file = future.result()
                
                # Ricarica in memoria il chunk processato dal disco locale
                with open(chunk_path, "rb") as f:
                    clusters = pickle.load(f)
                    
                if isdata:
                    data_clusters_dict[mapkey].extend(clusters)
                else:
                    sim_clusters_dict[mapkey].extend(clusters)
                    
                print(f"Completato: {completed_file} ({len(clusters)} clusters trovati)")
                
            except Exception as e:
                failed_file = futures[future][0]
                print(f"ERRORE irreversibile nel file {failed_file}:\n{e}")
    
    # save some metadata information
    metadata = {
        "version": conf,
        "description": "CYGNO cluster dataset for sim-data shape translation",
        
    }
    
    metadata["features"] = {
        "pix": ["x_centered", "y_centered", "charge"],
        "all_cluster_variables": all_cluster_variables,
        "flow_scalar_variables": target_cluster_variables
    }

    metadata["conditioning"] = {
        "sim": ["z", "alpha", "lambda"],
        "data": ["z", "P", "T", "H"],
        "shared_latent": ["z"]
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
        "data": data_clusters_dict,
        "sim": sim_clusters_dict,
        "metadata": metadata
    }
    
    os.makedirs(cachedir, exist_ok=True)
    with open(f"{cachedir}/cygno_clusters_dataset.pkl", "wb") as f:
        pickle.dump(dataset_bundle, f, protocol=4)
    

def inspect_cluster_set(clusters, name=""):

    print(f"\n{name} clusters: {len(clusters)}")

    c0 = clusters[0]

    print("Cluster fields:")

    for k, v in c0.__dict__.items():

        if isinstance(v, np.ndarray):
            print(
                f"  {k}: ndarray "
                f"shape={v.shape} "
                f"dtype={v.dtype}"
            )

        else:
            print(
                f"  {k}: {type(v).__name__} "
                f"value={v}"
            )
            

def make_cygno_collate_fn(dataset):
    
    def cygno_collate_fn(batch):

        out = {}
     
        # -----------------------------
        # stack simple tensors
        # -----------------------------
        out["z"] = torch.tensor(
            [b["z"] for b in batch],
            dtype=torch.float32
        )
     
        out["sim_cond"] = [
            torch.tensor(
                b["sim_cond"],
                dtype=torch.float32
            )
            for b in batch
        ]
     
        out["data_cond"] = [
            torch.tensor(
                b["data_cond"],
                dtype=torch.float32
            )
            for b in batch
        ]
     
        out["sim_cond"] = torch.nn.utils.rnn.pad_sequence(
            out["sim_cond"],
            batch_first=True
        )
     
        out["data_cond"] = torch.nn.utils.rnn.pad_sequence(
            out["data_cond"],
            batch_first=True
        )
     
        # -----------------------------
        # keep raw clusters
        # -----------------------------
        out["sim_clusters"] = [
            b["sim_clusters"]
            for b in batch
        ]
     
        out["data_clusters"] = [
            b["data_clusters"]
            for b in batch
        ]
     
        # -----------------------------
        # images
        # -----------------------------
        sim_images = []
        data_images = []
     
        sim_scalars = []
        data_scalars = []
     
        for b in batch:
     
            sim_imgs = []
            sim_sca = []
     
            for c in b["sim_clusters"]:
     
                sim_imgs.append(
                    torch.tensor(
                        c.to_image(64),
                        dtype=torch.float32
                    )
                )
     
                sim_sca.append(
                    dataset.cluster_scalars_to_tensor(c,dataset.target_scalars)
                )
     
            sim_images.append(
                torch.stack(sim_imgs)
            )
     
            sim_scalars.append(
                torch.stack(sim_sca)
            )
     
            data_imgs = []
            data_sca = []
     
            for c in b["data_clusters"]:
     
                data_imgs.append(
                    torch.tensor(
                        c.to_image(64),
                        dtype=torch.float32
                    )
                )
     
                data_sca.append(
                    dataset.cluster_scalars_to_tensor(c,dataset.target_scalars)
                )
     
            data_images.append(
                torch.stack(data_imgs)
            )
     
            data_scalars.append(
                torch.stack(data_sca)
            )
     
        out["sim_images"] = torch.stack(
            sim_images
        )
     
        out["data_images"] = torch.stack(
            data_images
        )
     
        out["sim_scalars"] = torch.stack(
            sim_scalars
        )
     
        out["data_scalars"] = torch.stack(
            data_scalars
        )
     
        return out

    return cygno_collate_fn
    
def data_loader_test(dataset):
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=make_cygno_collate_fn(dataset)
    )

    batch = next(iter(loader))

    print("\n===== BATCH TEST =====")
    print("z:", batch["z"])
    print("sim_cond:", batch["sim_cond"].shape)
    print("data_cond:", batch["data_cond"].shape)
    print("flow target scalar variables:",dataset.target_scalars)

def image_test(sample):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].imshow(
        sample["sim_clusters"][0].to_image(64),
        origin="lower"
    )
    ax[0].set_title("SIM")
    
    ax[1].imshow(
        sample["data_clusters"][0].to_image(64),
        origin="lower"
    )
    ax[1].set_title("DATA")
    
    plt.show()

def dataset_sanity(inputfile):
        print(f"Reading source_data and target_data from pre-selected cluster datasets in {inputfile}")

        dataset = ConditionalClusterDataset(
            pkl_file=inputfile,
            n_clusters=32
        )

        debug_clusters_dataset(
            sim_dataset_dict=dataset.sim_dict,
            whitelist_scalars=["integral"],
            output_dir="plot/debug_fase_1",
        )
        
    
def integrity_tests(inputfile):
        print(f"Reading source_data and target_data from pre-selected cluster datasets in {inputfile}")

        dataset = ConditionalClusterDataset(
            pkl_file=inputfile,
            n_clusters=32
        )

        print("\n===== DATASET INFO =====")
        print("Shared z values:", dataset.shared_z)
        print("Number of sim keys:", sum(len(v) for v in dataset.sim_keys_by_z.values()))
        print("Number of data keys:", sum(len(v) for v in dataset.data_keys_by_z.values()))
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
        

def debug_clusters_dataset(sim_dataset_dict, whitelist_scalars=["integral"], output_dir="debug_plots"):
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

        # Estraiamo gli scalari richiesti ciclando sugli oggetti Cluster
        for var_name in whitelist_scalars:
            try:
                # Recuperiamo l'attributo dinamico (es. cluster.integral)
                vals = [getattr(c, var_name) for c in cluster_list]
                vals = np.array(vals)

                # Calcoliamo metriche di controllo rapide
                v_min, v_max = vals.min(), vals.max()
                v_mean, v_std = vals.mean(), vals.std()
                print(
                    f"  -> Var '{var_name}': Media={v_mean:.2f} ± {v_std:.2f} | Range=[{v_min:.2f}, {v_max:.2f}]"
                )

                # Generiamo un plot 1D isolato per questa chiave
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

                # Pulizia del nome del file dalle parentesi o caratteri speciali delle chiavi
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
