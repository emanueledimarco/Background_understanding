#python -m application.flow_correction --input data/sim/recosim_2k/sim_2k_200_399/digi_5-5/iron_step3/reco_run00001_3D.root --output application/friend.root --sim-cond 15.0 0.0210 1350 --target-cond 15.0 0.98 295.0 1.5 --checkpoint ../../CygnoCorrections/results/configuration_limerun4_v2/saved_states/best_model.pt

import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import uproot
import awkward as ak

from data_reading.cluster import build_clusters_from_event
from training.clusterTraining import CygnoTransportModel, compute_physical_scalars_from_image, noise_threshold

def run_inference_friend(
    input_root_file,
    output_friend_file,
    checkpoint_path,
    sim_cond,
    target_cond,
    scalar_vars=None,
    tree_name="Events",
    image_size=64,
    device=None
):

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if scalar_vars is None:
        scalar_vars = ["sc_integral", "sc_rms", "sc_length", "sc_width",
                       "sc_xmean", "sc_ymean", "sc_nhits" ]

    # 1. Caricamento Modello
    model = CygnoTransportModel().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model_state = state["model_state_dict"] if "model_state_dict" in state else state
    model.load_state_dict(model_state)
    model.eval()

    # 2. Lettura File ROOT Nominale
    with uproot.open(input_root_file) as f_in:
        tree = f_in[tree_name]
        arrays = tree.arrays(library="np")
        num_events = len(arrays["nSc"])

    # Preparazione tensori delle condizioni
    # sim_cond: (z, alpha, lambda) -> shape (1, 3)
    # target_cond: (z, P, T, H)    -> shape (1, 4)
    sim_cond_t = torch.tensor(sim_cond, dtype=torch.float32).unsqueeze(0).to(device)
    data_cond_t = torch.tensor(target_cond, dtype=torch.float32).unsqueeze(0).to(device)

    all_ev_ix_corr = []
    all_ev_iy_corr = []
    all_ev_iz_corr = []
    all_ev_redpix_idx = []
    all_ev_npix = []

    center = image_size // 2

    # 3. Processamento Evento per Evento
    for iev in range(num_events):
        
        # Costruzione oggetti Cluster On-The-Fly (applica centramento xmean, ymean)
        clusters_evt = build_clusters_from_event(
            arrays=arrays,
            iev=iev,
            scalar_vars=scalar_vars,
            conditions=sim_cond,
            isdata=False,
            selection_cfg=None
        )

        ev_ix_corr = []
        ev_iy_corr = []
        ev_iz_corr = []
        ev_redpix_idx = [0]
        ev_npix = []

        for isc, cluster in enumerate(clusters_evt):
            # Converti il cluster in immagine 2D (64x64) come nel dataset di training
            img_2d = cluster.to_image(image_size=image_size)
            img_tensor = torch.tensor(img_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            # Calcolo degli scalari fisici online dall'immagine
            with torch.no_grad():
                sim_scalars = compute_physical_scalars_from_image(img_tensor)
                out = model(img_tensor, sim_cond_t, data_cond_t, sim_scalars)
                
                # Applicazione soglia e clamp
                pred_img = F.relu(out["pred_images"]).squeeze().cpu().numpy()
                pred_img[pred_img <= noise_threshold] = 0.0

            # Estraggo le coordinate 2D e la carica dei pixel ricostruiti dal Flow
            y_indices, x_indices = np.where(pred_img > 0.0)
            q_values = pred_img[y_indices, x_indices]

            # Riconversione alle coordinate locali centrate
            local_x = x_indices - center
            local_y = y_indices - center

            # Ripristino coordinate globali aggiungendo xmean e ymean del supercluster originale
            global_x = local_x + cluster.xmean
            global_y = local_y + cluster.ymean

            ev_ix_corr.extend(global_x)
            ev_iy_corr.extend(global_y)
            ev_iz_corr.extend(q_values)

            # Aggiornamento indici e conteggi dei pixel corretti per il cluster corrente
            n_pix_corr = len(q_values)
            ev_npix.append(n_pix_corr)
            ev_redpix_idx.append(ev_redpix_idx[-1] + n_pix_corr)

        all_ev_ix_corr.append(ev_ix_corr)
        all_ev_iy_corr.append(ev_iy_corr)
        all_ev_iz_corr.append(ev_iz_corr)
        all_ev_redpix_idx.append(ev_redpix_idx[:-1]) # Rimuovo l'ultimo elemento cumulativo
        all_ev_npix.append(ev_npix)

    # 4. Scrittura Friend Tree
    awk_ix = ak.Array(all_ev_ix_corr)
    awk_iy = ak.Array(all_ev_iy_corr)
    awk_iz = ak.Array(all_ev_iz_corr)
    awk_idx = ak.Array(all_ev_redpix_idx)
    awk_npix = ak.Array(all_ev_npix)

    os.makedirs(os.path.dirname(os.path.abspath(output_friend_file)), exist_ok=True)

    with uproot.recreate(output_friend_file) as fout:
        fout.mktree('Friend', {
            "redpix_ix_corr": awk_ix,
            "redpix_iy_corr": awk_iy,
            "redpix_iz_corr": awk_iz,
            "sc_redpixIdx_corr": awk_idx,
            "sc_npix_corr": awk_npix
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generazione Friend Tree con collezioni redpix corrette tramite Transport Model")
    parser.add_argument("--input", type=str, required=True, help="Path del file ROOT di simulazione nominale")
    parser.add_argument("--output", type=str, required=True, help="Path del file ROOT Friend Tree di output")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path del modello addestrato (.pt)")
    parser.add_argument("--sim-cond", nargs=3, type=float, default=[15.0, 0.0210, 1350], help="Condizione SIM nominale: z alpha lambda")
    parser.add_argument("--target-cond", nargs=4, type=float, default=[15.0, 0.98, 295.0, 1.5], help="Condizione DATA target: z P T H")

    args = parser.parse_args()

    with open("cluster_training_list.json", "r") as file:
        json_data = json.load(file)
        
    all_cluster_variables = json_data["all_cluster_variables"]

    run_inference_friend(
        input_root_file=args.input,
        output_friend_file=args.output,
        checkpoint_path=args.checkpoint,
        sim_cond=args.sim_cond,
        target_cond=args.target_cond,
        scalar_vars=all_cluster_variables
    )
