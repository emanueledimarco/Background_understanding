from collections import defaultdict
import math
import os
import pickle
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from data_reading.clusterDataset import ConditionalClusterDataset
from data_reading.read_data_2D import make_cygno_collate_fn

noise_threshold = 1.5


# this class is to run with multiple workers in building the dataloader
class _PicklableCygnoCollate:
    def __init__(self, dataset):
        self.dataset = dataset
        self._fn = None

    def __call__(self, batch):
        if self._fn is None:
            self._fn = make_cygno_collate_fn(self.dataset)
        return self._fn(batch)


def build_dataloader(
    inputfile,
    batch_size=32,
    n_clusters=32,
    shuffle=True,
    is_test=False,
    num_workers=0,
):
    dataset = ConditionalClusterDataset(
        pkl_file=inputfile, n_clusters=n_clusters, is_test=is_test
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
        collate_fn=_PicklableCygnoCollate(dataset),
    )

    return dataset, loader


# ==============================================================================
# 1. ENCODER DELLE CONDIZIONI CON z CONDIVISO 1:1 TRA SIM E DATA
# ==============================================================================
class ConditionEncoder(nn.Module):

    def __init__(self, sim_dim=3, data_dim=4, emb_dim=64):
        super().__init__()
        self.sim_dim = sim_dim
        self.data_dim = data_dim

        # Buffer salvati automaticamente nello state_dict (.pt) per --validate
        self.register_buffer("sim_mean", torch.zeros(1, sim_dim))
        self.register_buffer("sim_std", torch.ones(1, sim_dim))
        self.register_buffer("data_mean", torch.zeros(1, data_dim))
        self.register_buffer("data_std", torch.ones(1, data_dim))

        # z è identica 1:1 tra SIM e DATA: viene codificata una sola volta come regime fisico condiviso
        self.z_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.LayerNorm(32),
        )
        # (z, alpha, lambda): parametri fisici della simulazione (saturazione e assorbimento)
        self.sim_net = nn.Sequential(
            nn.Linear(sim_dim, 64),
            nn.SiLU(),
            nn.Linear(64, emb_dim),
            nn.LayerNorm(emb_dim),
        )
        # (z, P, T, H): condizioni ambientali del rivelatore reale
        self.data_net = nn.Sequential(
            nn.Linear(data_dim, 64),
            nn.SiLU(),
            nn.Linear(64, emb_dim),
            nn.LayerNorm(emb_dim),
        )

    @torch.no_grad()
    def set_normalization(self, sim_mean, sim_std, data_mean, data_std):
        device = self.sim_mean.device
        self.sim_mean.copy_(
            torch.as_tensor(sim_mean, dtype=torch.float32, device=device).view(
                1, self.sim_dim
            )
        )
        self.sim_std.copy_(
            torch.as_tensor(sim_std, dtype=torch.float32, device=device)
            .view(1, self.sim_dim)
            .clamp(min=1e-6)
        )
        self.data_mean.copy_(
            torch.as_tensor(data_mean, dtype=torch.float32, device=device).view(
                1, self.data_dim
            )
        )
        self.data_std.copy_(
            torch.as_tensor(data_std, dtype=torch.float32, device=device)
            .view(1, self.data_dim)
            .clamp(min=1e-6)
        )

    def encode_z(self, sim_cond):
        z_norm = (sim_cond[:, :1] - self.sim_mean[:, :1]) / self.sim_std[:, :1]
        return self.z_net(z_norm)

    def encode_sim(self, sim_cond):
        sim_norm = (sim_cond - self.sim_mean) / self.sim_std
        return self.sim_net(sim_norm)

    def encode_data(self, data_cond):
        data_norm = (data_cond - self.data_mean) / self.data_std
        return self.data_net(data_norm)

    def forward(self, sim_cond, data_cond):
        return (
            self.encode_z(sim_cond),
            self.encode_sim(sim_cond),
            self.encode_data(data_cond),
        )


# ==============================================================================
# 2. OPERATORE FISICO LOCALE EQUIVARIANTE (v3.0: SATURAZIONE GEM + PSF ISOTROPA)
# ==============================================================================
class LocalDetectorTransportModel(nn.Module):
    """
    Trasporto locale fisicamente vincolato (v3.0):
      1. z condiviso 1:1 tra SIM (z, alpha, lambda) e DATA (z, P, T, H)
      2. PSF quasi-isotropa senza rumore casuale sulla forma (preserva Eccentricity)
      3. Esponente di compattezza radiale gamma in [0.43, 2.34] (coda sinistra di Width)
      4. Saturazione differenziale di carica spaziale GEM (Eq. 11 Amaro et al., EPJC 2026):
            I_out(x,y) = g * I_shaped(x,y) / (1 + s_sat * I_shaped(x,y) / I_ref)
         che agisce localmente sui pixel ad alta densità (piccole z) e si annulla
         automaticamente quando la nube è diffusa (grandi z), evitando smearing spuri.
    """

    def __init__(
        self,
        kernel_size=11,
        cond_dim=64,
        sim_dim=3,
        data_dim=4,
        noise_dim=2,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size deve essere dispari"
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.cond_dim = cond_dim
        self.noise_dim = max(noise_dim, 2)

        self.cond_encoder = ConditionEncoder(
            sim_dim=sim_dim, data_dim=data_dim, emb_dim=cond_dim
        )

        in_dim = 32 + 2 * cond_dim  # [h_z, h_sim, h_data]

        # 6 parametri fisici deterministici:
        # [mu_log_gain, raw_sigma, raw_ellip, raw_theta, raw_gamma, raw_sat]
        self.mean_net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 6),
        )

        # Solo piccolo jitter fisico sul guadagno (<= 0.04) e granularità sensore (<= 0.08)
        # NESSUN rumore casuale su PSF o saturazione -> niente smearing spurio
        self.std_net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )

        # Griglia locale centrata per la PSF Gaussiana
        ax = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size // 2)
        gy, gx = torch.meshgrid(ax, ax, indexing="ij")
        self.register_buffer("grid_x", gx.view(1, 1, kernel_size, kernel_size))
        self.register_buffer("grid_y", gy.view(1, 1, kernel_size, kernel_size))

        # Inizializzazione all'identità (g=1, sigma~0.09px, gamma=1, s_sat=0)
        nn.init.zeros_(self.mean_net[-1].weight)
        nn.init.zeros_(self.mean_net[-1].bias)
        nn.init.zeros_(self.std_net[-1].weight)
        nn.init.constant_(self.std_net[-1].bias, -3.0)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if (
            prefix + "transport_net.0.weight" in state_dict
            or prefix + "cond_encoder.z_net.0.weight" not in state_dict
        ):
            raise RuntimeError(
                "Il checkpoint salvato appartiene a una versione precedente del modello. "
                "Rilancia il training con '--train' per generare il nuovo modello v3.0."
            )

        gx_key = prefix + "grid_x"
        if gx_key in state_dict:
            expected_k = state_dict[gx_key].shape[-1]
            if expected_k != self.kernel_size:
                self.kernel_size = expected_k
                self.pad = expected_k // 2
                device = self.grid_x.device
                ax = torch.arange(
                    expected_k, dtype=torch.float32, device=device
                ) - (expected_k // 2)
                gy, gx = torch.meshgrid(ax, ax, indexing="ij")
                self.grid_x = gx.view(1, 1, expected_k, expected_k)
                self.grid_y = gy.view(1, 1, expected_k, expected_k)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _build_quasi_isotropic_psf(self, raw_sigma, raw_ellip, raw_theta):
        # Larghezza radiale di base in [0.05, 2.25] px (al init ~0.09 px -> delta di Dirac)
        sigma_0 = (0.05 + 2.20 * torch.sigmoid(raw_sigma - 4.0)).view(
            -1, 1, 1, 1
        )
        # Anisotropia strettamente limitata a +-10% per non distorcere Eccentricity
        ellip = (0.10 * torch.tanh(raw_ellip)).view(-1, 1, 1, 1)
        sx = sigma_0 * torch.exp(ellip)
        sy = sigma_0 * torch.exp(-ellip)
        theta = (0.5 * math.pi * torch.tanh(raw_theta)).view(-1, 1, 1, 1)

        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        u = self.grid_x * cos_t + self.grid_y * sin_t
        v = -self.grid_x * sin_t + self.grid_y * cos_t

        kernel = torch.exp(-0.5 * ((u / sx) ** 2 + (v / sy) ** 2))
        kernel = kernel / (kernel.sum(dim=(-2, -1), keepdim=True) + 1e-8)
        return kernel

    def _apply_transport(self, img, h_z, h_sim, h_data, noise, add_grain=True):
        B = img.shape[0]
        cond_feat = torch.cat([h_z, h_sim, h_data], dim=-1)

        mu_params = self.mean_net(cond_feat)
        std_params = self.std_net(cond_feat)

        # Jitter di guadagno strettamente fisico (<= 0.04 in log_gain) -> zero smearing artificiale
        std_log_gain = 0.04 * torch.sigmoid(std_params[:, 0:1])
        log_gain = mu_params[:, 0:1] + std_log_gain * noise[:, 0:1]

        # 1. Guadagno lineare non saturato: [0.135, 7.39]
        gain = torch.exp(2.0 * torch.tanh(0.5 * log_gain))

        # 2. Convoluzione con PSF quasi-isotropa
        psf_kernel = self._build_quasi_isotropic_psf(
            mu_params[:, 1:2], mu_params[:, 2:3], mu_params[:, 3:4]
        )
        img_padded = F.pad(
            img, (self.pad, self.pad, self.pad, self.pad), mode="replicate"
        )
        patches = F.unfold(img_padded, kernel_size=self.kernel_size)
        img_psf = F.relu(
            torch.bmm(psf_kernel.view(B, 1, -1), patches).view(
                B, 1, img.shape[-2], img.shape[-1]
            )
        )

        # 3. Esponente di compattezza radiale gamma in [0.43, 2.34] (at init gamma = 1.0)
        gamma = torch.exp(0.85 * torch.tanh(mu_params[:, 4:5])).view(
            B, 1, 1, 1
        )
        eps_pow = 1e-3
        base_offset = torch.pow(
            torch.tensor(eps_pow, device=img.device, dtype=img.dtype), gamma
        )
        img_gamma = (img_psf + eps_pow).pow(gamma) - base_offset
        flux_psf = img_psf.sum(dim=(-1, -2), keepdim=True) + 1e-8
        flux_gamma = img_gamma.sum(dim=(-1, -2), keepdim=True) + 1e-8
        img_shaped = img_gamma * (flux_psf / flux_gamma)

        # 4. Saturazione differenziale di carica spaziale GEM (Eq. 11 Amaro et al., EPJC 2026):
        #    G(n0) = G_tilde / (1 + beta * n0 * (G_tilde - 1))
        #    s_sat > 0: Data satura più della SIM (comprime i picchi densi a piccole z)
        #    s_sat < 0: Data satura meno della SIM (de-saturazione)
        sat_alpha = (0.60 * torch.tanh(mu_params[:, 5:6])).view(B, 1, 1, 1)
        charge_density = img_shaped / 100.0
        sat_denom = torch.clamp(1.0 + sat_alpha * charge_density, min=0.35)
        saturated = (gain.view(B, 1, 1, 1) * img_shaped) / sat_denom

        # 5. Granularità statistica contenuta (<= 0.08 * sqrt(I)) solo sul core attivo
        if add_grain:
            grain_scale = (0.08 * torch.sigmoid(std_params[:, 1:2])).view(
                B, 1, 1, 1
            )
            active_mask = (saturated > 0.5).to(saturated.dtype)
            pixel_noise = torch.randn_like(saturated)
            pred = F.relu(
                saturated
                + active_mask
                * grain_scale
                * torch.sqrt(saturated + 1e-4)
                * pixel_noise
            )
        else:
            pred = saturated

        delta_h = pred - img
        return pred, delta_h, gain, gamma.view(B, 1), sat_alpha.view(B, 1), psf_kernel

    def forward_identity(self, sim_images, sim_cond):
        B = sim_images.shape[0]
        h_z = self.cond_encoder.encode_z(sim_cond)
        h_sim = self.cond_encoder.encode_sim(sim_cond)
        zero_noise = torch.zeros(
            B,
            self.noise_dim,
            device=sim_images.device,
            dtype=sim_images.dtype,
        )
        pred_id, delta_h, gain, gamma, sat_alpha, psf_kernel = (
            self._apply_transport(
                sim_images, h_z, h_sim, h_sim, zero_noise, add_grain=False
            )
        )
        return {
            "pred_images": pred_id,
            "delta_h": delta_h,
            "gain": gain,
            "gamma": gamma,
            "sat_alpha": sat_alpha,
            "kernel_res": psf_kernel,
        }

    def forward(
        self,
        sim_images,
        sim_cond,
        data_cond,
        sim_scalars=None,
        compute_identity=False,
        noise=None,
    ):
        B = sim_images.shape[0]
        if noise is None:
            noise = torch.randn(
                B,
                self.noise_dim,
                device=sim_images.device,
                dtype=sim_images.dtype,
            )

        # z è forzata 1:1 da sim_cond a data_cond (stessa distanza dalle GEM)
        data_cond_matched = torch.cat(
            [sim_cond[:, :1], data_cond[:, 1:]], dim=-1
        )

        h_z = self.cond_encoder.encode_z(sim_cond)
        h_sim = self.cond_encoder.encode_sim(sim_cond)
        h_data = self.cond_encoder.encode_data(data_cond_matched)

        pred, delta_h, gain, gamma, sat_alpha, psf_kernel = (
            self._apply_transport(
                sim_images, h_z, h_sim, h_data, noise, add_grain=True
            )
        )

        out = {
            "pred_images": pred,
            "delta_h": delta_h,
            "gain": gain,
            "gamma": gamma,
            "sat_alpha": sat_alpha,
            "kernel_res": psf_kernel,
        }

        if compute_identity:
            zero_noise = torch.zeros_like(noise)
            pred_id, _, _, _, _, _ = self._apply_transport(
                sim_images, h_z, h_sim, h_sim, zero_noise, add_grain=False
            )
            out["pred_identity"] = pred_id

        return out


# Alias unico per flow_cygno_cluster.py
CygnoTransportModel = LocalDetectorTransportModel


# ==============================================================================
# 3. ESTRAZIONE DIFFERENZIABILE DEGLI SCALARI FISICI E PROFILI 2D
# ==============================================================================
def compute_physical_scalars_from_image(
    imgs, noise_threshold=noise_threshold, temperature=5.0
):
    """
    Estrae gli 8 descrittori fisici differenziabili sopra la soglia di zero-suppression:
    [log10(integral), length, width, log10(area), xmean, ymean, eccentricity, rel_peak]
    """
    B, C, H, W = imgs.shape
    device = imgs.device

    imgs_pos = F.relu(imgs)
    soft_mask = torch.sigmoid((imgs_pos - noise_threshold) * temperature)
    imgs_core = imgs_pos * soft_mask

    integral = imgs_core.sum(dim=(-1, -2)).squeeze(-1)
    soft_area = soft_mask.sum(dim=(-1, -2)).squeeze(-1)

    y_grid = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)
    x_grid = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, 1, W)

    norm = integral.view(B, 1, 1, 1) + 1e-8
    weights = imgs_core / norm

    x_mean = (weights * x_grid).sum(dim=(-1, -2)).squeeze(-1)
    y_mean = (weights * y_grid).sum(dim=(-1, -2)).squeeze(-1)

    dx = x_grid - x_mean.view(B, 1, 1, 1)
    dy = y_grid - y_mean.view(B, 1, 1, 1)

    cxx = (weights * (dx**2)).sum(dim=(-1, -2)).squeeze(-1)
    cyy = (weights * (dy**2)).sum(dim=(-1, -2)).squeeze(-1)
    cxy = (weights * (dx * dy)).sum(dim=(-1, -2)).squeeze(-1)

    trace = cxx + cyy
    det_term = torch.sqrt(torch.clamp((cxx - cyy) ** 2 + 4 * (cxy**2), min=1e-8))

    lambda_max = 0.5 * (trace + det_term)
    lambda_min = 0.5 * torch.clamp(trace - det_term, min=1e-6)

    length = 2.0 * torch.sqrt(lambda_max + 1e-6)
    width = 2.0 * torch.sqrt(lambda_min + 1e-6)
    eccentricity = torch.sqrt(torch.clamp(1.0 - (lambda_min / lambda_max), min=0.0))

    flat_imgs = imgs_core.view(B, -1)
    top_vals, _ = torch.topk(flat_imgs, k=4, dim=-1)
    peak_val = top_vals.mean(dim=-1)
    rel_peak = peak_val / (integral + 1e-8)

    return torch.stack(
        [
            torch.log10(integral + 1.0),
            length,
            width,
            torch.log10(soft_area + 1.0),
            x_mean,
            y_mean,
            eccentricity,
            rel_peak,
        ],
        dim=-1,
    )


def compute_centroids(imgs, noise_threshold=noise_threshold, temperature=5.0):
    B, C, H, W = imgs.shape
    device = imgs.device

    imgs_pos = F.relu(imgs)
    soft_mask = torch.sigmoid((imgs_pos - noise_threshold) * temperature)
    imgs_core = imgs_pos * soft_mask

    y_grid = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)
    x_grid = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, 1, W)

    norm = imgs_core.sum(dim=(-1, -2), keepdim=True) + 1e-8
    weights = imgs_core / norm

    x_mean = (weights * x_grid).sum(dim=(-1, -2)).squeeze(-1)
    y_mean = (weights * y_grid).sum(dim=(-1, -2)).squeeze(-1)
    return torch.stack([x_mean, y_mean], dim=-1)


def compute_radial_profile(imgs, num_bins=16):
    B, C, H, W = imgs.shape
    device = imgs.device

    imgs_pos = F.relu(imgs)
    y_grid = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)
    x_grid = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, 1, W)

    norm = imgs_pos.sum(dim=(-1, -2), keepdim=True) + 1e-8
    weights = imgs_pos / norm

    x_mean = (weights * x_grid).sum(dim=(-1, -2), keepdim=True)
    y_mean = (weights * y_grid).sum(dim=(-1, -2), keepdim=True)

    r = torch.sqrt((x_grid - x_mean) ** 2 + (y_grid - y_mean) ** 2 + 1e-8)
    bins = torch.arange(num_bins, device=device, dtype=torch.float32).view(
        1, num_bins, 1, 1
    )
    ring_masks = torch.exp(-0.5 * ((r - bins) / 0.8) ** 2)
    return (weights * ring_masks).sum(dim=(-1, -2))


_DIAG_PROJ_CACHE = {}


def get_diag_projection(H, W, device):
    key = (H, W, device)
    if key not in _DIAG_PROJ_CACHE:
        r = torch.arange(H).unsqueeze(1).expand(H, W)
        c = torch.arange(W).unsqueeze(0).expand(H, W)
        cols = []
        for offset in range(-H + 1, W):
            m1 = ((c - r) == offset).float().view(-1)
            cols.append(m1 / m1.sum().clamp(min=1.0))
        for offset in range(-H + 1, W):
            m2 = (((W - 1 - c) - r) == offset).float().view(-1)
            cols.append(m2 / m2.sum().clamp(min=1.0))
        _DIAG_PROJ_CACHE[key] = torch.stack(cols, dim=1).to(device)
    return _DIAG_PROJ_CACHE[key]


def extract_profiles_with_diagonals(img_tensor):
    B, C, H, W = img_tensor.shape
    img = img_tensor.view(B, H, W)
    prof_x = img.mean(dim=1)
    prof_y = img.mean(dim=2)
    proj = get_diag_projection(H, W, img.device)
    diags = img.reshape(B, H * W) @ proj
    return torch.cat([prof_x, prof_y, diags], dim=1)


# ==============================================================================
# 4. FUNZIONI DI LOSS
# ==============================================================================
def compute_mmd_loss(x, y, sigmas=(0.01, 0.05, 0.1, 0.5)):
    x_sq = (x**2).sum(dim=-1, keepdim=True)
    y_sq = (y**2).sum(dim=-1, keepdim=True)

    dist_xx = (
        x_sq + x_sq.transpose(-1, -2) - 2.0 * torch.matmul(x, x.transpose(-1, -2))
    ).clamp(min=0.0)
    dist_yy = (
        y_sq + y_sq.transpose(-1, -2) - 2.0 * torch.matmul(y, y.transpose(-1, -2))
    ).clamp(min=0.0)
    dist_xy = (
        x_sq + y_sq.transpose(-1, -2) - 2.0 * torch.matmul(x, y.transpose(-1, -2))
    ).clamp(min=0.0)

    mmd = 0.0
    for s in sigmas:
        gamma = 1.0 / (2.0 * (s**2))
        k_xx = torch.exp(-gamma * dist_xx).mean(dim=(-2, -1))
        k_yy = torch.exp(-gamma * dist_yy).mean(dim=(-2, -1))
        k_xy = torch.exp(-gamma * dist_xy).mean(dim=(-2, -1))
        mmd = mmd + (k_xx + k_yy - 2.0 * k_xy).mean()
    return mmd / len(sigmas)


def wasserstein_1d_loss(pred_vec, target_vec, mode="l1"):
    p_idx = torch.argsort(pred_vec.detach().cpu(), dim=-1).to(pred_vec.device)
    t_idx = torch.argsort(target_vec.detach().cpu(), dim=-1).to(target_vec.device)
    p_s = torch.gather(pred_vec, -1, p_idx)
    t_s = torch.gather(target_vec, -1, t_idx)
    loss_fn = F.mse_loss if mode == "mse" else F.l1_loss
    return loss_fn(p_s, t_s)


def dist_match_loss(pred_vec, target_vec, mode="l1", robust_outliers=False):
    """
    Combina Wasserstein 1D + allineamento della posizione centrale (mediana).
    Non forza la varianza globale su bin bimodali così i picchi stretti restano stretti.
    """
    w_loss = wasserstein_1d_loss(pred_vec, target_vec, mode=mode)
    if robust_outliers:
        med_loss = F.smooth_l1_loss(
            pred_vec.median(dim=1).values,
            target_vec.median(dim=1).values,
            beta=0.25,
        )
        p_q75, p_q25 = (
            torch.quantile(pred_vec, 0.75, dim=1),
            torch.quantile(pred_vec, 0.25, dim=1),
        )
        t_q75, t_q25 = (
            torch.quantile(target_vec, 0.75, dim=1),
            torch.quantile(target_vec, 0.25, dim=1),
        )
        iqr_loss = F.smooth_l1_loss(p_q75 - p_q25, t_q75 - t_q25, beta=0.25)
        return w_loss + med_loss + 0.5 * iqr_loss
    else:
        med_loss = F.mse_loss(
            pred_vec.median(dim=1).values,
            target_vec.median(dim=1).values,
        )
        return w_loss + 2.0 * med_loss


def get_loss_weights(epoch, total_epochs=50):
    return {
        "mmd_shape_1d": 3500.0,
        "radial": 80.0,
        "geometry_1d": 60.0,
        "integral": 5000.0,
        "mmd_physics": 80.0,
        "centroid": 100.0,
        "shape_anchor": 20.0,
        "transport": 200.0,
        "identity": 50.0,
    }


def compute_cygno_loss(
    pred,
    delta_h,
    loss_weights,
    pred_identity=None,
    sim_images=None,
    data_images=None,
    noise_threshold=noise_threshold,
    batch_shape=None,
):
    M = pred.shape[0]
    B, N = batch_shape if batch_shape is not None else (1, M)

    pred_clamped = F.relu(pred)
    data_clamped = F.relu(data_images)
    sim_clamped = F.relu(sim_images) if sim_images is not None else None

    pred_core = pred_clamped * torch.sigmoid(
        (pred_clamped - noise_threshold) * 5.0
    )
    data_core = data_clamped * torch.sigmoid(
        (data_clamped - noise_threshold) * 5.0
    )

    pred_n = pred_core / (pred_core.sum(dim=(-1, -2), keepdim=True) + 1e-8)
    sim_n = (
        sim_clamped / (sim_clamped.sum(dim=(-1, -2), keepdim=True) + 1e-8)
        if sim_clamped is not None
        else None
    )
    data_n = data_core / (data_core.sum(dim=(-1, -2), keepdim=True) + 1e-8)

    prof_pred = extract_profiles_with_diagonals(pred_n).view(B, N, -1)
    prof_target = extract_profiles_with_diagonals(data_n).view(B, N, -1)
    L_mmd_shape_1d = compute_mmd_loss(
        prof_pred, prof_target, sigmas=(0.01, 0.05, 0.1, 0.5)
    )

    rad_pred = compute_radial_profile(pred_core, num_bins=16).view(B, N, 16)
    rad_target = compute_radial_profile(data_core, num_bins=16).view(B, N, 16)
    L_radial = compute_mmd_loss(
        rad_pred, rad_target, sigmas=(0.01, 0.05, 0.1, 0.2)
    )

    if sim_n is not None:
        weight_mask = 1.0 + 10.0 * (
            sim_n / (sim_n.amax(dim=(-1, -2), keepdim=True) + 1e-8)
        )
        L_shape_anchor = (weight_mask * torch.abs(pred_n - sim_n)).mean()
    else:
        L_shape_anchor = torch.tensor(0.0, device=pred.device)

    if sim_clamped is not None:
        pred_c = compute_centroids(pred_clamped, noise_threshold=noise_threshold)
        sim_c = compute_centroids(sim_clamped, noise_threshold=noise_threshold)
        L_centroid = F.mse_loss(pred_c, sim_c)
    else:
        L_centroid = torch.tensor(0.0, device=pred.device)

    pred_physics_feat = compute_physical_scalars_from_image(
        pred_clamped, noise_threshold=noise_threshold
    ).view(B, N, 8)
    target_physics_feat = compute_physical_scalars_from_image(
        data_clamped, noise_threshold=noise_threshold
    ).view(B, N, 8)

    L_integral = dist_match_loss(
        pred_physics_feat[:, :, 0],
        target_physics_feat[:, :, 0],
        mode="mse",
        robust_outliers=False,
    )
    L_length = dist_match_loss(
        pred_physics_feat[:, :, 1],
        target_physics_feat[:, :, 1],
        mode="l1",
        robust_outliers=True,
    )
    L_width = dist_match_loss(
        pred_physics_feat[:, :, 2],
        target_physics_feat[:, :, 2],
        mode="l1",
        robust_outliers=True,
    )
    L_eccentricity = dist_match_loss(
        pred_physics_feat[:, :, 6],
        target_physics_feat[:, :, 6],
        mode="l1",
        robust_outliers=True,
    )
    L_rel_peak = wasserstein_1d_loss(
        pred_physics_feat[:, :, 7], target_physics_feat[:, :, 7], mode="l1"
    )
    L_geometry_1d = (
        1.5 * L_length
        + 2.5 * L_width
        + 2.0 * L_eccentricity
        + 2.0 * L_rel_peak
    )

    feat_mean = target_physics_feat.mean(dim=1, keepdim=True)
    feat_std = target_physics_feat.std(dim=1, keepdim=True) + 1e-5
    pred_feat_norm = (pred_physics_feat - feat_mean) / feat_std
    target_feat_norm = (target_physics_feat - feat_mean) / feat_std
    L_mmd_physics = compute_mmd_loss(
        pred_feat_norm, target_feat_norm, sigmas=(0.5, 1.0, 2.0)
    )

    if sim_n is not None:
        L_transport = F.mse_loss(pred_n, sim_n)
    else:
        L_transport = (delta_h**2).mean()

    if pred_identity is not None and sim_clamped is not None:
        id_clamped = F.relu(pred_identity)
        id_n = id_clamped / (id_clamped.sum(dim=(-1, -2), keepdim=True) + 1e-8)
        sim_int = torch.log10(sim_clamped.sum(dim=(-1, -2)).squeeze(-1) + 1.0)
        id_int = torch.log10(id_clamped.sum(dim=(-1, -2)).squeeze(-1) + 1.0)
        L_identity = 100.0 * F.l1_loss(id_n, sim_n) + F.mse_loss(id_int, sim_int)
    else:
        L_identity = torch.tensor(0.0, device=pred.device)

    w = loss_weights
    t_mmd_shape = w["mmd_shape_1d"] * L_mmd_shape_1d
    t_radial = w["radial"] * L_radial
    t_anchor = w["shape_anchor"] * L_shape_anchor
    t_centroid = w["centroid"] * L_centroid
    t_integral = w["integral"] * L_integral
    t_geom = w["geometry_1d"] * L_geometry_1d
    t_mmd_phys = w["mmd_physics"] * L_mmd_physics
    t_transp = w["transport"] * L_transport
    t_id = w["identity"] * L_identity

    total_loss = (
        t_mmd_shape
        + t_radial
        + t_anchor
        + t_centroid
        + t_integral
        + t_geom
        + t_mmd_phys
        + t_transp
        + t_id
    )

    vals = (
        torch.stack(
            [
                total_loss.detach(),
                t_mmd_shape.detach(),
                t_radial.detach(),
                t_anchor.detach(),
                t_centroid.detach(),
                t_integral.detach(),
                t_geom.detach(),
                t_mmd_phys.detach(),
                t_transp.detach(),
                t_id.detach(),
            ]
        )
        .cpu()
        .tolist()
    )

    keys = [
        "total",
        "mmd_shape_1d",
        "radial",
        "shape_anchor",
        "centroid",
        "integral",
        "geometry_1d",
        "mmd_physics",
        "transport",
        "identity",
    ]
    return total_loss, dict(zip(keys, vals))


# ==============================================================================
# 5. TRAINING LOOP E HELPER
# ==============================================================================
def train_epoch(
    model, dataloader, optimizer, loss_weights, device, max_batches=80
):
    model.train()
    stats = defaultdict(float)
    steps = 0

    for ibatch, batch in enumerate(dataloader):
        if max_batches is not None and ibatch >= max_batches:
            break
        if ibatch % 10 == 0:
            print(f"\t\t  running ibatch {ibatch}...", flush=True)

        sim_images = F.relu(batch["sim_images"]).to(device)
        data_images = F.relu(batch["data_images"]).to(device)
        sim_cond = batch["sim_cond"].to(device)
        data_cond = batch["data_cond"].to(device)
        sim_scalars_phys = batch["sim_scalars"].to(device)

        B, N, H, W = sim_images.shape

        s_img = sim_images.reshape(B * N, 1, H, W)
        d_img = data_images.reshape(B * N, 1, H, W)
        s_cond = (
            sim_cond.reshape(B * N, -1)
            if sim_cond.ndim == 3
            else sim_cond.repeat_interleave(N, dim=0)
        )
        d_cond = (
            data_cond.reshape(B * N, -1)
            if data_cond.ndim == 3
            else data_cond.repeat_interleave(N, dim=0)
        )
        s_scal = sim_scalars_phys.reshape(B * N, -1)

        optimizer.zero_grad()

        out = model(s_img, s_cond, d_cond, s_scal, compute_identity=True)
        loss_batch, info_batch = compute_cygno_loss(
            pred=out["pred_images"],
            delta_h=out["delta_h"],
            loss_weights=loss_weights,
            pred_identity=out["pred_identity"],
            sim_images=s_img,
            data_images=d_img,
            batch_shape=(B, N),
        )

        loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        for k, v in info_batch.items():
            stats[k] += v
        steps += 1

    for k in stats:
        stats[k] /= max(steps, 1)

    return dict(stats)


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
    return ckpt


def _compute_condition_stats(dataset, loader=None):
    if isinstance(dataset, str):
        if loader is not None and hasattr(loader, "dataset"):
            dataset = loader.dataset
        else:
            with open(dataset, "rb") as f:
                payload = pickle.load(f)
            sim_keys = np.asarray(list(payload["sim"].keys()), dtype=np.float32)
            data_keys = np.asarray(
                list(payload["data"].keys()), dtype=np.float32
            )
            return (
                sim_keys.mean(axis=0),
                np.maximum(sim_keys.std(axis=0), 1e-6),
                data_keys.mean(axis=0),
                np.maximum(data_keys.std(axis=0), 1e-6),
            )

    if hasattr(dataset, "sim_dict"):
        sim_keys = np.asarray(list(dataset.sim_dict.keys()), dtype=np.float32)
    else:
        sim_keys = np.asarray(
            [k for keys in dataset.sim_keys_by_z.values() for k in keys],
            dtype=np.float32,
        )

    if hasattr(dataset, "data_dict"):
        data_keys = np.asarray(list(dataset.data_dict.keys()), dtype=np.float32)
    else:
        data_keys = np.asarray(
            [k for keys in dataset.data_keys_by_z.values() for k in keys],
            dtype=np.float32,
        )

    sim_mean = sim_keys.mean(axis=0)
    sim_std = np.maximum(sim_keys.std(axis=0), 1e-6)
    data_mean = data_keys.mean(axis=0)
    data_std = np.maximum(data_keys.std(axis=0), 1e-6)
    return sim_mean, sim_std, data_mean, data_std


def _resolve_device(device):
    if device is None or str(device).lower() in ("none", "auto", ""):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def _resolve_dataset_and_loader(
    dataset,
    loader,
    device=None,
    file_arg=None,
    batch_size=32,
    n_clusters=32,
    is_test=False,
):
    if isinstance(dataset, str):
        inputfile = dataset
        if isinstance(loader, str):
            if loader.startswith(("cpu", "mps", "cuda")):
                device = loader
            else:
                file_arg = loader
        dataset, loader = build_dataloader(
            inputfile,
            batch_size=batch_size,
            n_clusters=n_clusters,
            shuffle=not is_test,
            is_test=is_test,
            num_workers=0,
        )
    device = _resolve_device(device)
    return dataset, loader, device, file_arg


def build_model_from_dataset(
    dataset, device, loader=None, kernel_size=11, cond_dim=64, noise_dim=2
):
    sim_mean, sim_std, data_mean, data_std = _compute_condition_stats(
        dataset, loader=loader
    )
    model = LocalDetectorTransportModel(
        kernel_size=kernel_size,
        cond_dim=cond_dim,
        sim_dim=len(sim_mean),
        data_dim=len(data_mean),
        noise_dim=noise_dim,
    ).to(device)
    model.cond_encoder.set_normalization(sim_mean, sim_std, data_mean, data_std)
    return model


def forward_test(
    dataset, loader=None, device="cpu", batch_size=32, n_clusters=32, **kwargs
):
    dataset, loader, device, _ = _resolve_dataset_and_loader(
        dataset,
        loader,
        device=device,
        batch_size=batch_size,
        n_clusters=n_clusters,
        is_test=False,
    )
    model = build_model_from_dataset(dataset, device=device, loader=loader)

    batch = next(iter(loader))
    sim_images = F.relu(batch["sim_images"]).to(device)
    sim_cond = batch["sim_cond"].to(device)
    data_cond = batch["data_cond"].to(device)
    sim_scalars = batch["sim_scalars"].to(device)

    B, N, H, W = sim_images.shape
    s_img = sim_images[0].unsqueeze(1)
    s_cond = sim_cond[0].unsqueeze(0).repeat(N, 1)
    d_cond = data_cond[0].unsqueeze(0).repeat(N, 1)
    s_scal = sim_scalars[0]

    out = model(s_img, s_cond, d_cond, s_scal)
    print("Predicted images shape:", out["pred_images"].shape)
    print("Delta_h shape:", out["delta_h"].shape)
    print("Mean gain at init:", out["gain"].mean().item())


def train_model(
    dataset,
    loader=None,
    device="cpu",
    epochs=50,
    lr=1e-3,
    outputfile="model_last.pt",
    checkpoint=None,
    plot_losses=True,
    plotdir="plots",
    batch_size=32,
    n_clusters=32,
    max_batches=80,
    **kwargs,
):
    for alias in ("steps_per_epoch", "n_batches", "max_steps"):
        if alias in kwargs and kwargs[alias] is not None:
            max_batches = kwargs[alias]

    if "outputmodel" in kwargs and kwargs["outputmodel"] is not None:
        outputfile = kwargs["outputmodel"]

    dataset, loader, device, outputfile = _resolve_dataset_and_loader(
        dataset,
        loader,
        device=device,
        file_arg=outputfile,
        batch_size=batch_size,
        n_clusters=n_clusters,
        is_test=False,
    )

    print(
        "=== TRAIN THE MODEL (v3.0 Physics-Informed GEM Saturation + 1:1 z) ==="
    )
    os.makedirs(os.path.dirname(os.path.abspath(outputfile)), exist_ok=True)

    model = build_model_from_dataset(dataset, device=device, loader=loader)

    print(f"Initialized LocalDetectorTransportModel on device {device}...")
    print(f"  SIM cond mean: {model.cond_encoder.sim_mean.cpu().numpy().ravel()}")
    print(f"  SIM cond std : {model.cond_encoder.sim_std.cpu().numpy().ravel()}")
    print(f"  DATA cond mean: {model.cond_encoder.data_mean.cpu().numpy().ravel()}")
    print(f"  DATA cond std : {model.cond_encoder.data_std.cpu().numpy().ravel()}")

    start_epoch = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if checkpoint is not None and os.path.exists(checkpoint):
        print(f"Loading model state dict from {checkpoint}")
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(_extract_state_dict(ckpt))
        if isinstance(ckpt, dict) and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if isinstance(ckpt, dict) and "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - start_epoch, 1), eta_min=1e-5
    )

    history = defaultdict(list)
    best_val_loss = float("inf")

    for epoch in range(start_epoch, epochs):
        print(f"\t|Start epoch n. {epoch}...")
        loss_weights = get_loss_weights(epoch, epochs)

        stats = train_epoch(
            model,
            loader,
            optimizer,
            loss_weights,
            device,
            max_batches=max_batches,
        )

        scheduler.step()

        for k, v in stats.items():
            history[k].append(v)

        print(f"epoch {epoch}")
        for k, v in stats.items():
            print(f"{k}: {v:.4f}")

        if np.isfinite(stats["total"]) and stats["total"] < best_val_loss:
            best_val_loss = stats["total"]
            best_path = (
                outputfile.replace("last", "best")
                if "last" in os.path.basename(outputfile)
                else outputfile
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                best_path,
            )
            print(
                f"Nuovo miglior modello salvato all'epoca {epoch} con loss ="
                f" {stats['total']:.4f}"
            )

    if plot_losses:
        plot_training_history(history, output_dir=plotdir)

    torch.save(
        {
            "epoch": epochs - 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        outputfile,
    )

    best_path = (
        outputfile.replace("last", "best")
        if "last" in os.path.basename(outputfile)
        else outputfile
    )
    if os.path.exists(best_path):
        best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(_extract_state_dict(best_ckpt))

    test_training(model, dataset, output_dir=plotdir, device=device)

    return model, dict(history)


def _get_root_rainbow_cmap():
    base = plt.get_cmap("rainbow")
    colors = np.vstack(
        [
            np.array([[1.0, 1.0, 1.0, 1.0]]),  # bianco per zero
            base(np.linspace(0, 1, 255)),
        ]
    )
    return mcolors.ListedColormap(colors)


# ==============================================================================
# 6. TEST VISIVO SU 10 CLUSTER RANDOM + VALIDAZIONE VETTORIZZATA SU GPU
# ==============================================================================
def test_training(
    model_or_path,
    inputfile,
    output_dir="plot/validation_plots",
    device=None,
    **kwargs,
):
    if "plotdir" in kwargs and kwargs["plotdir"] is not None:
        output_dir = kwargs["plotdir"]

    device = _resolve_device(device)

    if isinstance(model_or_path, nn.Module):
        model = model_or_path.to(device)
    else:
        if (
            isinstance(inputfile, str)
            and inputfile.endswith((".pt", ".pth"))
            and not str(model_or_path).endswith((".pt", ".pth"))
        ):
            model_or_path, inputfile = inputfile, model_or_path

        print(f"\nLoading model from:\n{model_or_path}")
        model = CygnoTransportModel().to(device)
        state = torch.load(
            model_or_path, map_location=device, weights_only=False
        )
        if isinstance(state, dict) and "model_state_dict" in state:
            model_state = state["model_state_dict"]
            current_epoch = state.get("epoch", "?")
            print(
                f"===> Rilevato checkpoint completo. Estraggo 'model_state_dict', all'epoca {current_epoch}"
            )
        else:
            model_state = state
        model.load_state_dict(model_state)

    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(inputfile, str):
        _, loader = build_dataloader(
            inputfile, batch_size=1, is_test=True, shuffle=True, num_workers=0
        )
    else:
        loader = DataLoader(
            inputfile,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            collate_fn=_PicklableCygnoCollate(inputfile),
        )

    cmap = _get_root_rainbow_cmap()

    test_sims = []
    test_preds = []
    test_datas = []
    test_gains = []
    test_sats = []
    test_cond_sim_labels = []
    test_cond_data_labels = []

    with torch.inference_mode():
        for ibatch, batch in enumerate(loader):
            if ibatch >= 10:
                break

            sim = F.relu(batch["sim_images"]).to(device)
            sim_cond = batch["sim_cond"].to(device)
            data_cond = batch["data_cond"].to(device)

            B, N, H, W = sim.shape
            s_img = sim[0].unsqueeze(1)
            s_c = (
                sim_cond[0]
                if sim_cond.ndim == 3
                else sim_cond[0].unsqueeze(0).repeat(N, 1)
            )
            d_c = (
                data_cond[0]
                if data_cond.ndim == 3
                else data_cond[0].unsqueeze(0).repeat(N, 1)
            )
            s_scal = compute_physical_scalars_from_image(s_img)

            out = model(s_img, s_c, d_c, s_scal)
            pred_img = F.relu(out["pred_images"])
            pred_img = torch.where(
                pred_img > noise_threshold,
                pred_img,
                torch.zeros_like(pred_img),
            )

            test_sims.append(sim[0, 0].cpu())
            test_preds.append(pred_img[0, 0].cpu())
            test_datas.append(F.relu(batch["data_images"][0, 0]).cpu())
            test_gains.append(out["gain"][0].item())
            test_sats.append(out["sat_alpha"][0].item())

            sim_label = (
                f"z={s_c[0, 0].item():.1f}cm,\n"
                f"$\\alpha$={s_c[0, 1].item():.4f},\n"
                f"$\\lambda$={s_c[0, 2].item():.0f}mm"
            )
            test_cond_sim_labels.append(sim_label)

            data_label = (
                f"z={d_c[0, 0].item():.1f}cm,\n"
                f"P={d_c[0, 1].item():.3f}bar,\n"
                f"T={d_c[0, 2].item():.1f},\n"
                f"H={d_c[0, 3].item():.1f}ppk"
            )
            test_cond_data_labels.append(data_label)

    n_cols = len(test_sims)
    if n_cols == 0:
        return model

    fig, ax = plt.subplots(nrows=3, ncols=n_cols, figsize=(2.4 * n_cols, 9.5))
    if n_cols == 1:
        ax = ax.reshape(3, 1)

    for i in range(n_cols):
        vmax = max(
            test_sims[i].max().item(),
            test_preds[i].max().item(),
            test_datas[i].max().item(),
            1.0,
        )

        im_sim = ax[0, i].imshow(
            test_sims[i], origin="lower", vmin=0, vmax=vmax, cmap=cmap
        )
        ax[0, i].set_title(f"SIM (Ex {i})", fontsize=10)
        ax[0, i].text(
            0.5,
            -0.24,
            test_cond_sim_labels[i],
            transform=ax[0, i].transAxes,
            ha="center",
            va="top",
            fontsize=8.5,
            bbox=dict(
                boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.5
            ),
        )

        im_pred = ax[1, i].imshow(
            test_preds[i], origin="lower", vmin=0, vmax=vmax, cmap=cmap
        )
        ax[1, i].set_title(
            f"CORR (g={test_gains[i]:.2f}, s={test_sats[i]:+.2f})",
            fontsize=9.5,
        )
        ax[1, i].text(
            0.5,
            -0.24,
            test_cond_data_labels[i],
            transform=ax[1, i].transAxes,
            ha="center",
            va="top",
            fontsize=8.5,
            bbox=dict(
                boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.5
            ),
        )

        im_data = ax[2, i].imshow(
            test_datas[i], origin="lower", vmin=0, vmax=vmax, cmap=cmap
        )
        ax[2, i].set_title(f"DATA (Ex {i})", fontsize=10)

        for row, im in enumerate((im_sim, im_pred, im_data)):
            cbar = fig.colorbar(im, ax=ax[row, i], fraction=0.046, pad=0.05)
            cbar.ax.tick_params(labelsize=7)
            if i == 0:
                cbar.set_label("Counts", fontsize=8)

    plt.subplots_adjust(hspace=0.52, wspace=0.38)
    cluster_test_path = os.path.join(output_dir, "clusters10_test.png")
    plt.savefig(cluster_test_path, dpi=140, bbox_inches="tight")
    if output_dir != "plots":
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/clusters10_test.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"\t--> Griglia con 10 clusters random salvata in: {cluster_test_path}")

    return model


def plot_training_history(history, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history["total"]) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(
        epochs, history["total"], label="Total Loss", color="black", linewidth=2
    )
    plt.xlabel("Epoca")
    plt.ylabel("Loss Value")
    plt.title("Andamento della Total Loss durante il Training")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history_loss_total.png"))
    plt.close()

    plt.figure(figsize=(12, 7))
    for key, values in history.items():
        if key == "total":
            continue
        plt.plot(epochs, values, label=key, linewidth=1.5)

    plt.xlabel("Epoca")
    plt.ylabel("Loss Component Value")
    plt.title("Andamento delle Componenti della Loss")
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "training_history_loss_components.png")
    )
    plt.close()


@torch.inference_mode()
def run_sampled_and_detailed_test(
    model,
    test_loader,
    device,
    max_batches=80,
    num_sim_ctx_to_sample=3,
    num_data_ctx_to_sample=3,
    save_individual_plots=False,
    output_dir="plots",
):
    device = _resolve_device(device)
    model.to(device).eval()
    os.makedirs(output_dir, exist_ok=True)

    pair_dict = defaultdict(list)

    print(f"Now accumulating statistics over max {max_batches} batches")
    for batch_idx, batch in enumerate(test_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        sim_images = F.relu(batch["sim_images"]).to(device)
        data_images = F.relu(batch["data_images"]).to(device)
        B, N, H, W = sim_images.shape

        s_img = sim_images.reshape(B * N, 1, H, W)
        d_img = data_images.reshape(B * N, 1, H, W)

        sim_cond = batch["sim_cond"].to(device)
        data_cond = batch["data_cond"].to(device)
        s_c = (
            sim_cond.reshape(B * N, -1)
            if sim_cond.ndim == 3
            else sim_cond.repeat_interleave(N, dim=0)
        )
        d_c = (
            data_cond.reshape(B * N, -1)
            if data_cond.ndim == 3
            else data_cond.repeat_interleave(N, dim=0)
        )

        s_scal = compute_physical_scalars_from_image(s_img)
        d_scal = compute_physical_scalars_from_image(d_img)

        out = model(s_img, s_c, d_c, s_scal)
        pred_img_clamped = F.relu(out["pred_images"])
        pred_scalars_clamped = compute_physical_scalars_from_image(
            pred_img_clamped
        )

        s_c_np = np.round(s_c.cpu().numpy(), 3)
        d_c_np = np.round(d_c.cpu().numpy(), 3)
        s_scal_np = s_scal.cpu().numpy()
        p_scal_np = pred_scalars_clamped.cpu().numpy()
        d_scal_np = d_scal.cpu().numpy()

        for idx in range(B * N):
            key = (tuple(s_c_np[idx]), tuple(d_c_np[idx]))
            pair_dict[key].append(
                {
                    "sim_val": s_scal_np[idx],
                    "pred_val": p_scal_np[idx],
                    "data_val": d_scal_np[idx],
                }
            )

    sorted_keys = sorted(
        pair_dict.keys(), key=lambda k: len(pair_dict[k]), reverse=True
    )
    if not sorted_keys:
        print("[CRITICO] Nessuna coppia trovata nel dataset di test.")
        return

    n_plots = num_sim_ctx_to_sample * num_data_ctx_to_sample
    coppie_selezionate = sorted_keys[: min(n_plots, len(sorted_keys))]

    whitelist_recomputed = [
        "integral_recomputed",
        "length_recomputed",
        "width_recomputed",
    ]
    num_scalars = 3
    n_cols = min(3, num_data_ctx_to_sample)
    n_rows = (len(coppie_selezionate) + n_cols - 1) // n_cols

    for idx_var in range(num_scalars):
        var_name = whitelist_recomputed[idx_var]
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
        )
        axes_flat = axes.flatten()

        for idx_coppia, coppia in enumerate(coppie_selezionate):
            ax = axes_flat[idx_coppia]
            s_ctx, d_ctx = np.array(coppia[0]), np.array(coppia[1])
            events = pair_dict[coppia]

            s_vals = np.array([e["sim_val"][idx_var] for e in events])
            p_vals = np.array([e["pred_val"][idx_var] for e in events])
            d_vals = np.array([e["data_val"][idx_var] for e in events])

            all_vals = np.concatenate([s_vals, p_vals, d_vals])
            vmin, vmax = np.percentile(all_vals, 1.0), np.percentile(
                all_vals, 99.0
            )
            bins = np.linspace(vmin, vmax, 30)

            ax.hist(
                s_vals,
                bins=bins,
                alpha=0.5,
                histtype="step",
                linewidth=2,
                label="SIM",
                color="tab:blue",
                density=True,
            )
            ax.hist(
                p_vals,
                bins=bins,
                alpha=0.7,
                histtype="step",
                linewidth=2,
                label="PRED",
                color="tab:orange",
                density=True,
            )

            counts, bin_edges = np.histogram(d_vals, bins=bins)
            counts_density, _ = np.histogram(d_vals, bins=bins, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

            valid_bins = counts > 0
            scaling_factor = np.divide(
                counts_density,
                counts,
                out=np.zeros_like(counts_density),
                where=(counts > 0),
            )
            errors_density = np.sqrt(counts) * scaling_factor

            ax.errorbar(
                bin_centers[valid_bins],
                counts_density[valid_bins],
                yerr=errors_density[valid_bins],
                fmt="o",
                markersize=4,
                color="black",
                capsize=2,
                label="DATA",
            )

            sim_str = (
                r"$\mathbf{SIM:}$ "
                + f"z={s_ctx[0]:.1f}, "
                + r"$\alpha$="
                + f"{s_ctx[1]:.3f}, "
                + r"$\lambda$="
                + f"{s_ctx[2]:.2f}"
            )
            data_str = (
                r"$\mathbf{DATA:}$ "
                + f"z={d_ctx[0]:.1f}, "
                + r"$P$="
                + f"{d_ctx[1]:.2f}, "
                + r"$T$="
                + f"{d_ctx[2]:.1f}, "
                + r"$H$="
                + f"{d_ctx[3]:.2f}"
            )

            ax.set_title(
                f"{sim_str}\n{data_str}", fontsize=8.5, pad=8, loc="left"
            )
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.tick_params(axis="both", which="major", labelsize=8)
            ax.legend(loc="upper right", fontsize=7)

        for i in range(idx_coppia + 1, len(axes_flat)):
            axes_flat[i].axis("off")

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"griglia_ottimizzata_{var_name}.png"),
            dpi=120,
        )
        plt.close()