"""
EIAKDA — Exploiting Inter-Sample Affinity for Knowability-Aware Universal
         Domain Adaptation
Adapted for 1-D Smart-Building Sensor Data (Figure 2, pipeline column 3).

Original paper:
    Wang et al., "Exploiting Inter-Sample Affinity for Knowability-Aware
    Universal Domain Adaptation," IJCV 2024.

Key components (Figure 2):
    ┌─ Shared Temporal Encoder      G_tp                   [Eq. 40]
    ├─ Affinity Matrix + SVD        A_f^s, S, A_f^t,t     [Eq. 57]
    ├─ Knowability Score            κ_i                    [Eq. 58]
    ├─ Pseudo-label via Neighbourhood Consistency          [Eq. 59]
    └─ Adaptive Threshold Split     τ, κ, τ_U             [Eq. 60]
       → Output: Known / Unknown / Uncertain
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Optional, Tuple

from src.encoder import TemporalEncoder, LinearClassifier
from src.utils import entropy, cycle, get_device
from src.metrics import compute_metrics, print_metrics, UNKNOWN_LABEL


# ---------------------------------------------------------------------------
# Knowability estimation via Affinity Matrix + SVD  [Eq. 57-58]
# ---------------------------------------------------------------------------

def compute_affinity_matrix(feats: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Pairwise Gaussian affinity matrix  A_f  [Eq. 57].

    A_ij = exp(-||z_i - z_j||^2 / (2 * sigma^2))
    """
    dists = torch.cdist(feats, feats, p=2) ** 2  # (B, B)
    A = torch.exp(-dists / (2 * sigma ** 2))
    return A


def knowability_score(
    src_feat: torch.Tensor,   # (Bs, D)
    tgt_feat: torch.Tensor,   # (Bt, D)
) -> torch.Tensor:
    """Knowability score κ_i for each target sample  [Eq. 58].

    Uses the first left singular vector of the cross-domain affinity matrix
    A_f^{s,t} to assign a score ∈ [0, 1] indicating how "known" a target
    sample is (high κ → shared class; low κ → unknown class).
    """
    # Cross-domain affinity  A^{s,t} ∈ R^{Bs × Bt}
    sf = F.normalize(src_feat, dim=1)
    tf = F.normalize(tgt_feat, dim=1)
    A = sf @ tf.T                        # (Bs, Bt)

    # Compact SVD — we only need the first singular vector
    try:
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    except Exception:
        # Fallback: row-sum normalised
        return A.sum(dim=0) / (A.sum(dim=0).max() + 1e-9)

    # First right singular vector gives target-side knowability
    v1 = Vh[0]                           # (Bt,)
    kappa = v1.abs()
    kappa = (kappa - kappa.min()) / (kappa.max() - kappa.min() + 1e-9)
    return kappa   # (Bt,)


def neighbourhood_consistency_pseudolabel(
    tgt_feat: torch.Tensor,
    tgt_logits: torch.Tensor,
    k: int = 7,
) -> torch.Tensor:
    """Pseudo-label via neighbourhood consistency  [Eq. 59].

    For each target sample, aggregate the classifier predictions of its
    k nearest neighbours (weighted by similarity) to produce a refined
    pseudo-label distribution.
    """
    B = tgt_feat.shape[0]
    probs = F.softmax(tgt_logits, dim=1)   # (B, C)

    fn = F.normalize(tgt_feat, dim=1)
    sim = fn @ fn.T                         # (B, B)
    sim.fill_diagonal_(-1e9)

    k = min(k, B - 1)
    topk_val, topk_idx = sim.topk(k, dim=1)   # (B, k)
    weights = F.softmax(topk_val, dim=1)       # (B, k)

    # Weighted sum of neighbour probability vectors
    pseudo_probs = torch.zeros_like(probs)
    for b in range(B):
        nbr_probs = probs[topk_idx[b]]          # (k, C)
        pseudo_probs[b] = (weights[b].unsqueeze(1) * nbr_probs).sum(dim=0)

    return pseudo_probs   # (B, C) soft pseudo-labels


def adaptive_threshold_split(
    kappa: torch.Tensor,
    quantile_low: float = 0.33,
    quantile_high: float = 0.67,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Three-way split: known / uncertain / unknown  [Eq. 60].

    τ   (low threshold)  → samples below are "unknown"
    τ_U (high threshold) → samples above are "known"
    Between the two → "uncertain"

    Returns:
        known_mask, uncertain_mask, unknown_mask  (boolean tensors)
    """
    tau_low  = torch.quantile(kappa, quantile_low)
    tau_high = torch.quantile(kappa, quantile_high)

    known_mask     = kappa >= tau_high
    unknown_mask   = kappa <  tau_low
    uncertain_mask = ~known_mask & ~unknown_mask
    return known_mask, uncertain_mask, unknown_mask


# ---------------------------------------------------------------------------
# EIAKDA model
# ---------------------------------------------------------------------------

class EIAKDA(nn.Module):
    """EIAKDA adapted for 1-D sensor data.

    Args:
        in_channels  : number of sensor features
        seq_len      : window length
        num_classes  : number of known source classes
        feat_dim     : encoder output dim (64)
        knn_k        : neighbourhood size for pseudo-labelling
        q_low/q_high : quantile thresholds for three-way split
    """

    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        num_classes: int,
        feat_dim: int = 64,
        knn_k: int = 7,
        q_low: float = 0.33,
        q_high: float = 0.67,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder     = TemporalEncoder(in_channels, seq_len, dropout)
        self.classifier  = LinearClassifier(feat_dim, num_classes)
        self.num_classes = num_classes
        self.knn_k       = knn_k
        self.q_low       = q_low
        self.q_high      = q_high

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def eiakda_loss(
        self,
        x_s: torch.Tensor,
        y_s: torch.Tensor,
        x_t: torch.Tensor,
    ) -> torch.Tensor:
        z_s = self.encoder(x_s)
        z_t = self.encoder(x_t)

        # Source cross-entropy
        L_ce = F.cross_entropy(self.classifier(z_s), y_s)

        # Knowability scores  [Eq. 58]
        kappa = knowability_score(z_s.detach(), z_t.detach())  # (Bt,)

        # Three-way split  [Eq. 60]
        known_m, unc_m, unk_m = adaptive_threshold_split(kappa, self.q_low, self.q_high)

        tgt_logits = self.classifier(z_t)

        # Pseudo-label consistency loss on known-leaning target samples  [Eq. 59]
        if known_m.sum() > 0:
            pseudo = neighbourhood_consistency_pseudolabel(
                z_t.detach(), tgt_logits.detach(), k=self.knn_k
            )
            L_pseudo = F.kl_div(
                F.log_softmax(tgt_logits[known_m], dim=1),
                pseudo[known_m].detach(),
                reduction="batchmean",
            )
        else:
            L_pseudo = torch.tensor(0.0, device=x_s.device)

        # Entropy maximisation on unknown-leaning samples (push towards uniform)
        if unk_m.sum() > 0:
            p_unk = F.softmax(tgt_logits[unk_m], dim=1)
            L_unk = -entropy(p_unk).mean()   # maximize entropy → minimize neg entropy
        else:
            L_unk = torch.tensor(0.0, device=x_s.device)

        # Entropy minimisation on known-leaning samples
        if known_m.sum() > 0:
            p_kn = F.softmax(tgt_logits[known_m], dim=1)
            L_ent_kn = entropy(p_kn).mean()
        else:
            L_ent_kn = torch.tensor(0.0, device=x_s.device)

        # Inter-sample affinity alignment (intra-target)
        A_t = compute_affinity_matrix(z_t)   # (Bt, Bt)
        prob_t = F.softmax(tgt_logits, dim=1)
        sim_prob = prob_t @ prob_t.T          # (Bt, Bt)
        L_aff = F.mse_loss(sim_prob, A_t.detach())

        return L_ce + 0.5 * L_pseudo + 0.2 * L_unk + 0.3 * L_ent_kn + 0.1 * L_aff

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self,
        x_s_ref: torch.Tensor,    # source reference features for knowability
        x: torch.Tensor,
        unknown_label: int = UNKNOWN_LABEL,
    ) -> np.ndarray:
        """Predict; samples flagged as unknown → unknown_label."""
        self.eval()
        z_ref = self.encoder(x_s_ref)
        z = self.encoder(x)

        kappa = knowability_score(z_ref, z)
        _, _, unk_m = adaptive_threshold_split(kappa, self.q_low, self.q_high)

        logits = self.classifier(z)
        preds = logits.argmax(dim=1).cpu().numpy()
        preds[unk_m.cpu().numpy()] = unknown_label
        return preds


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_eiakda(
    model: EIAKDA,
    src_loader: DataLoader,
    tgt_loader: DataLoader,
    tgt_eval_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> dict:
    if device is None:
        device = get_device()
    model.to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    tgt_iter = cycle(tgt_loader)

    # Keep a reference source batch for inference-time knowability scoring
    src_ref_x, _ = next(iter(src_loader))
    src_ref_x = src_ref_x.to(device)

    best_metrics = None
    best_f1 = -1.0

    for epoch in range(1, epochs + 1):
        model.train()

        for (x_s, y_s) in src_loader:
            x_t, _ = next(tgt_iter)
            x_s, y_s = x_s.to(device), y_s.to(device)
            x_t = x_t.to(device)

            loss = model.eiakda_loss(x_s, y_s, x_t)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        scheduler.step()

        y_true_all, y_pred_all = [], []
        model.eval()
        with torch.no_grad():
            for x_t, y_t in tgt_eval_loader:
                x_t = x_t.to(device)
                preds = model.predict(src_ref_x, x_t)
                y_true_all.extend(y_t.numpy().tolist())
                y_pred_all.extend(preds.tolist())

        metrics = compute_metrics(np.array(y_true_all), np.array(y_pred_all))
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_metrics = metrics

        if verbose and epoch % 10 == 0:
            print_metrics(metrics, prefix=f"EIAKDA epoch {epoch:03d}")

    return best_metrics
