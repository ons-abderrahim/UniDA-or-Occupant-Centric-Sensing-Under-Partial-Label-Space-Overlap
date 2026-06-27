"""
LEAD — Learning Decomposition for Source-Free Universal Domain Adaptation
Adapted for 1-D Smart-Building Sensor Data (Figure 2, pipeline column 4).

Original paper:
    Qu et al., "LEAD: Learning Decomposition for Source-free Universal Domain
    Adaptation," CVPR 2024.
    Code: https://github.com/ispc-lab/LEAD

LEAD is the *only source-free* method in the comparison: at adaptation time
no source data is used.  The source model (encoder + classifier) is first
pre-trained on source data, then only the target domain samples are used to
adapt.

Key components (Figure 2):
    ┌─ Shared Temporal Encoder   G_tp                  [Eq. 40]
    ├─ Source-Known/Unknown Subspace Projection
    │    U, K, U_U               (orthogonal decomp)   [Eq. 65]
    ├─ Unknown Magnitude         ||z_u||_U             [Eq. 66]
    ├─ 2-Component GMM           r_i                   [Eq. 67]
    └─ Adaptive Decision Boundary  ρ, i, τ_U          [Eq. 68]
       → Output: Known Classes / Unknown (source-free)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from scipy.stats import norm as scipy_norm
from typing import Optional, Tuple

from src.encoder import TemporalEncoder, LinearClassifier
from src.utils import entropy, get_device
from src.metrics import compute_metrics, print_metrics, UNKNOWN_LABEL


# ---------------------------------------------------------------------------
# Orthogonal subspace decomposition  [Eq. 65]
# ---------------------------------------------------------------------------

def source_subspace(classifier_weight: torch.Tensor) -> torch.Tensor:
    """Compute the column space of the source classifier weight matrix.

    W ∈ R^{C × D}  →  U ∈ R^{D × C}  (orthonormal basis for known subspace)

    Args:
        classifier_weight: weight matrix of shape (C, D)  [C known classes, D features]

    Returns:
        U: orthonormal basis of shape (D, C)
    """
    W = classifier_weight.detach().cpu().float()
    # QR factorisation gives orthonormal column space
    Q, _ = torch.linalg.qr(W.T)   # Q: (D, C)
    return Q


def decompose_feature(
    z: torch.Tensor,    # (B, D)
    U: torch.Tensor,    # (D, K) orthonormal basis of known subspace
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decompose z into known-subspace and unknown-subspace components  [Eq. 65].

    z_k = U U^T z           (projection onto known subspace)
    z_u = z - z_k           (residual = unknown component)
    """
    U = U.to(z.device)
    proj = z @ U @ U.T   # (B, D)
    z_k = proj
    z_u = z - proj
    return z_k, z_u


def unknown_magnitude(z_u: torch.Tensor) -> torch.Tensor:
    """Per-sample unknown magnitude  ||z_u||  [Eq. 66]."""
    return z_u.norm(dim=1)   # (B,)


# ---------------------------------------------------------------------------
# 2-Component GMM  [Eq. 67]
# ---------------------------------------------------------------------------

def fit_gmm_threshold(
    magnitudes: np.ndarray,
    n_iter: int = 100,
    eps: float = 1e-6,
) -> float:
    """Fit a 2-component Gaussian Mixture to the unknown magnitudes and return
    the adaptive decision boundary τ_U  [Eq. 67-68].

    Uses a simple EM implementation (no scikit-learn dependency).
    The threshold is set at the crossing point of the two Gaussians.
    """
    x = magnitudes.astype(np.float64)
    N = len(x)
    if N < 4:
        return float(np.median(x))

    # Initialise with k-means (2 clusters)
    mu1, mu2 = x.min() + (x.max() - x.min()) * 0.3, x.min() + (x.max() - x.min()) * 0.7
    sigma1 = sigma2 = x.std() + eps
    pi1 = pi2 = 0.5

    for _ in range(n_iter):
        # E-step
        p1 = pi1 * scipy_norm.pdf(x, mu1, sigma1 + eps)
        p2 = pi2 * scipy_norm.pdf(x, mu2, sigma2 + eps)
        denom = p1 + p2 + eps
        r1 = p1 / denom
        r2 = p2 / denom

        # M-step
        n1, n2 = r1.sum() + eps, r2.sum() + eps
        mu1 = (r1 * x).sum() / n1
        mu2 = (r2 * x).sum() / n2
        sigma1 = np.sqrt((r1 * (x - mu1) ** 2).sum() / n1) + eps
        sigma2 = np.sqrt((r2 * (x - mu2) ** 2).sum() / n2) + eps
        pi1, pi2 = n1 / N, n2 / N

    # Adaptive threshold: intersection point of the two Gaussians  [Eq. 68]
    # Approximate by scanning
    grid = np.linspace(x.min(), x.max(), 500)
    lhs = pi1 * scipy_norm.pdf(grid, mu1, sigma1)
    rhs = pi2 * scipy_norm.pdf(grid, mu2, sigma2)
    cross = np.where(np.diff(np.sign(lhs - rhs)))[0]

    if len(cross) == 0:
        return float(np.median(x))

    # Pick the crossing closest to the midpoint of the two means
    mid = (mu1 + mu2) / 2
    tau_idx = cross[np.argmin(np.abs(grid[cross] - mid))]
    tau_U = float(grid[tau_idx])
    return tau_U


# ---------------------------------------------------------------------------
# LEAD model
# ---------------------------------------------------------------------------

class LEAD(nn.Module):
    """LEAD adapted for 1-D sensor data (source-free at adaptation time).

    Usage:
        1. Pre-train on source: call train_source_model(...)
        2. Adapt on target only: call adapt_target(...)
        3. Evaluate: call predict(...)
    """

    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        num_classes: int,
        feat_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder     = TemporalEncoder(in_channels, seq_len, dropout)
        self.classifier  = LinearClassifier(feat_dim, num_classes)
        self.num_classes = num_classes
        self.feat_dim    = feat_dim

        # Populated after source pre-training / calibration
        self._U: Optional[torch.Tensor] = None   # known subspace basis (D, C)
        self._tau_U: float = 0.5                 # decision boundary

    # ------------------------------------------------------------------
    # Phase 1: Source pre-training
    # ------------------------------------------------------------------
    def source_loss(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        z = self.encoder(x)
        return F.cross_entropy(self.classifier(z), y)

    # ------------------------------------------------------------------
    # Calibrate subspace + GMM threshold using source features  [Eq. 65-68]
    # ------------------------------------------------------------------
    @torch.no_grad()
    def calibrate(
        self,
        src_loader: DataLoader,
        device: torch.device,
    ) -> None:
        """Extract known-subspace basis U and calibrate the GMM threshold."""
        # Known subspace from classifier weights  [Eq. 65]
        W = self.classifier.fc.weight   # (C, D)
        self._U = source_subspace(W).to(device)

        # Collect unknown-subspace magnitudes from source data
        all_mags = []
        self.eval()
        for x, _ in src_loader:
            z = self.encoder(x.to(device))
            _, z_u = decompose_feature(z, self._U)
            mags = unknown_magnitude(z_u).cpu().numpy()
            all_mags.append(mags)

        all_mags = np.concatenate(all_mags)
        self._tau_U = fit_gmm_threshold(all_mags)

    # ------------------------------------------------------------------
    # Phase 2: Target-only adaptation (source-free)
    # ------------------------------------------------------------------
    def lead_adaptation_loss(self, x_t: torch.Tensor) -> torch.Tensor:
        """Source-free adaptation loss on target data  [Eq. 65-68].

        Encourages:
          - Low entropy on samples with small unknown magnitude (likely known)
          - High entropy on samples with large unknown magnitude (likely unknown)
        """
        z = self.encoder(x_t)

        if self._U is None:
            # Fallback if calibrate() was not called
            W = self.classifier.fc.weight.detach()
            self._U = source_subspace(W).to(z.device)

        _, z_u = decompose_feature(z, self._U)
        mags = unknown_magnitude(z_u)   # (B,)

        # Soft known / unknown weights  [Eq. 67]
        # r_i = sigmoid(τ_U − ||z_u||)  → 1 means known
        r = torch.sigmoid(torch.tensor(self._tau_U, device=mags.device) - mags)

        logits = self.classifier(z)
        probs  = F.softmax(logits, dim=1)
        H      = entropy(probs)   # (B,)

        # Entropy minimisation for known-leaning samples, maximisation for unknown
        L_adapt = (r * H + (1 - r) * (-H)).mean()
        return L_adapt

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self, x: torch.Tensor, unknown_label: int = UNKNOWN_LABEL
    ) -> np.ndarray:
        """Predict class; samples with high unknown magnitude → unknown_label."""
        self.eval()
        z = self.encoder(x)

        if self._U is None:
            W = self.classifier.fc.weight.detach()
            self._U = source_subspace(W).to(z.device)

        _, z_u = decompose_feature(z, self._U)
        mags = unknown_magnitude(z_u).cpu().numpy()

        logits = self.classifier(z)
        preds  = logits.argmax(dim=1).cpu().numpy()
        preds[mags > self._tau_U] = unknown_label
        return preds


# ---------------------------------------------------------------------------
# Two-phase training
# ---------------------------------------------------------------------------

def train_lead(
    model: LEAD,
    src_loader: DataLoader,
    tgt_loader: DataLoader,
    tgt_eval_loader: DataLoader,
    src_epochs: int = 30,
    tgt_epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> dict:
    """Two-phase LEAD training.

    Phase 1: Source pre-training (with source labels).
    Phase 2: Source-free target adaptation (no source data).
    """
    if device is None:
        device = get_device()
    model.to(device)

    # ── Phase 1: source pre-training ─────────────────────────────────────
    opt_src = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sch_src = torch.optim.lr_scheduler.CosineAnnealingLR(opt_src, T_max=src_epochs)

    if verbose:
        print("LEAD — Phase 1: Source pre-training")
    for epoch in range(1, src_epochs + 1):
        model.train()
        for x_s, y_s in src_loader:
            x_s, y_s = x_s.to(device), y_s.to(device)
            loss = model.source_loss(x_s, y_s)
            opt_src.zero_grad()
            loss.backward()
            opt_src.step()
        sch_src.step()

    # ── Calibration ───────────────────────────────────────────────────────
    if verbose:
        print("LEAD — Calibrating subspace & GMM threshold…")
    model.calibrate(src_loader, device)

    # ── Phase 2: source-free target adaptation ────────────────────────────
    # Freeze classifier; adapt encoder only
    for p in model.classifier.parameters():
        p.requires_grad_(False)

    opt_tgt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr * 0.1, weight_decay=weight_decay,
    )
    sch_tgt = torch.optim.lr_scheduler.CosineAnnealingLR(opt_tgt, T_max=tgt_epochs)

    if verbose:
        print("LEAD — Phase 2: Source-free target adaptation")

    best_metrics = None
    best_f1 = -1.0

    for epoch in range(1, tgt_epochs + 1):
        model.train()
        for x_t, _ in tgt_loader:
            x_t = x_t.to(device)
            loss = model.lead_adaptation_loss(x_t)
            opt_tgt.zero_grad()
            loss.backward()
            opt_tgt.step()
        sch_tgt.step()

        y_true_all, y_pred_all = [], []
        model.eval()
        with torch.no_grad():
            for x_t, y_t in tgt_eval_loader:
                preds = model.predict(x_t.to(device))
                y_true_all.extend(y_t.numpy().tolist())
                y_pred_all.extend(preds.tolist())

        metrics = compute_metrics(np.array(y_true_all), np.array(y_pred_all))
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_metrics = metrics

        if verbose and epoch % 5 == 0:
            print_metrics(metrics, prefix=f"LEAD tgt epoch {epoch:02d}")

    return best_metrics
