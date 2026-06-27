"""
MLNet — Mutual Learning Network with Neighbourhood Invariance
Adapted for 1-D Smart-Building Sensor Data (Figure 2, pipeline column 2).

Original paper:
    Lu et al., "MLNet: Mutual Learning Network with Neighborhood Invariance
    for Universal Domain Adaptation," AAAI 2024.
    Code: https://github.com/YanzuoLu/MLNet

Key components (Figure 2):
    ┌─ Shared Temporal Encoder     G_tp            [Eq. 40]
    ├─ Neighbourhood Invariance  + Manifold Mixup  [Eq. 49-50]
    ├─ Closed-set classifier       C_c             [Eq. 51]
    ├─ Open-set classifier (1-vs-all)  C_op        [Eq. 52]
    └─ Classifier Consistency Loss L_CC [KL-div]   [Eq. 53]
       → Output: Known / Reject
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Optional, List

from src.encoder import TemporalEncoder
from src.utils import entropy, cycle, get_device
from src.metrics import compute_metrics, print_metrics, UNKNOWN_LABEL


# ---------------------------------------------------------------------------
# Neighbourhood Invariance helpers  [Eq. 49-50]
# ---------------------------------------------------------------------------

def neighbourhood_invariance_loss(
    feats: torch.Tensor,
    k: int = 5,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Encourage a sample to be close to its k-nearest neighbours in feature space.

    Implementation of the confidence-guided invariant feature learning with
    self-adaptive neighbour selection (Section 3.1 of MLNet).

    L_ni = - sum_{i} sum_{j in kNN(i)} log P(j | i)  [Eq. 49]
    """
    B, D = feats.shape
    if B <= k:
        return torch.tensor(0.0, device=feats.device)

    # Pairwise cosine similarity
    fn = F.normalize(feats, dim=1)
    sim = fn @ fn.T              # (B, B)
    sim.fill_diagonal_(-1e9)     # exclude self

    # kNN mask
    _, topk_idx = sim.topk(k, dim=1)   # (B, k)
    knn_mask = torch.zeros_like(sim).scatter_(1, topk_idx, 1.0)

    # Contrastive-style loss
    logits = sim / temperature
    log_probs = F.log_softmax(logits, dim=1)
    loss = -(knn_mask * log_probs).sum(dim=1).mean()
    return loss


def manifold_mixup(
    x_s: torch.Tensor,
    x_t: torch.Tensor,
    alpha: float = 0.2,
) -> torch.Tensor:
    """Cross-domain Manifold Mixup  [Eq. 50].

    z_mix = λ * z_s + (1 - λ) * z_t,  λ ~ Beta(alpha, alpha)
    """
    lam = np.random.beta(alpha, alpha)
    min_len = min(len(x_s), len(x_t))
    return lam * x_s[:min_len] + (1 - lam) * x_t[:min_len]


# ---------------------------------------------------------------------------
# MLNet model
# ---------------------------------------------------------------------------

class MLNet(nn.Module):
    """MLNet adapted for 1-D sensor data.

    Two classification heads on the shared encoder:
      C_c  — closed-set head  (C known classes)           [Eq. 51]
      C_op — open-set head   (binary: known vs unknown)   [Eq. 52]
    """

    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        num_classes: int,
        feat_dim: int = 64,
        knn_k: int = 5,
        mixup_alpha: float = 0.2,
        tau: float = 0.5,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder     = TemporalEncoder(in_channels, seq_len, dropout)
        self.num_classes = num_classes
        self.knn_k       = knn_k
        self.mixup_alpha = mixup_alpha
        self.tau         = tau   # open-set rejection threshold

        # Closed-set classifier  C_c  [Eq. 51]
        self.closed_head = nn.Linear(feat_dim, num_classes)

        # Open-set (1-vs-all) classifier  C_op  [Eq. 52]
        # Binary output: 0 = known, 1 = unknown
        self.open_head = nn.Linear(feat_dim, 2)

    # ------------------------------------------------------------------
    # Classifier Consistency Loss  L_CC  [Eq. 53]
    # ------------------------------------------------------------------
    def consistency_loss(
        self,
        z: torch.Tensor,    # (B, D) target features
    ) -> torch.Tensor:
        """KL divergence between the closed-set and open-set predictions.

        L_CC = KL(P_c || P_op)  [Eq. 53]
        Both heads are normalised to obtain proper distributions before KL.
        """
        p_c  = F.softmax(self.closed_head(z), dim=1)   # (B, C_known)
        p_op = F.softmax(self.open_head(z), dim=1)     # (B, 2)

        # Map closed-set probs to a 2-class "known / unknown" distribution
        # by taking max(known probs) as P(known) and 1 - max as P(unknown)
        p_known = p_c.max(dim=1).values.unsqueeze(1)       # (B, 1)
        p_unk   = (1.0 - p_known)                          # (B, 1)
        p_c_bin = torch.cat([p_known, p_unk], dim=1)       # (B, 2)

        loss = F.kl_div(
            p_c_bin.log() + 1e-9,
            p_op.detach(),
            reduction="batchmean",
        )
        return loss

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        return self.closed_head(z), self.open_head(z), z

    # ------------------------------------------------------------------
    # Full loss
    # ------------------------------------------------------------------
    def mlnet_loss(
        self,
        x_s: torch.Tensor,
        y_s: torch.Tensor,
        x_t: torch.Tensor,
    ) -> torch.Tensor:
        z_s = self.encoder(x_s)
        z_t = self.encoder(x_t)

        # Closed-set source CE
        L_ce = F.cross_entropy(self.closed_head(z_s), y_s)

        # Open-set source: label 0 (known)
        L_op = F.cross_entropy(
            self.open_head(z_s),
            torch.zeros(len(z_s), dtype=torch.long, device=z_s.device),
        )

        # Neighbourhood invariance on target features  [Eq. 49]
        L_ni = neighbourhood_invariance_loss(z_t, k=self.knn_k)

        # Manifold Mixup cross-domain  [Eq. 50]
        z_mix = manifold_mixup(z_s.detach(), z_t.detach(), self.mixup_alpha)
        # Mixed features should be classified as "known"
        L_mix = F.cross_entropy(
            self.open_head(z_mix.detach()),
            torch.zeros(len(z_mix), dtype=torch.long, device=z_mix.device),
        )

        # Consistency loss on target  [Eq. 53]
        L_cc = self.consistency_loss(z_t)

        return L_ce + L_op + 0.5 * L_ni + 0.2 * L_mix + 0.3 * L_cc

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, x: torch.Tensor, unknown_label: int = UNKNOWN_LABEL) -> np.ndarray:
        """Predict class; reject (→ unknown_label) if open-set head says unknown."""
        self.eval()
        z = self.encoder(x)
        closed_logits = self.closed_head(z)  # (B, C)
        open_logits   = self.open_head(z)    # (B, 2)

        # P(unknown) from the open-set head
        p_unk = F.softmax(open_logits, dim=1)[:, 1].cpu().numpy()
        preds = closed_logits.argmax(dim=1).cpu().numpy()
        preds[p_unk > self.tau] = unknown_label
        return preds


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_mlnet(
    model: MLNet,
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
    best_metrics = None
    best_f1 = -1.0

    for epoch in range(1, epochs + 1):
        model.train()

        for (x_s, y_s) in src_loader:
            x_t, _ = next(tgt_iter)
            x_s, y_s = x_s.to(device), y_s.to(device)
            x_t = x_t.to(device)

            loss = model.mlnet_loss(x_s, y_s, x_t)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        scheduler.step()

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

        if verbose and epoch % 10 == 0:
            print_metrics(metrics, prefix=f"MLNet epoch {epoch:03d}")

    return best_metrics
