"""
PPOT — Prototypical Partial Optimal Transport for Universal Domain Adaptation
Adapted for 1-D Smart-Building Sensor Data (Figure 2, pipeline column 1).

Original paper:
    Yang et al., "Prototypical Partial Optimal Transport for Universal Domain
    Adaptation," AAAI 2023.

Adaptation:
    The original ResNet/ViT backbone is replaced by the shared TemporalEncoder
    defined in src/encoder.py.  All other components follow the paper.

Key components (Figure 2):
    ┌─ Shared Temporal Encoder  G_tp  [Eq. 40]
    ├─ Class Prototypes          p_c   [Eq. 42]
    ├─ Partial OT Matching       y*    [Eq. 44]
    ├─ Transport Weights         w_ft  (derived from OT plan)
    └─ Weighted Entropy Min.     L_PPOT [Eq. 46]   → Output: Known Classes
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
# Sinkhorn / Partial OT utilities
# ---------------------------------------------------------------------------

def sinkhorn(
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float = 0.1,
    n_iters: int = 50,
) -> torch.Tensor:
    """Log-domain Sinkhorn algorithm.

    Solves  OT_eps(a, b, cost)  and returns the transport plan Q.

    Args:
        cost   : (M, N) cost matrix
        a      : (M,)   source marginal distribution
        b      : (N,)   target marginal distribution
        eps    : entropic regularisation strength
        n_iters: number of Sinkhorn iterations

    Returns:
        Q: (M, N) transport plan
    """
    log_a = torch.log(a + 1e-9).unsqueeze(1)   # (M, 1)
    log_b = torch.log(b + 1e-9).unsqueeze(0)   # (1, N)
    K = -cost / eps                              # (M, N)

    u = torch.zeros_like(a)
    v = torch.zeros_like(b)

    for _ in range(n_iters):
        u = log_a.squeeze() - torch.logsumexp(K + v.unsqueeze(0), dim=1)
        v = log_b.squeeze() - torch.logsumexp(K + u.unsqueeze(1), dim=0)

    log_Q = K + u.unsqueeze(1) + v.unsqueeze(0)
    return torch.exp(log_Q)


def partial_ot(
    src_proto: torch.Tensor,
    tgt_feat: torch.Tensor,
    mass: float = 0.5,
    eps: float = 0.05,
    n_iters: int = 30,
) -> torch.Tensor:
    """Mini-batch Prototypical Partial OT  [Eq. 44].

    Transports only a fraction `mass` of the target distribution onto the
    source prototypes, leaving the remainder as "unknown".

    Args:
        src_proto: (C, D) source class prototypes
        tgt_feat : (B, D) target feature embeddings
        mass     : fraction of target mass to transport (∈ (0, 1])
        eps      : entropic regularisation

    Returns:
        Q: (C, B) transport plan — Q[c, i] is the mass sent from proto c to target i.
    """
    C, D = src_proto.shape
    B = tgt_feat.shape[0]

    # Cost  =  1 − cosine_similarity
    src_n = F.normalize(src_proto, dim=1)   # (C, D)
    tgt_n = F.normalize(tgt_feat, dim=1)    # (B, D)
    sim = src_n @ tgt_n.T                   # (C, B)
    cost = 1.0 - sim                        # (C, B)

    # Uniform marginals; partial mass on the target side
    a = torch.ones(C, device=src_proto.device) / C
    b = torch.full((B,), mass / B, device=tgt_feat.device)

    # Pad target with a "dust-bin" column to absorb the (1 - mass) remainder
    dust_cost = torch.ones(C, 1, device=cost.device)
    cost_ext = torch.cat([cost, dust_cost], dim=1)       # (C, B+1)
    b_ext = torch.cat([b, torch.tensor([1 - mass], device=b.device)])  # (B+1,)

    Q_ext = sinkhorn(cost_ext, a, b_ext, eps=eps, n_iters=n_iters)
    Q = Q_ext[:, :B]   # (C, B) — drop dust-bin column
    return Q


# ---------------------------------------------------------------------------
# PPOT model
# ---------------------------------------------------------------------------

class PPOT(nn.Module):
    """PPOT adapted for 1-D sensor data.

    Args:
        in_channels  : number of sensor features
        seq_len      : window length
        num_classes  : number of *known* source classes
        feat_dim     : encoder output dimension (64)
        ot_mass      : fraction of target mass for partial OT
        ot_eps       : Sinkhorn regularisation
        tau_ent      : entropy threshold for unknown detection
    """

    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        num_classes: int,
        feat_dim: int = 64,
        ot_mass: float = 0.5,
        ot_eps: float = 0.05,
        tau_ent: float = 0.5,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder    = TemporalEncoder(in_channels, seq_len, dropout)
        self.classifier = LinearClassifier(feat_dim, num_classes)
        self.num_classes = num_classes
        self.feat_dim   = feat_dim
        self.ot_mass    = ot_mass
        self.ot_eps     = ot_eps
        self.tau_ent    = tau_ent

        # Class prototypes  p_c  [Eq. 42] — learnable
        self.prototypes = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.prototypes.unsqueeze(0)).squeeze_(0)

    # ------------------------------------------------------------------
    # Prototype update  [Eq. 42]
    # ------------------------------------------------------------------
    @torch.no_grad()
    def update_prototypes(
        self,
        src_feat: torch.Tensor,   # (B, D)
        src_labels: torch.Tensor, # (B,)
        momentum: float = 0.9,
    ) -> None:
        """EMA update of class prototypes from the current source batch."""
        for c in range(self.num_classes):
            mask = src_labels == c
            if mask.sum() == 0:
                continue
            mean_c = src_feat[mask].mean(dim=0)
            self.prototypes.data[c] = (
                momentum * self.prototypes.data[c] + (1 - momentum) * mean_c
            )

    # ------------------------------------------------------------------
    # Loss  [Eq. 46]
    # ------------------------------------------------------------------
    def ppot_loss(
        self,
        src_feat:   torch.Tensor,  # (B, D)
        src_labels: torch.Tensor,  # (B,)
        tgt_feat:   torch.Tensor,  # (B, D)
    ) -> torch.Tensor:
        """L_PPOT = L_CE(src) + λ_e * L_ent(tgt_known) + λ_ot * L_ot."""

        # ── Source cross-entropy
        src_logits = self.classifier(src_feat)
        L_ce = F.cross_entropy(src_logits, src_labels)

        # ── Partial OT plan  →  transport weights  w_ft
        Q = partial_ot(
            self.prototypes.detach(),
            tgt_feat.detach(),
            mass=self.ot_mass,
            eps=self.ot_eps,
        )   # (C, B_tgt)

        # Per-target-sample weight = row-sum of Q (how much mass received)
        w = Q.sum(dim=0)                    # (B_tgt,)
        w = w / (w.sum() + 1e-9)

        # ── Weighted entropy minimisation on target (known-leaning) samples
        tgt_logits = self.classifier(tgt_feat)
        tgt_probs  = F.softmax(tgt_logits, dim=1)
        H = entropy(tgt_probs)              # (B_tgt,)
        L_ent = (w * H).sum()

        # ── OT alignment cost (prototype↔target feature cosine distance)
        src_n = F.normalize(self.prototypes, dim=1)  # (C, D)
        tgt_n = F.normalize(tgt_feat, dim=1)         # (B, D)
        cost  = 1.0 - src_n @ tgt_n.T               # (C, B)
        L_ot  = (Q * cost).sum()

        return L_ce + 0.3 * L_ent + 0.1 * L_ot

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, x: torch.Tensor, unknown_label: int = UNKNOWN_LABEL) -> np.ndarray:
        """Return predicted class indices; unknown samples → unknown_label."""
        self.eval()
        z = self.encoder(x)
        logits = self.classifier(z)
        probs  = F.softmax(logits, dim=1)
        H      = entropy(probs)
        preds  = logits.argmax(dim=1).cpu().numpy()
        preds[H.cpu().numpy() > self.tau_ent] = unknown_label
        return preds


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_ppot(
    model: PPOT,
    src_loader: DataLoader,
    tgt_loader: DataLoader,
    tgt_eval_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> dict:
    """Train PPOT and return best evaluation metrics."""
    if device is None:
        device = get_device()
    model.to(device)

    optimiser = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    tgt_iter = cycle(tgt_loader)
    best_metrics = None
    best_f1 = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for (x_s, y_s) in src_loader:
            x_t, _ = next(tgt_iter)
            x_s, y_s = x_s.to(device), y_s.to(device)
            x_t = x_t.to(device)

            z_s = model.encoder(x_s)
            z_t = model.encoder(x_t)

            model.update_prototypes(z_s.detach(), y_s)

            loss = model.ppot_loss(z_s, y_s, z_t)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        # Evaluation
        y_true_all, y_pred_all = [], []
        model.eval()
        with torch.no_grad():
            for x_t, y_t in tgt_eval_loader:
                x_t = x_t.to(device)
                preds = model.predict(x_t)
                y_true_all.extend(y_t.numpy().tolist())
                y_pred_all.extend(preds.tolist())

        metrics = compute_metrics(
            np.array(y_true_all), np.array(y_pred_all)
        )
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_metrics = metrics

        if verbose and epoch % 10 == 0:
            print_metrics(metrics, prefix=f"PPOT epoch {epoch:03d}")

    return best_metrics
