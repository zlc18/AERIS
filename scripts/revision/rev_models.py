"""Self-contained sequence models for the revision.

A single configurable network covers both the proposed AERIS architecture and
every ablation / sequence baseline the reviewers asked for, so that all of them
share the identical training loop, optimiser, schedule, checkpoint rule and
loss bookkeeping.  Only the switches listed in `ModelConfig` differ.

The temporal block, pooling, gate and variational head reproduce the original
implementation (scripts/run_exp51_enhanced_experiments.EnhancedTCNAVAE and
scripts/advanced_experiments.ResidualTCNBlock) exactly.
"""
from __future__ import annotations

import copy
import os
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# The experiments are run on CPU; use all available cores unless REV_TORCH_THREADS
# caps them (useful when several scripts run side by side).
torch.set_num_threads(max(1, int(os.environ.get("REV_TORCH_THREADS", 0)) or os.cpu_count() or 4))


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ResidualTCNBlock(nn.Module):
    """Causal-in-spirit dilated mixing block, identical to the submitted code."""

    def __init__(self, hidden_dim: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.dilation = dilation
        self.mix1 = nn.Linear(hidden_dim * 3, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mix2 = nn.Linear(hidden_dim * 3, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _dilated_mix(self, x: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
        pad = torch.zeros(x.size(0), self.dilation, x.size(2), device=x.device, dtype=x.dtype)
        left = torch.cat([pad, x[:, :-self.dilation, :]], dim=1)
        right = torch.cat([x[:, self.dilation:, :], pad], dim=1)
        return linear(torch.cat([left, x, right], dim=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(F.gelu(self.norm1(self._dilated_mix(x, self.mix1))))
        out = self.dropout(F.gelu(self.norm2(self._dilated_mix(out, self.mix2))))
        return out + x


class CausalTCNBlock(ResidualTCNBlock):
    """Strictly causal variant: the right (future) branch is zeroed out."""

    def _dilated_mix(self, x: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
        pad = torch.zeros(x.size(0), self.dilation, x.size(2), device=x.device, dtype=x.dtype)
        left = torch.cat([pad, x[:, :-self.dilation, :]], dim=1)
        right = torch.zeros_like(x)
        return linear(torch.cat([left, x, right], dim=-1))


@dataclass
class ModelConfig:
    encoder: str = "tcn"           # tcn | causal_tcn | lstm | gru | transformer
    pooling: str = "attention"     # attention | mean | last
    use_gate: bool = True
    bottleneck: str = "vae"        # vae | ae | none
    hidden_dim: int = 96
    latent_dim: int = 4
    attention_heads: int = 4
    dropout: float = 0.10
    dilations: tuple[int, ...] = (1, 2, 4, 1, 2)
    n_layers: int = 2              # for lstm / gru / transformer
    recon_target: str = "mean"     # mean (window average, as submitted) | window (full window)
    seq_len: int = 8               # only needed when recon_target == "window"
    # Skip connection from the terminal encoder state to the prediction heads.
    # The latent state then carries the compressed risk representation the density
    # layer needs, while the heads keep direct access to the most recent context
    # instead of having to squeeze the near-autoregressive signal through the
    # bottleneck.
    head_skip: bool = False
    # Explicit window-summary branch. The boosting baselines are fitted on the
    # per-feature mean, terminal value and maximum of the window; giving the
    # network the same view removes an input-representation advantage that has
    # nothing to do with the architecture being compared.
    summary_dim: int = 0           # 0 disables the branch


@dataclass
class TrainConfig:
    max_epochs: int = 12
    lr: float = 5e-4
    weight_decay: float = 2e-4
    batch_size: int = 128
    use_focal: bool = True
    ranking_weight: float = 0.03
    recon_weight: float = 0.15
    kl_weight: float = 0.01
    # Weight of the auxiliary head that regresses the continuous future composite
    # score. The binary label is a thresholded version of that score, so the
    # regression target carries strictly more information about the same quantity
    # and is available wherever the label is. Zero disables the head.
    aux_weight: float = 0.0
    # Loss on the auxiliary regression target. Screening cares about the ordering
    # of the upper tail, not about fitting the bulk, so a tail-weighted variant is
    # offered alongside the plain squared error.
    aux_loss: str = "mse"          # mse | huber | tail
    # Which score the validation checkpoint is selected on; it must match the
    # score the deployed model is ranked by.
    select_mode: str = "cls"
    seed: int = 42
    device: str = "cpu"
    history: list = field(default_factory=list)


class ScreeningNet(nn.Module):
    def __init__(self, input_dim: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_dim
        self.input_projection = nn.Linear(input_dim, h)

        if cfg.encoder in ("tcn", "causal_tcn"):
            block = ResidualTCNBlock if cfg.encoder == "tcn" else CausalTCNBlock
            self.encoder_net = nn.Sequential(*[block(h, d, cfg.dropout) for d in cfg.dilations])
        elif cfg.encoder in ("lstm", "gru"):
            rnn = nn.LSTM if cfg.encoder == "lstm" else nn.GRU
            self.encoder_net = rnn(h, h, num_layers=cfg.n_layers, batch_first=True,
                                   dropout=cfg.dropout if cfg.n_layers > 1 else 0.0)
        elif cfg.encoder == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=h, nhead=cfg.attention_heads, dim_feedforward=2 * h,
                dropout=cfg.dropout, batch_first=True, activation="gelu")
            self.encoder_net = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
            self.pos = nn.Parameter(torch.randn(1, 64, h) * 0.02)
        else:
            raise ValueError(cfg.encoder)

        if cfg.pooling == "attention":
            self.attn_pool = nn.MultiheadAttention(h, cfg.attention_heads,
                                                   dropout=cfg.dropout, batch_first=True)
            self.query = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        if cfg.use_gate:
            self.context_gate = nn.Sequential(
                nn.Linear(h * 2, h), nn.GELU(), nn.Linear(h, h), nn.Sigmoid())

        if cfg.summary_dim:
            self.summary_proj = nn.Sequential(
                nn.Linear(cfg.summary_dim, h), nn.GELU(), nn.LayerNorm(h),
                nn.Dropout(cfg.dropout), nn.Linear(h, h), nn.GELU())
        context_dim = h * 2 + (h if cfg.summary_dim else 0)
        self.encoder_head = nn.Sequential(nn.Linear(context_dim, h), nn.GELU(), nn.LayerNorm(h))
        latent = cfg.latent_dim if cfg.bottleneck != "none" else h
        if cfg.bottleneck == "vae":
            self.mu_layer = nn.Linear(h, latent)
            self.logvar_layer = nn.Linear(h, latent)
        elif cfg.bottleneck == "ae":
            self.mu_layer = nn.Linear(h, latent)
        if cfg.bottleneck != "none":
            out_dim = input_dim if cfg.recon_target == "mean" else input_dim * cfg.seq_len
            self.decoder = nn.Sequential(nn.Linear(latent, h), nn.GELU(), nn.Linear(h, out_dim))
        head_in = latent + (h if cfg.head_skip else 0) + (h if cfg.summary_dim else 0)
        self.regressor = nn.Sequential(nn.Linear(head_in, h), nn.GELU(),
                                       nn.Dropout(cfg.dropout), nn.Linear(h, 1))
        self.aux_head = nn.Sequential(nn.Linear(head_in, max(h // 2, 16)), nn.GELU(),
                                      nn.Linear(max(h // 2, 16), 1))

    def encode_sequence(self, projected: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        if cfg.encoder in ("tcn", "causal_tcn"):
            return self.encoder_net(projected)
        if cfg.encoder in ("lstm", "gru"):
            out, _ = self.encoder_net(projected)
            return out
        seq = projected + self.pos[:, : projected.size(1), :]
        mask = torch.triu(torch.ones(projected.size(1), projected.size(1),
                                     device=projected.device, dtype=torch.bool), diagonal=1)
        return self.encoder_net(seq, mask=mask)

    def forward(self, inputs: torch.Tensor, summary: torch.Tensor | None = None):
        cfg = self.cfg
        projected = self.input_projection(inputs)
        encoded = self.encode_sequence(projected)
        last_ctx = encoded[:, -1, :]
        if cfg.pooling == "attention":
            q = self.query.expand(inputs.size(0), -1, -1)
            pooled, _ = self.attn_pool(q, encoded, encoded)
            pooled = pooled.squeeze(1)
        elif cfg.pooling == "mean":
            pooled = encoded.mean(dim=1)
        else:
            pooled = last_ctx

        if cfg.use_gate:
            gate = self.context_gate(torch.cat([pooled, last_ctx], dim=-1))
            context = torch.cat([gate * pooled, (1.0 - gate) * last_ctx], dim=-1)
        else:
            context = torch.cat([0.5 * pooled, 0.5 * last_ctx], dim=-1)

        summary_emb = None
        if cfg.summary_dim:
            if summary is None:
                raise ValueError("the summary branch is enabled but no summary was given")
            summary_emb = self.summary_proj(summary)
            context = torch.cat([context, summary_emb], dim=-1)

        enc = self.encoder_head(context)
        if cfg.bottleneck == "vae":
            mu = self.mu_layer(enc)
            logvar = self.logvar_layer(enc)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        elif cfg.bottleneck == "ae":
            mu = self.mu_layer(enc)
            logvar = torch.zeros_like(mu)
            z = mu
        else:
            mu = enc
            logvar = torch.zeros_like(enc)
            z = enc
        recon = self.decoder(z) if cfg.bottleneck != "none" else None
        target = (inputs.mean(dim=1) if cfg.recon_target == "mean"
                  else inputs.reshape(inputs.size(0), -1))
        parts = [z]
        if cfg.head_skip:
            parts.append(last_ctx)
        if summary_emb is not None:
            parts.append(summary_emb)
        head_in = torch.cat(parts, dim=-1) if len(parts) > 1 else z
        logits = self.regressor(head_in).squeeze(-1)
        aux = self.aux_head(head_in).squeeze(-1)
        return logits, recon, target, mu, logvar, z, aux


def pairwise_ranking_loss(logits: torch.Tensor, labels: torch.Tensor, max_pairs: int = 64):
    pos = logits[labels > 0.5]
    neg = logits[labels <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return logits.new_tensor(0.0)
    pos = torch.sort(pos.flatten(), descending=False).values[:max_pairs]
    neg = torch.sort(neg.flatten(), descending=True).values[:max_pairs]
    return F.softplus(neg.unsqueeze(0) - pos.unsqueeze(1)).mean()


def auxiliary_loss(prediction: torch.Tensor, target: torch.Tensor, kind: str) -> torch.Tensor:
    """Loss on the future composite score.

    'tail' weights each sample by how far its target sits above the bulk, which
    concentrates the fit on the upper tail the screening task actually ranks.
    """
    if kind == "mse":
        return F.mse_loss(prediction, target)
    if kind == "huber":
        return F.smooth_l1_loss(prediction, target)
    if kind == "tail":
        weight = 1.0 + F.softplus(target)
        return (weight * (prediction - target) ** 2).mean() / weight.mean().clamp_min(1e-6)
    raise ValueError(kind)


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values))
    return ranks / max(len(values) - 1, 1)


def combine_scores(prob: np.ndarray, aux: np.ndarray, mode: str) -> np.ndarray:
    """Turn the two heads into one ranking score.

    'cls'        rank by the screening probability (the original behaviour);
    'aux'        rank by the predicted future composite score;
    'rank_blend' average the two percentile ranks, which needs no rescaling
                 between a probability and a standardised score.
    """
    if mode == "cls":
        return prob
    if mode == "aux":
        return aux
    if mode == "rank_blend":
        return 0.5 * (_percentile_rank(prob) + _percentile_rank(aux))
    raise ValueError(mode)


def predict(model: nn.Module, x: np.ndarray, device: str, batch_size: int = 512,
            return_latent: bool = False, mode: str = "cls",
            summary: np.ndarray | None = None):
    tensors = [torch.tensor(x, dtype=torch.float32)]
    if summary is not None:
        tensors.append(torch.tensor(summary, dtype=torch.float32))
    loader = DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=False)
    probs, auxes, latents = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            bx = batch[0].to(device)
            bs = batch[1].to(device) if len(batch) > 1 else None
            logits, _, _, mu, _, _, aux = model(bx, bs)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            auxes.append(aux.cpu().numpy())
            if return_latent:
                latents.append(mu.cpu().numpy())
    p = combine_scores(np.concatenate(probs), np.concatenate(auxes), mode)
    return (p, np.concatenate(latents)) if return_latent else p


def train(model: nn.Module, tr: np.ndarray, y_tr: np.ndarray, va: np.ndarray, y_va: np.ndarray,
          tcfg: TrainConfig, aux_tr: np.ndarray | None = None,
          summary_tr: np.ndarray | None = None,
          summary_va: np.ndarray | None = None) -> tuple[nn.Module, dict]:
    set_seed(tcfg.seed)
    device = tcfg.device
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tcfg.max_epochs, eta_min=tcfg.lr / 10)
    pos_ratio = float(np.clip(y_tr.mean(), 1e-4, 1 - 1e-4))
    pw = torch.tensor((1 - pos_ratio) / pos_ratio, dtype=torch.float32, device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)
    mse = nn.MSELoss()

    use_aux = tcfg.aux_weight > 0 and aux_tr is not None
    aux_array = (np.asarray(aux_tr, dtype=np.float32) if use_aux
                 else np.zeros(len(y_tr), dtype=np.float32))
    summary_array = (np.asarray(summary_tr, dtype=np.float32) if summary_tr is not None
                     else np.zeros((len(y_tr), 1), dtype=np.float32))
    loader = DataLoader(
        TensorDataset(torch.tensor(tr, dtype=torch.float32),
                      torch.tensor(y_tr, dtype=torch.float32),
                      torch.tensor(aux_array),
                      torch.tensor(summary_array)),
        batch_size=tcfg.batch_size, shuffle=True)

    best_val, best_state, history = -1.0, None, []
    t0 = time.perf_counter()
    for epoch in range(tcfg.max_epochs):
        model.train()
        for bx, by, ba, bs in loader:
            bx, by, ba = bx.to(device), by.to(device), ba.to(device)
            bs = bs.to(device) if summary_tr is not None else None
            opt.zero_grad()
            logits, recon, tgt, mu, logvar, _, aux = model(bx, bs)
            if tcfg.use_focal:
                p = torch.sigmoid(logits)
                focal_w = (1 - torch.where(by == 1, p, 1 - p)) ** 2
                focal = (focal_w * F.binary_cross_entropy_with_logits(
                    logits, by, reduction="none")).mean()
                cls = 0.5 * bce(logits, by) + 0.5 * focal
            else:
                cls = bce(logits, by)
            loss = cls
            if tcfg.ranking_weight > 0:
                loss = loss + tcfg.ranking_weight * pairwise_ranking_loss(logits, by)
            if recon is not None and tcfg.recon_weight > 0:
                loss = loss + tcfg.recon_weight * mse(recon, tgt)
            if model.cfg.bottleneck == "vae" and tcfg.kl_weight > 0:
                loss = loss + tcfg.kl_weight * (-0.5 * torch.mean(
                    1 + logvar - mu.pow(2) - logvar.exp()))
            if use_aux:
                loss = loss + tcfg.aux_weight * auxiliary_loss(aux, ba, tcfg.aux_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        vs = predict(model, va, device, mode=tcfg.select_mode, summary=summary_va)
        vpr = float(average_precision_score(y_va.astype(int), vs))
        history.append({"epoch": epoch + 1, "valid_pr_auc": vpr})
        if vpr > best_val:
            best_val, best_state = vpr, copy.deepcopy(model.state_dict())
    train_seconds = time.perf_counter() - t0
    model.load_state_dict(best_state)
    model.eval()
    info = {
        "best_valid_pr_auc": best_val,
        "train_seconds": train_seconds,
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "history": history,
    }
    return model, info


def inference_latency_ms(model: nn.Module, x: np.ndarray, device: str, repeats: int = 3,
                         summary: np.ndarray | None = None) -> float:
    model.eval()
    xb = torch.tensor(x[:2048], dtype=torch.float32, device=device)
    sb = (torch.tensor(summary[:2048], dtype=torch.float32, device=device)
          if summary is not None else None)
    with torch.no_grad():
        model(xb[:8], sb[:8] if sb is not None else None)  # warm-up
        t0 = time.perf_counter()
        for _ in range(repeats):
            model(xb, sb)
        dt = (time.perf_counter() - t0) / repeats
    return 1000 * dt / len(xb)
