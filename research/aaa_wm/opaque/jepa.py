"""J-2/J-4: an action-conditioned JEPA objective for the opaque.v0 world model.

    context encoder  z_t      = E_theta(program_t)                   ([CLS] representation)
    edit embedding   a        = A_theta(edit site context, old -> new tokens)
    predictor        z_hat    = P_theta(z_t, a)                      (MLP)
    target encoder   z_{t+1}  = E_xi(program_{t+1}),  xi <- EMA(theta)   (stop-gradient)
    JEPA loss        || normalize(z_hat) - normalize(sg(z_{t+1})) ||^2
    anti-collapse    VICReg variance hinge (std >= 1) + covariance penalty on z_hat and z_t
    consequence      the ``wm`` head on the context encoder's query slots (J-4); weight ``lam_c``

The transition in text space is exact (applying an edit is string substitution). The
JEPA term therefore does not replace the transition. It is an auxiliary objective that asks the
representation to *predict how an edit changes the program's representation*. The hypothesis
under test is that this improves the consequence head's generalization. The planner still
re-encodes each candidate program exactly. A variant that plans purely in latent space
(``z_hat`` -> consequence head) is evaluated as J-2.

EMA target parameters are part of the checkpoint and of the adaptive-state count.
Collapse diagnostics: per-dimension std, effective rank (exp of the entropy of normalized
singular values), mean pairwise cosine, and action sensitivity
(``||P(z, a) - P(z, a')||`` for different edits of the same program).
"""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import torch
from torch import nn

from . import tokens as tok
from .data import Dataset
from .models import QUERY_SLOTS, Arm, ArmConfig, Batcher, _tail_queries, assemble


class JepaArm(nn.Module):
    def __init__(self, c: ArmConfig, *, lam_c: float = 1.0, lam_j: float = 1.0, ema: float = 0.996) -> None:
        super().__init__()
        self.inner = Arm(ArmConfig("wm", c.d_model, c.layers, c.heads, c.ff, c.dropout, c.seed))
        d = c.d_model
        self.edit = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, d))
        self.pred = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.target = copy.deepcopy(self.inner.backbone)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.lam_c, self.lam_j, self.ema = lam_c, lam_j, ema
        self.config = c

    @property
    def consequence(self) -> nn.Module:
        return self.inner.consequence

    def hidden(self, ids: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
        return self.inner.hidden(ids, types)

    @torch.no_grad()
    def update_target(self) -> None:
        for pt, po in zip(self.target.parameters(), self.inner.backbone.parameters(), strict=True):
            pt.mul_(self.ema).add_(po.detach(), alpha=1 - self.ema)

    def edit_embedding(self, h_before: torch.Tensor, site: torch.Tensor, old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        emb = self.inner.backbone.tok
        ctx = h_before[torch.arange(h_before.shape[0]), site]
        return self.edit(torch.cat([ctx, emb(old), emb(new)], dim=-1))


def _vicreg(z: torch.Tensor) -> torch.Tensor:
    z = z - z.mean(0)
    std = torch.sqrt(z.var(0) + 1e-4)
    var_loss = torch.relu(1 - std).mean()
    cov = (z.T @ z) / (z.shape[0] - 1)
    off = cov - torch.diag(torch.diag(cov))
    return var_loss + (off**2).sum() / z.shape[1]


def collapse_diagnostics(z: torch.Tensor) -> dict[str, float]:
    z = z.float()
    zc = z - z.mean(0)
    s = torch.linalg.svdvals(zc)
    p = s / s.sum()
    erank = float(torch.exp(-(p * torch.log(p + 1e-12)).sum()))
    zn = nn.functional.normalize(z, dim=-1)
    cos = (zn @ zn.T)
    n = z.shape[0]
    mean_cos = float((cos.sum() - n) / (n * (n - 1)))
    return {"mean_std": float(zc.std(0).mean()), "min_std": float(zc.std(0).min()), "effective_rank": erank, "mean_pairwise_cosine": mean_cos}


class TransitionBatcher:
    """(program before, edit, program after) triples from the dataset's policy records."""

    def __init__(self, ds: Dataset, seed: int) -> None:
        self.ds = ds
        self.rng = np.random.default_rng(seed)
        # before-state of each policy record: the program at the same task whose sources differ at one token
        self.records = ds.policy

    def sample(self, B: int, sources_before: dict[int, str]) -> Any:
        raise NotImplementedError


def build_transitions(ds: Dataset, tasks: list[Any], limit: int | None = None) -> dict[str, np.ndarray]:
    """Explicit transition records (before-state code, site index, old id, new id, after-state code)."""

    from .data import _code_row
    from .program import apply, edits

    before, after, site, old, new = [], [], [], [], []
    names = tuple(f"api{k}" for k in range(6))
    for t in tasks[: limit or len(tasks)]:
        b_row, b_pos = _code_row(t.buggy)
        for e in edits(t.buggy, names):
            a_src = apply(t.buggy, e)
            k = next((i for i, w in enumerate(b_pos) if w == (e.row, e.col)), None)
            if k is None:
                continue
            before.append(b_row)
            after.append(_code_row(a_src)[0])
            site.append(k + 1)
            old.append(tok.INDEX.get(e.old, tok.number(int(e.old)) if e.old.isdigit() else tok.INDEX["<UNK>"]))
            new.append(tok.INDEX.get(e.new, tok.number(int(e.new)) if e.new.isdigit() else tok.INDEX["<UNK>"]))
    return {
        "before": np.stack(before),
        "after": np.stack(after),
        "site": np.array(site),
        "old": np.array(old),
        "new": np.array(new),
    }


def train_jepa(
    model: JepaArm, ds: Dataset, trans: dict[str, np.ndarray], *, steps: int, batch: int = 256, lr: float = 1e-3, device: str = "cuda", amp: bool = True, vic: float = 1.0
) -> dict[str, Any]:
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    wm_batches = Batcher(ds, "wm", model.config.seed + 17)
    rng = np.random.default_rng(model.config.seed + 29)
    curve = []
    empty_tail = np.zeros((batch, 9), dtype=np.int64)
    model.train()
    for step in range(steps):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            ids, types, target = wm_batches.wm(batch)
            h = model.hidden(ids.to(device), types.to(device))
            logits = model.consequence(h[:, QUERY_SLOTS]).float()
            loss_c = nn.functional.cross_entropy(logits.reshape(-1, tok.N_RESULTS), target.to(device).reshape(-1))
            j = rng.integers(0, len(trans["site"]), batch)
            bi, bt = assemble(trans["before"][j], empty_tail, empty_tail)
            hb = model.hidden(bi.to(device), bt.to(device))
            z = hb[:, 0]
            site = torch.from_numpy(np.minimum(trans["site"][j], hb.shape[1] - 1)).to(device)
            a = model.edit_embedding(hb, site, torch.from_numpy(trans["old"][j]).to(device), torch.from_numpy(trans["new"][j]).to(device))
            z_hat = model.pred(torch.cat([z, a], dim=-1)).float()
            with torch.no_grad():
                ai, at = assemble(trans["after"][j], empty_tail, empty_tail)
                z_next = model.target(ai.to(device), at.to(device))[:, 0].float()
            loss_j = ((nn.functional.normalize(z_hat, dim=-1) - nn.functional.normalize(z_next, dim=-1)) ** 2).sum(-1).mean()
            loss_v = _vicreg(z_hat) + _vicreg(z.float()) if vic > 0 else torch.zeros((), device=device)
            loss = model.lam_c * loss_c + model.lam_j * (loss_j + vic * loss_v)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        model.update_target()
        if not math.isfinite(loss.item()):
            raise FloatingPointError("non-finite loss")
        if step % 500 == 0 or step == steps - 1:
            curve.append((step, float(loss_c.item()), float(loss_j.item()), float(loss_v.item())))
    model.eval()
    with torch.no_grad():
        j = rng.integers(0, len(trans["site"]), 1024)
        bi, bt = assemble(trans["before"][j], np.zeros((1024, 9), dtype=np.int64), np.zeros((1024, 9), dtype=np.int64))
        hb = model.hidden(bi.to(device), bt.to(device))
        a = model.edit_embedding(hb, torch.from_numpy(np.minimum(trans["site"][j], hb.shape[1] - 1)).to(device), torch.from_numpy(trans["old"][j]).to(device), torch.from_numpy(trans["new"][j]).to(device))
        z_hat = model.pred(torch.cat([hb[:, 0], a], dim=-1))
        a_shuf = a[torch.randperm(a.shape[0])]
        z_shuf = model.pred(torch.cat([hb[:, 0], a_shuf], dim=-1))
        diag = {
            "z": collapse_diagnostics(hb[:, 0]),
            "z_hat": collapse_diagnostics(z_hat),
            "action_sensitivity": float((z_hat - z_shuf).norm(dim=-1).mean() / (z_hat.norm(dim=-1).mean() + 1e-9)),
        }
    return {"curve": curve, "diagnostics": diag, "steps": steps, "vic": vic}


__all__ = ["JepaArm", "build_transitions", "collapse_diagnostics", "train_jepa", "_tail_queries"]
