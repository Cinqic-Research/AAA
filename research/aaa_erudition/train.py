"""Train an Erudition Model on simulated counterfactual returns.

Streams are split into training and validation by stream index. The decision
margin (how much better than waiting an action must look) is chosen on
validation streams by minimum regret, and recorded with the model. Every
input file, its SHA-256, the architecture, seed, hardware and final metrics
go into the model's metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .erudition import Architecture, EruditionController, EruditionNet, count_parameters, menu_vector, window
from .lifecycle import ACTION_NAMES, CONDITIONS

MARGINS = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.04)


def load(paths: list[Path]) -> list[dict[str, Any]]:
    streams: list[dict[str, Any]] = []
    for path in paths:
        with path.open() as stream:
            streams.extend(json.loads(line) for line in stream if line.strip())
    streams.sort(key=lambda s: (s["continuation"], s["index"]))
    return streams


def tensors(streams: list[dict[str, Any]], diag_stride: int) -> dict[str, torch.Tensor]:
    xs, pads, menus, targets, tmask, diags = [], [], [], [], [], []
    for s in streams:
        features = np.array(s["features"], dtype=np.float32)
        menu = menu_vector(CONDITIONS[s["condition"]])
        decision_steps = {e["step"]: e for e in s["examples"]}
        for step in range(len(features)):
            example = decision_steps.get(step)
            if example is None and step % diag_stride:
                continue
            x, pad = window(features[: step + 1])
            target = np.zeros(len(ACTION_NAMES), dtype=np.float32)
            mask = np.zeros(len(ACTION_NAMES), dtype=np.float32)
            if example is not None:
                returns = example["returns"]
                centre = float(np.mean(list(returns.values())))
                for action, value in returns.items():
                    target[ACTION_NAMES.index(action)] = value - centre
                    mask[ACTION_NAMES.index(action)] = 1.0
            xs.append(x)
            pads.append(pad)
            menus.append(menu)
            targets.append(target)
            tmask.append(mask)
            diags.append(s["diagnoses"][step])
    return {
        "x": torch.from_numpy(np.stack(xs)),
        "pad": torch.from_numpy(np.stack(pads)),
        "menu": torch.from_numpy(np.stack(menus)),
        "target": torch.from_numpy(np.stack(targets)),
        "tmask": torch.from_numpy(np.stack(tmask)),
        "diag": torch.tensor(diags, dtype=torch.long),
    }


def evaluate(
    net: EruditionNet, data: dict[str, torch.Tensor], device: str, chunk: int = 1024
) -> dict[str, Any]:
    net.eval()
    qs, ds = [], []
    with torch.no_grad():
        for i in range(0, len(data["x"]), chunk):
            q, d = net(
                data["x"][i : i + chunk].to(device),
                data["pad"][i : i + chunk].to(device),
                data["menu"][i : i + chunk].to(device),
            )
            qs.append(q.cpu())
            ds.append(d.cpu())
    q = torch.cat(qs)
    d = torch.cat(ds)
    decision = data["tmask"].sum(1) > 0
    target, mask = data["target"][decision], data["tmask"][decision]
    qd = q[decision]
    wait = ACTION_NAMES.index("wait")
    best_value = (target - (1 - mask) * 1e9).max(1).values
    regrets = {}
    for margin in MARGINS:
        masked = qd - (1 - mask) * 1e9
        masked[:, wait] = -1e9
        best = masked.argmax(1)
        take = (masked.gather(1, best[:, None])[:, 0] - qd[:, wait]) > margin
        chosen = torch.where(take, best, torch.full_like(best, wait))
        regrets[str(margin)] = float((best_value - target.gather(1, chosen[:, None])[:, 0]).mean())
    q_mse = float((((qd - target) ** 2) * mask).sum() / mask.sum())
    diag_acc = float((d.argmax(1) == data["diag"]).float().mean())
    oracle_wait_regret = float((best_value - target[:, wait]).mean())
    return {
        "regret": regrets,
        "q_mse": q_mse,
        "diagnosis_accuracy": diag_acc,
        "wait_regret": oracle_wait_regret,
        "decisions": int(decision.sum()),
    }


def train(
    data_paths: list[Path],
    arch: Architecture,
    epochs: int,
    seed: int,
    device: str,
    output: Path,
    validation_fraction: float,
    micro_batch: int = 256,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    streams = load(data_paths)
    indices = sorted({s["index"] for s in streams})
    cut = int(len(indices) * (1 - validation_fraction))
    held = set(indices[cut:])
    train_data = tensors([s for s in streams if s["index"] not in held], diag_stride=4)
    val_data = tensors([s for s in streams if s["index"] in held], diag_stride=4)
    net = EruditionNet(arch).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.01)
    steps_per_epoch = (len(train_data["x"]) + 255) // 256
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-4, total_steps=epochs * steps_per_epoch
    )
    history = []
    started = time.time()
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        net.train()
        order = torch.randperm(len(train_data["x"]), generator=generator)
        total = 0.0
        for i in range(0, len(order), 256):
            batch = order[i : i + 256]
            optimizer.zero_grad()
            # Gradient accumulation keeps the effective batch at 256 when a model does not fit
            # beside the resident Language Model; the update is the same up to float rounding.
            for j in range(0, len(batch), micro_batch):
                part = batch[j : j + micro_batch]
                x = train_data["x"][part].to(device)
                q, d = net(x, train_data["pad"][part].to(device), train_data["menu"][part].to(device))
                mask = train_data["tmask"][part].to(device)
                target = train_data["target"][part].to(device)
                q_loss = (((q - target) ** 2) * mask).sum() / train_data["tmask"][batch].sum().clamp(min=1.0)
                diag_loss = nn.functional.cross_entropy(
                    d, train_data["diag"][part].to(device), reduction="sum"
                ) / len(batch)
                loss = 100.0 * q_loss + diag_loss
                loss.backward()
                total += float(loss) * len(batch)
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            schedule.step()
        metrics = evaluate(net, val_data, device, chunk=4 * micro_batch)
        history.append({"epoch": epoch, "train_loss": total / len(order), **metrics})
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "loss": round(total / len(order), 5),
                    "regret": metrics["regret"],
                    "diag": round(metrics["diagnosis_accuracy"], 3),
                }
            ),
            flush=True,
        )
    final = history[-1]
    margin = float(min(MARGINS, key=lambda m: final["regret"][str(m)]))
    net = net.cpu()
    meta = {
        "seed": seed,
        "epochs": epochs,
        "device": device,
        "hardware": platform.platform(),
        "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "torch": torch.__version__,
        "train_seconds": time.time() - started,
        "micro_batch": micro_batch,
        "data": [{"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in data_paths],
        "streams": len(streams),
        "validation_streams": len(held),
        "train_examples": len(train_data["x"]),
        "validation": final,
        "history": history,
        "selection": "margin with minimum validation regret",
    }
    controller = EruditionController(net, margin, meta)
    controller.save(output)
    return {"trainable_parameters": count_parameters(net), "margin": margin, **final}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m research.aaa_erudition.train")
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", default="transformer")
    parser.add_argument("--d-model", type=int, default=160)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--micro-batch", type=int, default=256)
    args = parser.parse_args(argv)
    arch = Architecture(
        kind=args.kind, d_model=args.d_model, layers=args.layers, heads=args.heads, ff=args.ff
    )
    result = train(
        args.data,
        arch,
        args.epochs,
        args.seed,
        args.device,
        args.output,
        args.validation_fraction,
        args.micro_batch,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
