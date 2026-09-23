"""The batched online learner: ``B`` independent cells stepping in lockstep.

A *cell* is one model with its own parameters, hidden state, TBPTT window,
traces, hyperparameters and counters. Cells never exchange information; the
batch exists only so one array program can advance hundreds of experiments at
once on either backend. Per-cell hyperparameters (learning rate, gradient-clip
threshold, auxiliary-loss weight, ablation switches) are vectors, so a
hyperparameter sweep is simply a wider batch.

Learning rules
--------------
``live`` (TBPTT)
    AAA-1K's rule, reproduced exactly: the gradient of the latest step's loss
    is backpropagated through the last ``tbptt_steps`` *cached* activations
    using the *current* parameters. Cached activations were produced under
    older parameters, so this is online TBPTT with stale activations
    (``AAA-169``), not the exact truncated gradient.
``replay`` (TBPTT)
    the window is recomputed forward from its realized starting state under
    the current parameters, then backpropagated: the exact gradient of the
    latest loss at the current parameters, holding the window's start state
    and recorded inputs fixed.
``rtrl``
    diagonal cores only (:class:`~research.aaa_1k_v2.cores.LRUCore`):
    untruncated real-time recurrent learning with one trace per recurrent
    parameter.

Every rule applies plain per-cell SGD with an optional per-cell global-norm
clip. There is no optimizer state.

Failure is evidence
-------------------
A cell whose forward pass, gradient or update produces a non-finite value is
marked ``failed`` at that step, exactly as ``research.aaa_1k.model`` would
raise for that model alone. Cells never share arithmetic, so a non-finite
value cannot leave its cell; the failed cell stops updating and every metric
downstream reports it as having no primitive from its failure step on.
Nothing is silently repaired, and detection needs no host synchronization.

Loss
----
``L = (o0 - d*)^2 + lambda * (softplus(o1) - stopgrad(|o0 - d*|))^2`` with
two outputs, or just the first term for single-output cores. The arithmetic
order of every expression matches ``research.aaa_1k.model`` so the CPU path
can be compared with the historical implementation cell by cell.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from aaa.compute import Backend

from .cores import Core, LRUCore, mtv, outer, sigmoid

RULES = ("live", "replay", "rtrl")


@dataclass(frozen=True)
class CellConfig:
    """Hyperparameters of one cell."""

    learning_rate: float
    gradient_clip: float | None = None
    error_loss_weight: float = 0.25
    freeze_recurrent: bool = False
    reset_state_every_step: bool = False
    zero_input: tuple[int, ...] = ()
    optimizer: str = "sgd"
    """``sgd`` (no state), ``momentum`` (one parameter-sized vector) or ``adam`` (two). The tournament is
    SGD-only; the stateful optimizers exist for the optimizer diagnostic and are always accounted."""

    def __post_init__(self) -> None:
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"unknown optimizer {self.optimizer!r}")

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "learning_rate": self.learning_rate,
            "gradient_clip": self.gradient_clip,
            "error_loss_weight": self.error_loss_weight,
            "freeze_recurrent": self.freeze_recurrent,
            "reset_state_every_step": self.reset_state_every_step,
            "zero_input": list(self.zero_input),
        }
        if self.optimizer != "sgd":
            record["optimizer"] = self.optimizer
        return record


OPTIMIZERS = ("sgd", "momentum", "adam")
MOMENTUM = 0.9
ADAM_BETAS = (0.9, 0.999)
ADAM_EPSILON = 1e-8


def softplus(xp: Any, value: Any) -> Any:
    return xp.logaddexp(xp.zeros_like(value), value)


class BatchedLearner:
    """``B`` cells of one core type, on one backend."""

    def __init__(
        self,
        core: Core,
        backend: Backend,
        *,
        seeds: Sequence[int],
        configs: Sequence[CellConfig],
        tbptt_steps: int = 4,
        rule: str = "live",
        init_options: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        if len(seeds) != len(configs) or not seeds:
            raise ValueError("seeds and configs must be non-empty and the same length")
        if rule not in RULES:
            raise ValueError(f"unknown rule {rule!r}")
        if (rule == "rtrl") != (core.online_rule == "rtrl"):
            raise ValueError(f"core {core.kind} learns by {core.online_rule}, not {rule}")
        if isinstance(tbptt_steps, bool) or not isinstance(tbptt_steps, int) or tbptt_steps < 1:
            raise ValueError("tbptt_steps must be a positive integer")
        self.core = core
        self.backend = backend
        self.xp = backend.xp
        self.rule = rule
        self.tbptt_steps = tbptt_steps if rule != "rtrl" else 0
        self.seeds = [int(seed) for seed in seeds]
        self.configs = list(configs)
        self.size = len(seeds)
        options = list(init_options) if init_options is not None else [{} for _ in seeds]
        if len(options) != self.size:
            raise ValueError("init_options must match the number of cells")
        host = [core.init_cell(seed, **option) for seed, option in zip(self.seeds, options, strict=True)]
        shapes = core.shapes()
        for cell in host:
            if {name: tuple(array.shape) for name, array in cell.items()} != shapes:
                raise ValueError("core initialization does not match its declared shapes")
        self.params: dict[str, Any] = {
            name: backend.asarray(np.stack([cell[name] for cell in host])) for name in shapes
        }
        xp = self.xp
        B = self.size
        self.lr = backend.asarray([c.learning_rate for c in self.configs])
        self.clip = backend.asarray(
            [np.inf if c.gradient_clip is None else c.gradient_clip for c in self.configs]
        )
        self.error_weight = backend.asarray([c.error_loss_weight for c in self.configs])
        self.freeze_recurrent = backend.asarray([c.freeze_recurrent for c in self.configs], dtype=bool)
        self.reset_every_step = backend.asarray([c.reset_state_every_step for c in self.configs], dtype=bool)
        keep = np.ones((B, core.inputs), dtype=bool)
        for index, config in enumerate(self.configs):
            for column in config.zero_input:
                keep[index, column] = False
        self.input_keep = backend.asarray(keep, dtype=bool)
        self.any_reset_every_step = bool(np.any([c.reset_state_every_step for c in self.configs]))
        self.any_zero_input = not bool(keep.all())
        self.any_freeze = bool(np.any([c.freeze_recurrent for c in self.configs]))
        self._init_optimizer_state()
        self.h = backend.zeros((B, core.state_size()))
        self.epoch = xp.zeros(B, dtype=np.int64)
        self._caches: deque[dict[str, Any]] = deque(maxlen=max(self.tbptt_steps, 1))
        self.traces: dict[str, Any] = {}
        if isinstance(core, LRUCore):
            self.traces = {
                "E_lam": backend.zeros((B, 2, core.hidden)),
                "E_B": backend.zeros((B, 2, core.hidden, core.inputs)),
            }
        self._last: dict[str, Any] | None = None
        self.update_count = xp.zeros(B, dtype=np.int64)
        self.forward_count = 0
        self.clip_events = xp.zeros(B, dtype=np.int64)
        self.failed = xp.zeros(B, dtype=bool)
        self.fail_step = xp.full(B, -1, dtype=np.int64)
        self.step_index = 0

    def _init_optimizer_state(self) -> None:
        xp = self.xp
        kinds = [c.optimizer for c in self.configs]
        self.any_stateful = any(kind != "sgd" for kind in kinds)
        self.opt_momentum = xp.asarray(np.asarray([k == "momentum" for k in kinds]))
        self.opt_adam = xp.asarray(np.asarray([k == "adam" for k in kinds]))
        self.opt_m: dict[str, Any] = {}
        self.opt_v: dict[str, Any] = {}
        if self.any_stateful:
            self.opt_m = {name: xp.zeros_like(array) for name, array in self.params.items()}
            self.opt_v = {name: xp.zeros_like(array) for name, array in self.params.items()}

    # ------------------------------------------------------------------
    # accounting
    # ------------------------------------------------------------------
    def parameter_count(self) -> int:
        """Trainable scalars per cell, summed from the actual arrays."""

        return int(sum(int(np.prod(array.shape[1:])) for array in self.params.values()))

    def state_footprint(self) -> dict[str, int]:
        """Every adaptive scalar one cell carries, by category (no optimizer state: plain SGD)."""

        cache = self.core.cache_size() * self.tbptt_steps
        replay_extra = 0
        trainable = self.parameter_count()
        kinds = {c.optimizer for c in self.configs}
        per_cell_optimizer = 2 * trainable if "adam" in kinds else (trainable if "momentum" in kinds else 0)
        return {
            "trainable_parameters": trainable,
            "hidden_state_scalars": self.core.state_size(),
            "optimizer_state_scalars": per_cell_optimizer,
            "tbptt_buffer_scalars_capacity": cache + replay_extra,
            "eligibility_trace_scalars": self.core.trace_size(),
            "normalization_statistics_scalars": 0,
            "replay_memory_scalars": 0,
            "total_adaptive_state_scalars": trainable
            + self.core.state_size()
            + cache
            + self.core.trace_size()
            + per_cell_optimizer,
        }

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------
    def reset_state(self, mask: Any = None) -> None:
        """Begin new independent episodes: all cells, or the cells in ``mask``."""

        xp = self.xp
        if mask is None:
            self.h = xp.zeros_like(self.h)
            self._caches.clear()
            for name in self.traces:
                self.traces[name] = xp.zeros_like(self.traces[name])
            self.epoch = self.epoch + 1
            self._last = None
            return
        keep = ~xp.asarray(mask, dtype=bool)
        self.h = self.h * keep[:, None]
        for name, trace in self.traces.items():
            self.traces[name] = trace * keep.reshape((-1,) + (1,) * (trace.ndim - 1))
        self.epoch = self.epoch + (~keep).astype(np.int64)

    def _fail(self, bad: Any) -> None:
        """Mark cells failed at this step (device-side; no host synchronization).

        A failed cell's arrays are left as they are: cells never share
        arithmetic, so a non-finite value cannot leave its own cell. From the
        failure step on the cell no longer updates and every downstream metric
        treats it as having no primitive.
        """

        xp = self.xp
        new = bad & ~self.failed
        self.fail_step = xp.where(new, self.step_index, self.fail_step)
        self.failed = self.failed | new

    @staticmethod
    def _finite_rows(xp: Any, array: Any) -> Any:
        return xp.all(xp.isfinite(array.reshape(array.shape[0], -1)), axis=1)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, x: Any) -> Any:
        """Advance every cell one step and return raw outputs ``(B, outputs)``."""

        xp = self.xp
        if self.any_zero_input:
            x = xp.where(self.input_keep, x, 0.0)
        if self.any_reset_every_step:
            self.reset_state(self.reset_every_step)
        P = self.params
        if isinstance(self.core, LRUCore):
            h_new, traces = self.core.step_state(xp, P, x, self.h, self.traces)
            out, a = self.core.readout_forward(xp, P, h_new, x)
            self.traces = traces
            self._last = {"x": x, "h": h_new, "a": a, "out": out}
        else:
            h_new, cache = self.core.forward(xp, P, x, self.h)
            out = self.core.readout(xp, P, h_new)
            cache["out"] = out
            cache["epoch"] = self.epoch
            self._caches.append(cache)
            self._last = cache
        finite = self._finite_rows(xp, h_new) & self._finite_rows(xp, out)
        self.h = h_new
        self.forward_count += 1
        self._fail(~finite)
        return out

    # ------------------------------------------------------------------
    # learning
    # ------------------------------------------------------------------
    def output_gradient(self, out: Any, target: Any) -> Any:
        """``dL/d output``, in the historical arithmetic order."""

        xp = self.xp
        signed = out[:, 0] - target
        if out.shape[1] == 1:
            return (2.0 * signed)[:, None]
        estimate = softplus(xp, out[:, 1])
        error_target = xp.abs(signed)
        grad1 = self.error_weight * 2.0 * (estimate - error_target) * sigmoid(xp, out[:, 1])
        return xp.stack([2.0 * signed, grad1], axis=1)

    def _window_gradients(self, target: Any) -> dict[str, Any]:
        xp = self.xp
        P = self.params
        window = list(self._caches)
        if self.rule == "replay":
            window = self._replay_window(window)
        last = window[-1]
        d_out = self.output_gradient(last["out"], target)
        G = {name: xp.zeros_like(array) for name, array in P.items()}
        G["W_o"] += outer(d_out, last["h"])
        G["b_o"] += d_out
        d_h = mtv(xp, P["W_o"], d_out)
        for position, cache in enumerate(reversed(window)):
            if position > 0 or self.rule == "replay":
                # a transition from before this cell's last reset carries no gradient
                valid = cache["epoch"] == self.epoch
                d_h = xp.where(valid[:, None], d_h, 0.0)
            d_h = self.core.backward_step(xp, P, cache, d_h, G)
        return G

    def _replay_window(self, window: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Recompute the window forward from its realized start state under current parameters."""

        xp = self.xp
        P = self.params
        replayed: list[dict[str, Any]] = []
        h = window[0]["h_prev"]
        previous_valid = None
        for cache in window:
            valid = cache["epoch"] == self.epoch
            if previous_valid is not None:
                starts = valid & ~previous_valid
                h = xp.where(starts[:, None], cache["h_prev"], h)
            else:
                h = cache["h_prev"]
            h_new, fresh = self.core.forward(xp, P, cache["x"], h)
            fresh["out"] = self.core.readout(xp, P, h_new)
            fresh["epoch"] = cache["epoch"]
            replayed.append(fresh)
            h = h_new
            previous_valid = valid
        return replayed

    def gradients(self, target: Any) -> dict[str, Any]:
        """Gradient of the latest step's loss for every cell, under this learner's rule."""

        if self._last is None:
            raise RuntimeError("gradients() requires a forward step")
        if isinstance(self.core, LRUCore):
            last = self._last
            d_out = self.output_gradient(last["out"], target)
            return self.core.rtrl_gradients(
                self.xp, self.params, last["x"], last["h"], last["a"], d_out, self.traces
            )
        return self._window_gradients(target)

    def learn(self, target: Any, mask: Any) -> dict[str, Any]:
        """One SGD step on the latest forward pass, for cells in ``mask`` that have not failed."""

        xp = self.xp
        mask = xp.asarray(mask, dtype=bool) & ~self.failed
        G = self.gradients(target)
        total = xp.zeros(self.size)
        for name in self.params:
            g = G[name]
            total = total + xp.sum((g * g).reshape(self.size, -1), axis=1)
        norm = xp.sqrt(total)
        finite = xp.isfinite(norm)
        self._fail(~finite & mask)
        mask = mask & ~self.failed
        clipped = mask & (norm > self.clip)
        scale = xp.where(clipped, self.clip / xp.where(norm > 0, norm, 1.0), 1.0)
        self.clip_events = self.clip_events + clipped.astype(np.int64)
        step = xp.where(mask, self.lr * scale, 0.0)
        for name, array in self.params.items():
            if self.any_freeze and name in self.core.recurrent_names:
                cell_step = xp.where(self.freeze_recurrent, 0.0, step)
            else:
                cell_step = step
            shaped = cell_step.reshape((-1,) + (1,) * (array.ndim - 1))
            updated = array - shaped * G[name]
            if self.any_stateful:
                updated = self._stateful_update(name, array, updated, G[name], cell_step, scale, mask)
            self.params[name] = xp.where(mask.reshape((-1,) + (1,) * (array.ndim - 1)), updated, array)
        healthy = xp.ones(self.size, dtype=bool)
        for array in self.params.values():
            healthy = healthy & self._finite_rows(xp, array)
        self._fail(~healthy)
        self.update_count = self.update_count + (mask & ~self.failed).astype(np.int64)
        return {"gradient_norm": norm, "clipped": clipped, "updated": mask & ~self.failed}

    def _stateful_update(
        self, name: str, array: Any, sgd_updated: Any, gradient: Any, cell_step: Any, scale: Any, mask: Any
    ) -> Any:
        """Momentum / Adam for the cells that use them; SGD cells keep ``sgd_updated`` exactly."""

        xp = self.xp
        shape = (-1,) + (1,) * (array.ndim - 1)
        live = mask.reshape(shape)
        g = gradient * scale.reshape(shape)
        m_old, v_old = self.opt_m[name], self.opt_v[name]
        beta1, beta2 = ADAM_BETAS
        momentum_m = MOMENTUM * m_old + g
        adam_m = beta1 * m_old + (1.0 - beta1) * g
        adam_v = beta2 * v_old + (1.0 - beta2) * g * g
        t = (self.update_count + 1).astype(np.float64).reshape(shape)
        adam_step = (adam_m / (1.0 - beta1**t)) / (xp.sqrt(adam_v / (1.0 - beta2**t)) + ADAM_EPSILON)
        lr = xp.where(cell_step > 0, self.lr, 0.0).reshape(shape)
        momentum_cells = self.opt_momentum.reshape(shape)
        adam_cells = self.opt_adam.reshape(shape)
        new_m = xp.where(momentum_cells, momentum_m, xp.where(adam_cells, adam_m, m_old))
        self.opt_m[name] = xp.where(live, new_m, m_old)
        self.opt_v[name] = xp.where(live & adam_cells, adam_v, v_old)
        stateful = xp.where(momentum_cells, array - lr * momentum_m, array - lr * adam_step)
        return xp.where(momentum_cells | adam_cells, stateful, sgd_updated)

    def advance_clock(self) -> None:
        self.step_index += 1

    def take(self, indices: Sequence[int], configs: Sequence[CellConfig] | None = None) -> BatchedLearner:
        """A new learner whose cells are exact copies of ``indices`` (duplicates allowed).

        This is how a trunk branches: ``take(idx + idx)`` gives two bitwise
        identical copies of every trunk cell, whose update masks can then
        differ. ``configs`` may replace the copied hyperparameters, e.g. to
        make one copy frozen; the arrays, window, traces and counters are
        copied unchanged.
        """

        xp = self.xp
        index = xp.asarray(np.asarray(indices, dtype=np.int64))
        clone = object.__new__(BatchedLearner)
        clone.core, clone.backend, clone.xp, clone.rule = self.core, self.backend, xp, self.rule
        clone.tbptt_steps = self.tbptt_steps
        clone.seeds = [self.seeds[i] for i in indices]
        clone.configs = list(configs) if configs is not None else [self.configs[i] for i in indices]
        if len(clone.configs) != len(indices):
            raise ValueError("configs must match indices")
        clone.size = len(indices)
        clone.params = {name: array[index].copy() for name, array in self.params.items()}
        backend = self.backend
        clone.lr = backend.asarray([c.learning_rate for c in clone.configs])
        clone.clip = backend.asarray(
            [np.inf if c.gradient_clip is None else c.gradient_clip for c in clone.configs]
        )
        clone.error_weight = backend.asarray([c.error_loss_weight for c in clone.configs])
        clone.freeze_recurrent = backend.asarray([c.freeze_recurrent for c in clone.configs], dtype=bool)
        clone.reset_every_step = backend.asarray(
            [c.reset_state_every_step for c in clone.configs], dtype=bool
        )
        keep = np.ones((clone.size, self.core.inputs), dtype=bool)
        for row, config in enumerate(clone.configs):
            for column in config.zero_input:
                keep[row, column] = False
        clone.input_keep = backend.asarray(keep, dtype=bool)
        clone.any_reset_every_step = bool(np.any([c.reset_state_every_step for c in clone.configs]))
        clone.any_zero_input = not bool(keep.all())
        clone.any_freeze = bool(np.any([c.freeze_recurrent for c in clone.configs]))
        clone.any_stateful = any(c.optimizer != "sgd" for c in clone.configs)
        clone.opt_momentum = xp.asarray(np.asarray([c.optimizer == "momentum" for c in clone.configs]))
        clone.opt_adam = xp.asarray(np.asarray([c.optimizer == "adam" for c in clone.configs]))
        clone.opt_m = {name: array[index].copy() for name, array in self.opt_m.items()}
        clone.opt_v = {name: array[index].copy() for name, array in self.opt_v.items()}
        if clone.any_stateful and not clone.opt_m:
            clone.opt_m = {name: xp.zeros_like(array) for name, array in clone.params.items()}
            clone.opt_v = {name: xp.zeros_like(array) for name, array in clone.params.items()}
        clone.h = self.h[index].copy()
        clone.epoch = self.epoch[index].copy()
        clone._caches = deque(
            ({key: value[index].copy() for key, value in cache.items()} for cache in self._caches),
            maxlen=self._caches.maxlen,
        )
        clone.traces = {name: array[index].copy() for name, array in self.traces.items()}
        if self._last is None:
            clone._last = None
        elif isinstance(self.core, LRUCore):
            clone._last = {key: value[index].copy() for key, value in self._last.items()}
        else:
            clone._last = clone._caches[-1] if clone._caches else None
        clone.update_count = self.update_count[index].copy()
        clone.forward_count = self.forward_count
        clone.clip_events = self.clip_events[index].copy()
        clone.failed = self.failed[index].copy()
        clone.fail_step = self.fail_step[index].copy()
        clone.step_index = self.step_index
        return clone

    def cell_state_hashes(self) -> list[str]:
        """SHA-256 of each cell's complete learner state (branch-identity checks)."""

        import hashlib

        host = self.backend.to_host
        arrays = [host(a) for a in self.params.values()] + [host(self.h)]
        arrays += [host(v) for cache in self._caches for k, v in cache.items() if k != "epoch"]
        arrays += [host(a) for a in self.traces.values()]
        digests = []
        for row in range(self.size):
            digest = hashlib.sha256()
            for array in arrays:
                digest.update(np.ascontiguousarray(array[row]).tobytes())
            digests.append(digest.hexdigest())
        return digests

    # ------------------------------------------------------------------
    # checkpoint
    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """Complete host-side state: parameters, hidden state, window, traces, counters."""

        host = self.backend.to_host
        return {
            "core": self.core.describe(),
            "rule": self.rule,
            "tbptt_steps": self.tbptt_steps,
            "seeds": list(self.seeds),
            "configs": [c.to_dict() for c in self.configs],
            "params": {name: host(array).tolist() for name, array in self.params.items()},
            "h": host(self.h).tolist(),
            "epoch": host(self.epoch).tolist(),
            "caches": [{key: host(value).tolist() for key, value in cache.items()} for cache in self._caches],
            "traces": {name: host(array).tolist() for name, array in self.traces.items()},
            "optimizer_state": {
                "m": {name: host(array).tolist() for name, array in self.opt_m.items()},
                "v": {name: host(array).tolist() for name, array in self.opt_v.items()},
            },
            "last": None if self._last is None else {k: host(v).tolist() for k, v in self._last.items()},
            "counters": {
                "update_count": host(self.update_count).tolist(),
                "forward_count": self.forward_count,
                "clip_events": host(self.clip_events).tolist(),
                "failed": host(self.failed).tolist(),
                "fail_step": host(self.fail_step).tolist(),
                "step_index": self.step_index,
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore :meth:`state_dict` output; refuses a mismatched core or batch."""

        if state["core"] != self.core.describe() or state["rule"] != self.rule:
            raise ValueError("checkpoint core or rule does not match this learner")
        if list(state["seeds"]) != self.seeds or state["tbptt_steps"] != self.tbptt_steps:
            raise ValueError("checkpoint cells do not match this learner")
        asarray = self.backend.asarray
        shapes = self.core.shapes()
        for name, value in state["params"].items():
            array = asarray(value)
            if tuple(array.shape) != (self.size, *shapes[name]):
                raise ValueError(f"checkpoint parameter {name} has the wrong shape")
            if not bool(self.xp.all(self.xp.isfinite(array))):
                raise ValueError(f"checkpoint parameter {name} is not finite")
            self.params[name] = array
        self.h = asarray(state["h"])
        self.epoch = asarray(state["epoch"], dtype=np.int64)
        self._caches.clear()
        for cache in state["caches"]:
            self._caches.append(
                {
                    key: asarray(value, dtype=np.int64 if key == "epoch" else None)
                    for key, value in cache.items()
                }
            )
        self.traces = {name: asarray(value) for name, value in state["traces"].items()}
        optimizer_state = state.get("optimizer_state", {"m": {}, "v": {}})
        self.opt_m = {name: asarray(value) for name, value in optimizer_state["m"].items()}
        self.opt_v = {name: asarray(value) for name, value in optimizer_state["v"].items()}
        if self.any_stateful and set(self.opt_m) != set(self.params):
            raise ValueError("checkpoint is missing optimizer state for a stateful optimizer")
        last = state["last"]
        if last is None:
            self._last = None
        elif isinstance(self.core, LRUCore):
            self._last = {k: asarray(v) for k, v in last.items()}
        else:
            self._last = self._caches[-1] if self._caches else None
        counters = state["counters"]
        self.update_count = asarray(counters["update_count"], dtype=np.int64)
        self.forward_count = int(counters["forward_count"])
        self.clip_events = asarray(counters["clip_events"], dtype=np.int64)
        self.failed = asarray(counters["failed"], dtype=bool)
        self.fail_step = asarray(counters["fail_step"], dtype=np.int64)
        self.step_index = int(counters["step_index"])
