"""Recurrent cores for the batched online learner, written once for NumPy and CuPy.

Every core works on a *batch* of independent cells. Each cell has its own
parameters, hidden state and learning history; nothing is shared across cells
except the array program. Arrays carry a leading batch axis ``B``: a parameter
declared with per-cell shape ``(H, I)`` is stored as ``(B, H, I)``.

The array module ``xp`` is passed in (``numpy`` or ``cupy``), so the arithmetic
below is the arithmetic on either device. Nothing here branches on the device.

Cores
-----
``GRUCore``
    AAA-1K's frozen GRU convention exactly: ``z`` is the *keep* gate, one bias
    per gate, Glorot-uniform input and recurrent matrices drawn in the order
    ``W_z, U_z, W_r, U_r, W_n, U_n`` from one model-local generator, zero
    biases, zero output head. With ``inputs=3, hidden=16`` a cell initialized
    from seed ``s`` has bit-identical initial parameters to
    ``research.aaa_1k.model.AAA1KGRU(seed=s)``.
``ElmanCore``
    ``h' = tanh(W x + U h + b)``; ``inputs=3, hidden=28`` reproduces the
    initialization of ``VanillaRNNControl``.
``MGUCore``
    the minimal gated unit (Zhou et al. 2016): one gate ``f`` both resets the
    candidate's recurrent input and interpolates, ``h' = (1-f) h + f n``.
``MLPCore``
    the stateless control ``3 -> H -> H -> O`` (no recurrence).
``LRUCore``
    a complex-diagonal *linear* recurrence ``h' = lambda * h + B x`` with
    ``|lambda| < 1`` by construction (Orvieto et al. 2023), read out by a small
    tanh network. Because the recurrence is diagonal, the exact real-time
    recurrent-learning gradient needs only one trace per recurrent parameter
    (Zucchet et al. 2023), so this core learns with an *untruncated* online
    gradient instead of TBPTT. The traces are adaptive state and are counted.

Per-cell *initial* parameters are always generated on the host with NumPy's
``default_rng`` from the cell's seed and then transferred, so a cell's initial
weights never depend on the backend.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

Array = Any
"""A NumPy or CuPy array."""


def mv(xp: Any, matrix: Array, vector: Array) -> Array:
    """Batched matrix-vector product: ``(B, M, N) x (B, N) -> (B, M)``."""

    return xp.matmul(matrix, vector[..., None])[..., 0]


def mtv(xp: Any, matrix: Array, vector: Array) -> Array:
    """Batched transposed product ``A^T v``: ``(B, M, N) x (B, M) -> (B, N)``."""

    return xp.matmul(vector[:, None, :], matrix)[:, 0, :]


def outer(left: Array, right: Array) -> Array:
    """Batched outer product: ``(B, M) x (B, N) -> (B, M, N)``."""

    return left[:, :, None] * right[:, None, :]


def sigmoid(xp: Any, value: Array) -> Array:
    """The same split formula as ``research.aaa_1k.model.sigmoid``, elementwise."""

    t = xp.exp(-xp.abs(value))
    return xp.where(value >= 0, 1.0 / (1.0 + t), t / (1.0 + t))


def glorot(rng: np.random.Generator, rows: int, columns: int) -> np.ndarray:
    limit = math.sqrt(6.0 / (rows + columns))
    return rng.uniform(-limit, limit, size=(rows, columns))


@dataclass(frozen=True)
class Core:
    """Common interface. Subclasses define shapes, initialization, forward and backward."""

    inputs: int
    hidden: int
    outputs: int = 2
    kind: str = field(default="", init=False)
    online_rule: str = field(default="tbptt", init=False)
    recurrent_names: tuple[str, ...] = field(default=(), init=False)

    # -- declaration --------------------------------------------------
    def shapes(self) -> dict[str, tuple[int, ...]]:
        raise NotImplementedError

    def parameter_count(self) -> int:
        return int(sum(int(np.prod(shape)) for shape in self.shapes().values()))

    def state_size(self) -> int:
        """Recurrent state scalars per cell."""

        return self.hidden

    def cache_size(self) -> int:
        """Scalars one TBPTT cache entry holds per cell (0 for RTRL cores)."""

        raise NotImplementedError

    def trace_size(self) -> int:
        """Eligibility-trace scalars per cell (RTRL cores only)."""

        return 0

    def describe(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "inputs": self.inputs,
            "hidden": self.hidden,
            "outputs": self.outputs,
            "online_rule": self.online_rule,
            "parameters": self.parameter_count(),
            "parameter_shapes": {name: list(shape) for name, shape in self.shapes().items()},
        }

    def init_cell(self, seed: int, **options: Any) -> dict[str, np.ndarray]:
        raise NotImplementedError

    # -- computation --------------------------------------------------
    def forward(self, xp: Any, P: Mapping[str, Array], x: Array, h: Array) -> tuple[Array, dict[str, Array]]:
        raise NotImplementedError

    def backward_step(
        self, xp: Any, P: Mapping[str, Array], cache: Mapping[str, Array], d_h: Array, G: dict[str, Array]
    ) -> Array:
        """Accumulate one transition's parameter gradients into ``G``; return ``dL/dh_prev``."""

        raise NotImplementedError

    def readout(self, xp: Any, P: Mapping[str, Array], h: Array) -> Array:
        return mv(xp, P["W_o"], h) + P["b_o"]


def _readout_shapes(hidden: int, outputs: int) -> dict[str, tuple[int, ...]]:
    return {"W_o": (outputs, hidden), "b_o": (outputs,)}


def _zero_readout(hidden: int, outputs: int) -> dict[str, np.ndarray]:
    return {"W_o": np.zeros((outputs, hidden), dtype=float), "b_o": np.zeros(outputs, dtype=float)}


# ----------------------------------------------------------------------
# GRU (AAA-1K convention)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class GRUCore(Core):
    kind: str = field(default="gru", init=False)
    recurrent_names: tuple[str, ...] = field(default=("U_z", "U_r", "U_n"), init=False)

    def shapes(self) -> dict[str, tuple[int, ...]]:
        shapes: dict[str, tuple[int, ...]] = {}
        for gate in ("z", "r", "n"):
            shapes[f"W_{gate}"] = (self.hidden, self.inputs)
            shapes[f"U_{gate}"] = (self.hidden, self.hidden)
            shapes[f"b_{gate}"] = (self.hidden,)
        shapes.update(_readout_shapes(self.hidden, self.outputs))
        return shapes

    def cache_size(self) -> int:
        return self.inputs + 6 * self.hidden + self.outputs

    def init_cell(self, seed: int, *, keep_bias: Any = 0.0, **_: Any) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        params: dict[str, np.ndarray] = {}
        for gate in ("z", "r", "n"):
            params[f"W_{gate}"] = glorot(rng, self.hidden, self.inputs)
            params[f"U_{gate}"] = glorot(rng, self.hidden, self.hidden)
            params[f"b_{gate}"] = np.zeros(self.hidden, dtype=float)
        params["b_z"][:] = np.asarray(keep_bias, dtype=float)
        params.update(_zero_readout(self.hidden, self.outputs))
        return params

    def forward(self, xp: Any, P: Mapping[str, Array], x: Array, h: Array) -> tuple[Array, dict[str, Array]]:
        z = sigmoid(xp, mv(xp, P["W_z"], x) + mv(xp, P["U_z"], h) + P["b_z"])
        r = sigmoid(xp, mv(xp, P["W_r"], x) + mv(xp, P["U_r"], h) + P["b_r"])
        hr = r * h
        n = xp.tanh(mv(xp, P["W_n"], x) + mv(xp, P["U_n"], hr) + P["b_n"])
        h_new = z * h + (1.0 - z) * n
        return h_new, {"x": x, "h_prev": h, "z": z, "r": r, "n": n, "hr": hr, "h": h_new}

    def backward_step(
        self, xp: Any, P: Mapping[str, Array], cache: Mapping[str, Array], d_h: Array, G: dict[str, Array]
    ) -> Array:
        z, r, n, x, h_prev = cache["z"], cache["r"], cache["n"], cache["x"], cache["h_prev"]
        d_z = d_h * (h_prev - n)
        d_n = d_h * (1.0 - z)
        d_h_prev = d_h * z
        d_a_n = d_n * (1.0 - n * n)
        G["W_n"] += outer(d_a_n, x)
        G["U_n"] += outer(d_a_n, cache["hr"])
        G["b_n"] += d_a_n
        d_hr = mtv(xp, P["U_n"], d_a_n)
        d_r = d_hr * h_prev
        d_h_prev = d_h_prev + d_hr * r
        d_a_r = d_r * r * (1.0 - r)
        G["W_r"] += outer(d_a_r, x)
        G["U_r"] += outer(d_a_r, h_prev)
        G["b_r"] += d_a_r
        d_h_prev = d_h_prev + mtv(xp, P["U_r"], d_a_r)
        d_a_z = d_z * z * (1.0 - z)
        G["W_z"] += outer(d_a_z, x)
        G["U_z"] += outer(d_a_z, h_prev)
        G["b_z"] += d_a_z
        return d_h_prev + mtv(xp, P["U_z"], d_a_z)


# ----------------------------------------------------------------------
# Elman (ungated) recurrence
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ElmanCore(Core):
    kind: str = field(default="elman", init=False)
    recurrent_names: tuple[str, ...] = field(default=("U",), init=False)

    def shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "W": (self.hidden, self.inputs),
            "U": (self.hidden, self.hidden),
            "b": (self.hidden,),
            **_readout_shapes(self.hidden, self.outputs),
        }

    def cache_size(self) -> int:
        return self.inputs + 2 * self.hidden + self.outputs

    def init_cell(self, seed: int, *, recurrent_scale: float = 1.0, **_: Any) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        params = {
            "W": glorot(rng, self.hidden, self.inputs),
            "U": glorot(rng, self.hidden, self.hidden) * float(recurrent_scale),
            "b": np.zeros(self.hidden, dtype=float),
        }
        params.update(_zero_readout(self.hidden, self.outputs))
        return params

    def forward(self, xp: Any, P: Mapping[str, Array], x: Array, h: Array) -> tuple[Array, dict[str, Array]]:
        h_new = xp.tanh(mv(xp, P["W"], x) + mv(xp, P["U"], h) + P["b"])
        return h_new, {"x": x, "h_prev": h, "h": h_new}

    def backward_step(
        self, xp: Any, P: Mapping[str, Array], cache: Mapping[str, Array], d_h: Array, G: dict[str, Array]
    ) -> Array:
        h = cache["h"]
        d_a = d_h * (1.0 - h * h)
        G["W"] += outer(d_a, cache["x"])
        G["U"] += outer(d_a, cache["h_prev"])
        G["b"] += d_a
        return mtv(xp, P["U"], d_a)


# ----------------------------------------------------------------------
# Minimal gated unit
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MGUCore(Core):
    kind: str = field(default="mgu", init=False)
    recurrent_names: tuple[str, ...] = field(default=("U_f", "U_n"), init=False)

    def shapes(self) -> dict[str, tuple[int, ...]]:
        shapes: dict[str, tuple[int, ...]] = {}
        for gate in ("f", "n"):
            shapes[f"W_{gate}"] = (self.hidden, self.inputs)
            shapes[f"U_{gate}"] = (self.hidden, self.hidden)
            shapes[f"b_{gate}"] = (self.hidden,)
        shapes.update(_readout_shapes(self.hidden, self.outputs))
        return shapes

    def cache_size(self) -> int:
        return self.inputs + 5 * self.hidden + self.outputs

    def init_cell(self, seed: int, *, gate_bias: float = 0.0, **_: Any) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        params: dict[str, np.ndarray] = {}
        for gate in ("f", "n"):
            params[f"W_{gate}"] = glorot(rng, self.hidden, self.inputs)
            params[f"U_{gate}"] = glorot(rng, self.hidden, self.hidden)
            params[f"b_{gate}"] = np.zeros(self.hidden, dtype=float)
        params["b_f"][:] = float(gate_bias)
        params.update(_zero_readout(self.hidden, self.outputs))
        return params

    def forward(self, xp: Any, P: Mapping[str, Array], x: Array, h: Array) -> tuple[Array, dict[str, Array]]:
        f = sigmoid(xp, mv(xp, P["W_f"], x) + mv(xp, P["U_f"], h) + P["b_f"])
        fh = f * h
        n = xp.tanh(mv(xp, P["W_n"], x) + mv(xp, P["U_n"], fh) + P["b_n"])
        h_new = (1.0 - f) * h + f * n
        return h_new, {"x": x, "h_prev": h, "f": f, "fh": fh, "n": n, "h": h_new}

    def backward_step(
        self, xp: Any, P: Mapping[str, Array], cache: Mapping[str, Array], d_h: Array, G: dict[str, Array]
    ) -> Array:
        f, n, x, h_prev = cache["f"], cache["n"], cache["x"], cache["h_prev"]
        d_f = d_h * (n - h_prev)
        d_n = d_h * f
        d_h_prev = d_h * (1.0 - f)
        d_a_n = d_n * (1.0 - n * n)
        G["W_n"] += outer(d_a_n, x)
        G["U_n"] += outer(d_a_n, cache["fh"])
        G["b_n"] += d_a_n
        d_fh = mtv(xp, P["U_n"], d_a_n)
        d_f = d_f + d_fh * h_prev
        d_h_prev = d_h_prev + d_fh * f
        d_a_f = d_f * f * (1.0 - f)
        G["W_f"] += outer(d_a_f, x)
        G["U_f"] += outer(d_a_f, h_prev)
        G["b_f"] += d_a_f
        return d_h_prev + mtv(xp, P["U_f"], d_a_f)


# ----------------------------------------------------------------------
# stateless MLP control
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MLPCore(Core):
    """``x -> tanh -> tanh -> out``. ``hidden`` is both layers' width; state size 0."""

    kind: str = field(default="mlp", init=False)

    def shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "W1": (self.hidden, self.inputs),
            "b1": (self.hidden,),
            "W2": (self.hidden, self.hidden),
            "b2": (self.hidden,),
            **_readout_shapes(self.hidden, self.outputs),
        }

    def state_size(self) -> int:
        return 0

    def cache_size(self) -> int:
        return self.inputs + 2 * self.hidden + self.outputs

    def init_cell(self, seed: int, **_: Any) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        params = {
            "W1": glorot(rng, self.hidden, self.inputs),
            "b1": np.zeros(self.hidden, dtype=float),
            "W2": glorot(rng, self.hidden, self.hidden),
            "b2": np.zeros(self.hidden, dtype=float),
        }
        params.update(_zero_readout(self.hidden, self.outputs))
        return params

    def forward(self, xp: Any, P: Mapping[str, Array], x: Array, h: Array) -> tuple[Array, dict[str, Array]]:
        del h  # stateless: the recurrent state is never read
        a1 = xp.tanh(mv(xp, P["W1"], x) + P["b1"])
        a2 = xp.tanh(mv(xp, P["W2"], a1) + P["b2"])
        return a2, {"x": x, "a1": a1, "h": a2}

    def backward_step(
        self, xp: Any, P: Mapping[str, Array], cache: Mapping[str, Array], d_h: Array, G: dict[str, Array]
    ) -> Array:
        a1, a2 = cache["a1"], cache["h"]
        d_a2 = d_h * (1.0 - a2 * a2)
        G["W2"] += outer(d_a2, a1)
        G["b2"] += d_a2
        d_a1 = mtv(xp, P["W2"], d_a2) * (1.0 - a1 * a1)
        G["W1"] += outer(d_a1, cache["x"])
        G["b1"] += d_a1
        return xp.zeros_like(d_h)


# ----------------------------------------------------------------------
# complex-diagonal linear recurrence with exact online gradient
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class LRUCore(Core):
    """``h' = lambda h + B x`` (complex, diagonal) read out by ``tanh(W1 [Re h, Im h] + V x + b1)``.

    ``hidden`` is the number of complex modes; ``readout_hidden`` is the width
    of the tanh layer. ``lambda_k = exp(-exp(nu_k) + i theta_k)``, so
    ``|lambda_k| = exp(-exp(nu_k)) < 1`` for every finite ``nu``: the linear
    recurrence cannot become unstable however the parameters move.
    """

    readout_hidden: int = 13
    kind: str = field(default="lru", init=False)
    online_rule: str = field(default="rtrl", init=False)
    recurrent_names: tuple[str, ...] = field(default=("nu", "theta", "B_re", "B_im"), init=False)

    def shapes(self) -> dict[str, tuple[int, ...]]:
        H, n_in, M = self.hidden, self.inputs, self.readout_hidden
        return {
            "nu": (H,),
            "theta": (H,),
            "B_re": (H, n_in),
            "B_im": (H, n_in),
            "W1": (M, 2 * H),
            "V": (M, n_in),
            "b1": (M,),
            "W_o": (self.outputs, M),
            "b_o": (self.outputs,),
        }

    def state_size(self) -> int:
        return 2 * self.hidden

    def cache_size(self) -> int:
        return 0

    def trace_size(self) -> int:
        return 2 * self.hidden + 2 * self.hidden * self.inputs

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "readout_hidden": self.readout_hidden}

    def init_cell(
        self,
        seed: int,
        *,
        timescale_min: float = 2.0,
        timescale_max: float = 200.0,
        theta_max: float = math.pi / 2,
        **_: Any,
    ) -> dict[str, np.ndarray]:
        H, n_in, M = self.hidden, self.inputs, self.readout_hidden
        rng = np.random.default_rng(seed)
        timescale = np.exp(rng.uniform(math.log(timescale_min), math.log(timescale_max), size=H))
        nu = np.log(1.0 / timescale)  # |lambda| = exp(-1/timescale)
        theta = rng.uniform(0.0, theta_max, size=H)
        modulus = np.exp(-1.0 / timescale)
        gamma = np.sqrt(1.0 - modulus * modulus)[:, None]  # keeps each mode's stationary variance O(1)
        limit = math.sqrt(6.0 / (H + n_in))
        return {
            "nu": nu,
            "theta": theta,
            "B_re": rng.uniform(-limit, limit, size=(H, n_in)) * gamma,
            "B_im": rng.uniform(-limit, limit, size=(H, n_in)) * gamma,
            "W1": glorot(rng, M, 2 * H),
            "V": glorot(rng, M, n_in),
            "b1": np.zeros(M, dtype=float),
            "W_o": np.zeros((self.outputs, M), dtype=float),
            "b_o": np.zeros(self.outputs, dtype=float),
        }

    @staticmethod
    def lam(xp: Any, P: Mapping[str, Array]) -> tuple[Array, Array, Array]:
        modulus = xp.exp(-xp.exp(P["nu"]))
        return modulus * xp.cos(P["theta"]), modulus * xp.sin(P["theta"]), modulus

    def step_state(
        self, xp: Any, P: Mapping[str, Array], x: Array, h: Array, traces: dict[str, Array]
    ) -> tuple[Array, dict[str, Array]]:
        """Advance the complex state and the RTRL traces together.

        ``h`` is ``(B, 2H)`` = ``[Re, Im]``. Traces: ``E_lam`` = dh/dlambda
        (complex, ``(B, 2, H)``) and ``E_B`` = dh/dB (complex, ``(B, 2, H, I)``),
        both computed *before* the state update from ``h_{t-1}`` and ``x_t``.
        """

        H = self.hidden
        lr, li, _ = self.lam(xp, P)
        hr, hi = h[:, :H], h[:, H:]
        er, ei = traces["E_lam"][:, 0], traces["E_lam"][:, 1]
        new_er = lr * er - li * ei + hr
        new_ei = lr * ei + li * er + hi
        br, bi = traces["E_B"][:, 0], traces["E_B"][:, 1]
        lr3, li3 = lr[:, :, None], li[:, :, None]
        new_br = lr3 * br - li3 * bi + x[:, None, :]
        new_bi = lr3 * bi + li3 * br
        new_hr = lr * hr - li * hi + mv(xp, P["B_re"], x)
        new_hi = lr * hi + li * hr + mv(xp, P["B_im"], x)
        return (
            xp.concatenate([new_hr, new_hi], axis=1),
            {"E_lam": xp.stack([new_er, new_ei], axis=1), "E_B": xp.stack([new_br, new_bi], axis=1)},
        )

    def readout_forward(self, xp: Any, P: Mapping[str, Array], h: Array, x: Array) -> tuple[Array, Array]:
        a = xp.tanh(mv(xp, P["W1"], h) + mv(xp, P["V"], x) + P["b1"])
        return mv(xp, P["W_o"], a) + P["b_o"], a

    def rtrl_gradients(
        self,
        xp: Any,
        P: Mapping[str, Array],
        x: Array,
        h: Array,
        a: Array,
        d_out: Array,
        traces: Mapping[str, Array],
    ) -> dict[str, Array]:
        """Exact gradient of this step's loss for every parameter (no truncation)."""

        H = self.hidden
        G: dict[str, Array] = {}
        G["W_o"] = outer(d_out, a)
        G["b_o"] = d_out
        d_z = mtv(xp, P["W_o"], d_out) * (1.0 - a * a)
        G["W1"] = outer(d_z, h)
        G["V"] = outer(d_z, x)
        G["b1"] = d_z
        d_f = mtv(xp, P["W1"], d_z)
        cr, ci = d_f[:, :H], d_f[:, H:]
        lr, li, modulus = self.lam(xp, P)
        er, ei = traces["E_lam"][:, 0], traces["E_lam"][:, 1]
        # q = c * E_lam with c = cr - i ci
        qr = cr * er + ci * ei
        qi = cr * ei - ci * er
        # q * lambda
        pr = qr * lr - qi * li
        pi = qr * li + qi * lr
        G["nu"] = -xp.exp(P["nu"]) * pr
        G["theta"] = -pi
        br, bi = traces["E_B"][:, 0], traces["E_B"][:, 1]
        cr3, ci3 = cr[:, :, None], ci[:, :, None]
        G["B_re"] = cr3 * br + ci3 * bi
        G["B_im"] = ci3 * br - cr3 * bi
        del modulus
        return G


CORES: dict[str, type[Core]] = {
    "gru": GRUCore,
    "elman": ElmanCore,
    "mgu": MGUCore,
    "mlp": MLPCore,
    "lru": LRUCore,
}


def build_core(kind: str, **options: Any) -> Core:
    if kind not in CORES:
        raise ValueError(f"unknown core kind {kind!r}; expected one of {sorted(CORES)}")
    return CORES[kind](**options)
