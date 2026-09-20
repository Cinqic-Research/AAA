"""A compact, read-only Matplotlib view of what the model is actually doing.

Five panels, all showing real state rather than decorative neurons:

* **Environment** -- true position, the pre-reveal prediction, and the
  reflected constant-motion baseline on the same axes.
* **Hidden state** -- all sixteen GRU activations, as they were.
* **Gates** -- the update and reset gate vectors from the same step.
* **Error** -- realized absolute error, the network's own predicted error
  magnitude, and a rolling mean.
* **Learning** -- gradient norm, parameter norm and update count.

Two rules. Recording and rendering are separate: :func:`record_trace` runs the
model and captures what it did, and :func:`render` takes that recorded trace
and draws it. ``render`` touches no model at all -- it is handed a plain
dataclass of numbers -- and a test asserts a model's complete state hash is
unchanged across a render. And an event marker appears only at the step where
the evaluator has already seen the event, never in advance.

This is a diagnostic, not a product. There is no GUI framework, no interactive
key binding (`PR #12` discovered that Matplotlib cheerfully binds the same keys
to unrelated actions), and no animation loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .agents import ConstantMotionAgent, NeuralAgent
from .features import PublicScales
from .model import AAA1KGRU
from .stats import rolling_mean
from .streams import Stream


@dataclass
class Trace:
    """A recorded, immutable view of one run. Nothing here can change a model."""

    steps: list[int] = field(default_factory=list)
    true_position: list[float] = field(default_factory=list)
    prediction: list[float] = field(default_factory=list)
    baseline: list[float] = field(default_factory=list)
    hidden: list[list[float]] = field(default_factory=list)
    update_gate: list[list[float]] = field(default_factory=list)
    reset_gate: list[list[float]] = field(default_factory=list)
    realized_error: list[float] = field(default_factory=list)
    predicted_error: list[float] = field(default_factory=list)
    gradient_norm: list[float] = field(default_factory=list)
    parameter_norm: list[float] = field(default_factory=list)
    update_count: list[int] = field(default_factory=list)
    observed: list[bool] = field(default_factory=list)
    events: list[tuple[int, str]] = field(default_factory=list)
    online: bool = True

    def __len__(self) -> int:
        return len(self.steps)


def record_trace(stream: Stream, model: AAA1KGRU | None = None, **options: Any) -> Trace:
    """Run one stream and capture the real state the panels display.

    The loop is written out here rather than delegated to
    :func:`research.aaa_1k.runner.run_stream` because the hidden state and gate
    vectors have to be read *between* the forward pass and the reveal, which is
    the only moment they describe the prediction being scored. The causal order
    is identical to the runner's, and a test asserts both produce the same
    predictions.
    """

    model = model or AAA1KGRU(seed=0, learning_rate=0.03, tbptt_steps=4)

    resolved = options.pop("scales", None) or PublicScales()
    online = bool(options.pop("online", True))
    max_steps = options.pop("max_steps", None)
    agent = NeuralAgent(model, name="aaa1k", scales=resolved, update_enabled=online)
    baseline = ConstantMotionAgent(name="constant_motion_reflected", scales=resolved, reflect=True)
    trace = Trace(online=online)
    stop = None if max_steps is None else min(max_steps + 1, len(stream.steps) - 1)

    agent.begin_episode()
    baseline.begin_episode()
    agent.accept_observation(stream.steps[0].true_position)
    baseline.accept_observation(stream.steps[0].true_position)
    last = len(stream.steps) - 1 if stop is None else stop
    for index in range(0, last):
        target = stream.steps[index + 1]
        prediction = agent.predict()
        baseline_prediction = baseline.predict()
        # The gate panels are specific to the gated core, so the dashboard
        # reads the model it was handed rather than the agent's generic slot.
        gates = model.last_gates()
        trace.steps.append(target.index)
        trace.true_position.append(target.true_position)
        trace.prediction.append(prediction)
        trace.baseline.append(baseline_prediction)
        trace.hidden.append([float(value) for value in model.hidden])
        trace.update_gate.append(gates["update_gate"])
        trace.reset_gate.append(gates["reset_gate"])
        trace.predicted_error.append(float(agent.error_estimate))
        trace.realized_error.append(abs(prediction - target.true_position))
        trace.observed.append(target.observed)
        if target.event:
            trace.events.append((target.index, target.event))
        revealed = target.true_position if target.observed else None
        agent.accept_observation(revealed)
        baseline.accept_observation(revealed)
        trace.gradient_norm.append(float(agent.last_update.get("gradient_norm", 0.0)))
        trace.parameter_norm.append(float(agent.model.diagnostics()["parameter_norm"]))
        trace.update_count.append(int(agent.model.update_count))
    return trace


def render(trace: Trace, path: str | Path, *, title: str = "AAA-1K") -> Path:
    """Draw the five panels to a file. Headless-safe and read-only."""

    import matplotlib

    matplotlib.use("Agg", force=False)
    from matplotlib.figure import Figure

    if len(trace) == 0:
        raise ValueError("cannot render an empty trace")

    figure = Figure(figsize=(13.0, 11.0), dpi=110)
    axes = figure.subplots(5, 1, sharex=False)
    steps = np.asarray(trace.steps, dtype=float)

    environment, hidden_axis, gate_axis, error_axis, learning_axis = axes

    environment.plot(steps, trace.true_position, color="#1f2933", linewidth=1.6, label="true position")
    environment.plot(steps, trace.prediction, color="#c0392b", linewidth=1.1, label="AAA-1K prediction")
    environment.plot(
        steps,
        trace.baseline,
        color="#2980b9",
        linewidth=0.9,
        linestyle="--",
        label="constant motion (reflected)",
    )
    unobserved = [step for step, seen in zip(trace.steps, trace.observed, strict=True) if not seen]
    if unobserved:
        for step in unobserved:
            environment.axvspan(step - 0.5, step + 0.5, color="#f0f0f0", zorder=0)
        environment.plot([], [], color="#f0f0f0", linewidth=8, label="target not shown to the agent")
    for step, _event in trace.events:
        environment.axvline(step, color="#8e44ad", alpha=0.35, linewidth=1.0)
    environment.set_ylabel("position")
    environment.set_title(f"{title} -- environment ({'online' if trace.online else 'frozen'})")
    environment.legend(loc="upper right", fontsize=7, ncol=2)

    hidden = np.asarray(trace.hidden, dtype=float).T
    image = hidden_axis.imshow(
        hidden,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
        extent=(steps[0], steps[-1], hidden.shape[0] - 0.5, -0.5),
    )
    hidden_axis.set_ylabel("hidden unit")
    hidden_axis.set_title("all 16 GRU hidden activations (real values)")
    figure.colorbar(image, ax=hidden_axis, pad=0.01, fraction=0.03)

    update = np.asarray(trace.update_gate, dtype=float)
    reset = np.asarray(trace.reset_gate, dtype=float)
    gate_axis.plot(steps, update.mean(axis=1), color="#16a085", linewidth=1.2, label="update gate, mean")
    gate_axis.fill_between(
        steps, update.min(axis=1), update.max(axis=1), color="#16a085", alpha=0.18, label="update gate, range"
    )
    gate_axis.plot(steps, reset.mean(axis=1), color="#d35400", linewidth=1.2, label="reset gate, mean")
    gate_axis.fill_between(
        steps, reset.min(axis=1), reset.max(axis=1), color="#d35400", alpha=0.18, label="reset gate, range"
    )
    gate_axis.set_ylim(-0.02, 1.02)
    gate_axis.set_ylabel("gate value")
    gate_axis.set_title("gates (z keeps the old state, r controls what the candidate may read)")
    gate_axis.legend(loc="upper right", fontsize=7, ncol=2)

    scales = PublicScales()
    predicted_position_units = np.asarray(trace.predicted_error, dtype=float) * scales.displacement_scale
    error_axis.plot(steps, trace.realized_error, color="#c0392b", linewidth=0.8, label="realized |error|")
    error_axis.plot(
        steps, predicted_position_units, color="#f39c12", linewidth=1.2, label="predicted error magnitude"
    )
    error_axis.plot(
        steps,
        rolling_mean(trace.realized_error, 20),
        color="#2c3e50",
        linewidth=1.3,
        label="rolling MAE (20)",
    )
    error_axis.set_yscale("log")
    error_axis.set_ylabel("error (position units)")
    error_axis.set_title("realized error against the network's own estimate of it")
    error_axis.legend(loc="upper right", fontsize=7, ncol=3)

    learning_axis.plot(steps, trace.gradient_norm, color="#27ae60", linewidth=0.9, label="gradient norm")
    learning_axis.plot(steps, trace.parameter_norm, color="#2980b9", linewidth=1.2, label="parameter norm")
    learning_axis.set_ylabel("norm")
    learning_axis.set_xlabel("step")
    twin = learning_axis.twinx()
    twin.plot(steps, trace.update_count, color="#7f8c8d", linewidth=1.0, linestyle=":", label="update count")
    twin.set_ylabel("updates")
    handles, labels = learning_axis.get_legend_handles_labels()
    extra_handles, extra_labels = twin.get_legend_handles_labels()
    learning_axis.legend(handles + extra_handles, labels + extra_labels, loc="upper left", fontsize=7)
    learning_axis.set_title(
        f"learning ({'weights updating' if trace.online else 'weights frozen, recurrence still running'})"
    )

    figure.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination)
    return destination
