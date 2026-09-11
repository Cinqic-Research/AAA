"""Optional interactive moving-dot demonstration.

The temporal semantics live in :class:`AnimationSession`, which is a plain
object with no GUI dependency, so pause, resume, restart, frozen immutability,
online updating and end-of-horizon behaviour are all testable headlessly. The
GUI wrapper only draws whatever the session reports.

The session follows exactly the same ordering as :mod:`aaa.experiment`:
predict from the causally available history, advance, reveal, score, and only
then update an enabled model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .environment import MovingDotEnvironment, Scenario
from .predictors import OnlineLinearPredictor, OnlineRLSPredictor, Predictor

CHECKPOINT_LOADERS = {
    OnlineRLSPredictor.format_version: OnlineRLSPredictor,
    OnlineLinearPredictor.format_version: OnlineLinearPredictor,
}


def load_checkpoint_model(path: str | Path, *, online: bool, name: str = "model") -> Predictor:
    """Dispatch on the checkpoint's declared format version."""

    state = json.loads(Path(path).read_text(encoding="utf-8"))
    version = state.get("format_version")
    loader = CHECKPOINT_LOADERS.get(str(version))
    if loader is None:
        raise ValueError(
            f"unsupported checkpoint format {version!r}; expected one of {sorted(CHECKPOINT_LOADERS)}"
        )
    return loader.from_state_dict(state, name=name, update_enabled=online)  # type: ignore[attr-defined]


@dataclass
class Frame:
    """What the renderer needs for one displayed step."""

    step: int
    actual: float
    previous_prediction: float | None
    event: str
    finished: bool
    updated: bool


@dataclass
class AnimationSession:
    """Headless simulation state behind the animation."""

    scenario: Scenario = "changed"
    seed: int = 201
    config: ExperimentConfig = field(default_factory=ExperimentConfig)
    online: bool = True
    checkpoint: str | Path | None = None

    def __post_init__(self) -> None:
        self.environment = MovingDotEnvironment(
            scenario=self.scenario, seed=self.seed, config=self.config.world
        )
        self._initial_state: dict[str, Any] = self._build_model().state_dict()  # type: ignore[union-attr]
        self.paused = False
        self.history: list[float] = []
        self.prediction: float | None = None
        self.step_index = 0
        self.update_count = 0
        self.model: Predictor
        self.restart()

    # -- construction ----------------------------------------------------
    def _build_model(self) -> Predictor:
        if self.checkpoint is not None:
            return load_checkpoint_model(self.checkpoint, online=self.online, name="model")
        return OnlineRLSPredictor(
            lower_bound=self.config.world.lower_bound,
            upper_bound=self.config.world.upper_bound,
            displacement_scale=self.config.world.dt * self.config.world.speed_max,
            name="model",
            update_enabled=self.online,
        )

    def _restore_model(self) -> Predictor:
        version = str(self._initial_state.get("format_version"))
        loader = CHECKPOINT_LOADERS[version]
        return loader.from_state_dict(self._initial_state, name="model", update_enabled=self.online)  # type: ignore[attr-defined]

    # -- controls --------------------------------------------------------
    def restart(self) -> None:
        """Restore the environment, RNG, history, model and every counter."""

        self.model = self._restore_model()
        self.history = [self.environment.reset()]
        self.prediction = None
        self.step_index = 0
        self.update_count = 0
        self.paused = False

    def toggle_pause(self) -> bool:
        self.paused = not self.paused
        return self.paused

    def handle_key(self, key: str | None) -> str:
        """Handle a key press. ``None`` is a valid Matplotlib key event."""

        if key is None:
            return "ignored"
        lowered = key.lower()
        if lowered == " ":
            return "paused" if self.toggle_pause() else "resumed"
        if lowered == "r":
            self.restart()
            return "restarted"
        return "ignored"

    # -- simulation ------------------------------------------------------
    @property
    def finished(self) -> bool:
        return self.step_index >= self.config.world.steps_per_episode

    def step(self) -> Frame | None:
        """Advance one transition, or return ``None`` while paused/finished.

        While paused the simulation state does not advance at all: no
        environment step, no prediction, no update.
        """

        if self.paused:
            return None
        if self.finished:
            return Frame(self.step_index, self.history[-1], self.prediction, "end", True, False)
        length = self.config.world.history_length
        if len(self.history) >= length:
            self.prediction = float(self.model.predict(tuple(self.history[-length:])))
        window = tuple(self.history[-length:]) if len(self.history) >= length else None
        previous = self.prediction
        transition = self.environment.advance()
        self.step_index += 1
        updated = False
        if window is not None and self.online and self.model.update_enabled:
            # Scoring has happened (the frame carries the error the renderer
            # shows); only now may the model learn.
            self.model.update(window, transition.position)
            self.update_count += 1
            updated = True
        self.history.append(transition.position)
        event = "change" if transition.changed else "bounce" if transition.bounced else "steady"
        return Frame(self.step_index, transition.position, previous, event, self.finished, updated)


def launch_animation(
    *,
    checkpoint: str | Path | None = None,
    scenario: Scenario = "changed",
    seed: int = 201,
    config: ExperimentConfig | None = None,
    online: bool = True,
) -> None:  # pragma: no cover - requires an interactive display
    """Show actual positions and the previously issued prediction.

    Space pauses/resumes and ``r`` restarts. The animation is a visual aid; the
    retained step logs remain the reproducibility source.
    """

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    session = AnimationSession(
        scenario=scenario, seed=seed, config=config or ExperimentConfig(), online=online, checkpoint=checkpoint
    )
    world = session.config.world
    figure, axis = plt.subplots(figsize=(9, 3.2))
    axis.set_xlim(world.lower_bound, world.upper_bound)
    axis.set_ylim(-0.25, 0.25)
    axis.set_yticks([])
    axis.set_xlabel("position")
    mode = "online" if online else "frozen"
    axis.set_title(f"AAA: {mode} mode — actual dot and previously issued next-position prediction")
    (actual_artist,) = axis.plot([], [], "o", color="black", markersize=11, label="actual")
    (predicted_artist,) = axis.plot([], [], "x", color="crimson", markersize=10, mew=2, label="previous prediction")
    text_artist = axis.text(0.02, 0.9, "", transform=axis.transAxes, fontsize=9)
    axis.legend(loc="upper right", fontsize=8)

    def draw(_frame: int):
        frame = session.step()
        if frame is None:
            text_artist.set_text(f"{mode}; paused at step {session.step_index}")
            return actual_artist, predicted_artist, text_artist
        actual_artist.set_data([frame.actual], [0])
        if frame.previous_prediction is None:
            predicted_artist.set_data([], [])
        else:
            predicted_artist.set_data([frame.previous_prediction], [0])
        text_artist.set_text(
            f"{mode}; target step {frame.step}; {frame.event}; updates {session.update_count}"
        )
        return actual_artist, predicted_artist, text_artist

    def on_key(event) -> None:
        session.handle_key(getattr(event, "key", None))

    figure.canvas.mpl_connect("key_press_event", on_key)
    FuncAnimation(figure, draw, frames=world.steps_per_episode + 1, interval=100, blit=True, repeat=False)
    plt.show()
