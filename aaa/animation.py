"""Optional interactive moving-dot demonstration."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from .config import ExperimentConfig
from .environment import MovingDotEnvironment, Scenario
from .predictors import OnlineLinearPredictor


def launch_animation(
    *,
    checkpoint: str | Path | None = None,
    scenario: Scenario = "changed",
    seed: int = 201,
    config: ExperimentConfig | None = None,
) -> None:
    """Show actual positions and the previously issued prediction.

    Space pauses/resumes and ``r`` restarts the animation. The animation is a
    visual aid only; the JSONL evaluation remains the reproducibility source.
    """

    config = config or ExperimentConfig()
    environment = MovingDotEnvironment(scenario=scenario, seed=seed, config=config.world)
    model = (
        OnlineLinearPredictor.load(checkpoint, name="linear_online", update_enabled=False)
        if checkpoint
        else OnlineLinearPredictor(learning_rate=config.learning_rate, name="linear_online", update_enabled=False)
    )
    state: dict[str, object] = {"history": [], "prediction": None, "paused": False}
    figure, axis = plt.subplots(figsize=(9, 3.2))
    axis.set_xlim(config.world.lower_bound, config.world.upper_bound)
    axis.set_ylim(-0.25, 0.25)
    axis.set_yticks([])
    axis.set_xlabel("position")
    axis.set_title("AAA: actual dot and previously issued next-position prediction")
    actual_artist, = axis.plot([], [], "o", color="black", markersize=11, label="actual")
    predicted_artist, = axis.plot([], [], "x", color="crimson", markersize=10, mew=2, label="previous prediction")
    text_artist = axis.text(0.02, 0.9, "", transform=axis.transAxes, fontsize=9)
    axis.legend(loc="upper right", fontsize=8)

    def reset() -> None:
        state["history"] = [environment.reset()]
        state["prediction"] = None
        actual_artist.set_data([environment.observe()], [0])
        predicted_artist.set_data([], [])
        text_artist.set_text("warm-up")

    def update(frame: int):
        if state["paused"]:
            return actual_artist, predicted_artist, text_artist
        history = state["history"]
        assert isinstance(history, list)
        if frame == 0:
            reset()
            return actual_artist, predicted_artist, text_artist
        if len(history) >= config.world.history_length:
            state["prediction"] = model.predict(tuple(history[-config.world.history_length :]))
        transition = environment.advance()
        previous_prediction = state["prediction"]
        history.append(transition.position)
        actual_artist.set_data([transition.position], [0])
        if previous_prediction is None:
            predicted_artist.set_data([], [])
        else:
            predicted_artist.set_data([previous_prediction], [0])
        event = "change" if transition.changed else "bounce" if transition.bounced else ""
        text_artist.set_text(f"target step {transition.step_index + 1}; {event or 'steady'}")
        return actual_artist, predicted_artist, text_artist

    def on_key(event):
        if event.key == " ":
            state["paused"] = not state["paused"]
        elif event.key.lower() == "r":
            reset()
            animation.frame_seq = animation.new_frame_seq()
            animation.event_source.start()

    figure.canvas.mpl_connect("key_press_event", on_key)
    animation = FuncAnimation(
        figure,
        update,
        frames=config.world.steps_per_episode + 1,
        interval=100,
        blit=True,
        repeat=False,
    )
    reset()
    plt.show()
