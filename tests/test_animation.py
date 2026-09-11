"""Animation temporal semantics, exercised headlessly.

The original animation duplicated a subtly different temporal algorithm inside
a Matplotlib callback, crashed on ``event.key = None``, and could only load the
legacy linear checkpoint. The semantics now live in a plain object so they can
be tested without a display.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from aaa.animation import AnimationSession, load_checkpoint_model
from aaa.config import ExperimentConfig
from aaa.predictors import OnlineLinearPredictor, OnlineRLSPredictor

CONFIG = ExperimentConfig().quick()


def session(**overrides) -> AnimationSession:
    values = dict(scenario="changed", seed=201, config=CONFIG, online=True)
    values.update(overrides)
    return AnimationSession(**values)


class CheckpointDispatchTests(unittest.TestCase):
    def test_an_rls_checkpoint_loads_as_an_rls_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rls.json"
            OnlineRLSPredictor(displacement_scale=0.01).save(path)
            model = load_checkpoint_model(path, online=True, name="model")
        self.assertIsInstance(model, OnlineRLSPredictor)
        self.assertTrue(model.update_enabled)

    def test_a_legacy_checkpoint_still_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            OnlineLinearPredictor().save(path)
            model = load_checkpoint_model(path, online=False, name="model")
        self.assertIsInstance(model, OnlineLinearPredictor)
        self.assertFalse(model.update_enabled)

    def test_an_unknown_checkpoint_format_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "alien.json"
            path.write_text(json.dumps({"format_version": "alien.v9"}), encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                load_checkpoint_model(path, online=True)
            self.assertIn("unsupported checkpoint format", str(caught.exception))

    def test_the_default_model_is_the_current_candidate_architecture(self):
        self.assertIsInstance(session().model, OnlineRLSPredictor)


class KeyHandlingTests(unittest.TestCase):
    def test_a_none_key_is_ignored_rather_than_crashing(self):
        self.assertEqual(session().handle_key(None), "ignored")

    def test_space_toggles_pause_and_resume(self):
        instance = session()
        self.assertEqual(instance.handle_key(" "), "paused")
        self.assertTrue(instance.paused)
        self.assertEqual(instance.handle_key(" "), "resumed")
        self.assertFalse(instance.paused)

    def test_r_restarts_in_either_case(self):
        self.assertEqual(session().handle_key("r"), "restarted")
        self.assertEqual(session().handle_key("R"), "restarted")

    def test_an_unrelated_key_is_ignored(self):
        self.assertEqual(session().handle_key("q"), "ignored")


class PauseTests(unittest.TestCase):
    def test_pause_stops_the_simulation_state_not_only_the_drawing(self):
        instance = session()
        for _ in range(6):
            instance.step()
        before = (instance.step_index, instance.update_count, list(instance.history))
        instance.handle_key(" ")
        for _ in range(5):
            self.assertIsNone(instance.step())
        after = (instance.step_index, instance.update_count, list(instance.history))
        self.assertEqual(before, after)

    def test_resume_continues_from_where_it_paused(self):
        instance = session()
        for _ in range(6):
            instance.step()
        instance.handle_key(" ")
        instance.step()
        instance.handle_key(" ")
        frame = instance.step()
        self.assertIsNotNone(frame)
        self.assertEqual(frame.step, 7)


class RestartTests(unittest.TestCase):
    def test_restart_restores_every_piece_of_state(self):
        instance = session()
        first = [instance.step() for _ in range(8)]
        instance.handle_key("r")
        self.assertEqual(instance.step_index, 0)
        self.assertEqual(instance.update_count, 0)
        self.assertIsNone(instance.prediction)
        self.assertFalse(instance.paused)
        self.assertEqual(len(instance.history), 1)
        second = [instance.step() for _ in range(8)]
        self.assertEqual(
            [frame.actual for frame in first], [frame.actual for frame in second]
        )

    def test_restart_restores_the_model_parameters(self):
        instance = session()
        initial = json.dumps(instance.model.state_dict(), sort_keys=True)
        for _ in range(10):
            instance.step()
        self.assertNotEqual(json.dumps(instance.model.state_dict(), sort_keys=True), initial)
        instance.handle_key("r")
        self.assertEqual(json.dumps(instance.model.state_dict(), sort_keys=True), initial)

    def test_restart_clears_a_pause(self):
        instance = session()
        instance.handle_key(" ")
        instance.handle_key("r")
        self.assertFalse(instance.paused)
        self.assertIsNotNone(instance.step())


class UpdateModeTests(unittest.TestCase):
    def test_online_mode_actually_updates_after_scoring(self):
        instance = session(online=True)
        for _ in range(10):
            instance.step()
        self.assertGreater(instance.update_count, 0)
        self.assertGreater(instance.model.update_count, 0)

    def test_frozen_mode_never_updates(self):
        instance = session(online=False)
        before = json.dumps(instance.model.state_dict(), sort_keys=True)
        frames = [instance.step() for _ in range(12)]
        self.assertEqual(instance.update_count, 0)
        self.assertFalse(any(frame.updated for frame in frames))
        self.assertEqual(json.dumps(instance.model.state_dict(), sort_keys=True), before)

    def test_the_first_frames_have_no_previous_prediction(self):
        instance = session()
        first = instance.step()
        self.assertIsNone(first.previous_prediction)

    def test_a_prediction_uses_only_the_declared_history_window(self):
        length = CONFIG.world.history_length
        instance = session()
        seen = []

        original = instance.model.predict

        def spy(history):
            seen.append(tuple(history))
            return original(history)

        instance.model.predict = spy  # type: ignore[method-assign]
        for _ in range(10):
            instance.step()
        self.assertTrue(seen)
        self.assertTrue(all(len(window) == length for window in seen))

    def test_the_update_window_matches_the_prediction_window(self):
        instance = session()
        predicted: list[tuple[float, ...]] = []
        updated: list[tuple[float, ...]] = []
        original_predict = instance.model.predict
        original_update = instance.model.update

        def predict_spy(history):
            predicted.append(tuple(history))
            return original_predict(history)

        def update_spy(history, target):
            updated.append(tuple(history))
            return original_update(history, target)

        instance.model.predict = predict_spy  # type: ignore[method-assign]
        instance.model.update = update_spy  # type: ignore[method-assign]
        for _ in range(10):
            instance.step()
        self.assertEqual(predicted, updated)


class HorizonTests(unittest.TestCase):
    def test_the_session_finishes_at_the_declared_horizon(self):
        instance = session()
        horizon = CONFIG.world.steps_per_episode
        for _ in range(horizon):
            instance.step()
        self.assertTrue(instance.finished)

    def test_stepping_past_the_end_is_safe_and_marked_finished(self):
        instance = session()
        for _ in range(CONFIG.world.steps_per_episode + 5):
            frame = instance.step()
        self.assertTrue(frame.finished)
        self.assertEqual(frame.event, "end")

    def test_event_labels_are_reported_for_the_renderer_only(self):
        instance = session(scenario="changed")
        events = {frame.event for frame in (instance.step() for _ in range(CONFIG.world.steps_per_episode))}
        self.assertIn("steady", events)
        self.assertIn("change", events)


class ScenarioTests(unittest.TestCase):
    def test_every_supported_scenario_runs(self):
        for scenario in ("straight", "bouncing", "changed"):
            config = CONFIG if scenario != "straight" else replace(
                CONFIG, world=replace(CONFIG.world, steps_per_episode=30, change_step=None)
            )
            instance = AnimationSession(scenario=scenario, seed=7, config=config, online=True)
            frames = [instance.step() for _ in range(10)]
            self.assertTrue(all(frame is not None for frame in frames))


if __name__ == "__main__":
    unittest.main()
