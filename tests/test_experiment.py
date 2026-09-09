import unittest

import numpy as np

from aaa.config import WorldConfig
from aaa.environment import MovingDotEnvironment
from aaa.experiment import run_episode
from aaa.predictors import OnlineLinearPredictor, PersistencePredictor, Predictor


class RecordingEnvironment(MovingDotEnvironment):
    def __init__(self, *args, events, **kwargs):
        self.events = events
        super().__init__(*args, **kwargs)

    def advance(self):
        self.events.append("advance")
        return super().advance()


class TracingPredictor(Predictor):
    update_enabled = True

    def __init__(self, name, events):
        self.name = name
        self.events = events
        self.histories = []

    def predict(self, history):
        self.histories.append(tuple(history))
        self.events.append(f"predict:{self.name}")
        return history[-1]

    def update(self, history, target_position):
        self.events.append(f"update:{self.name}")


class ExperimentTests(unittest.TestCase):
    def test_prediction_score_update_order(self):
        events = []
        config = WorldConfig(steps_per_episode=5)
        environment = RecordingEnvironment("straight", 1, config, events=events)
        first = TracingPredictor("first", events)
        second = TracingPredictor("second", events)
        run_episode(
            environment,
            [first, second],
            learn=True,
            score_hook=lambda record: events.append("score"),
        )
        self.assertEqual(
            events[3:9],
            ["predict:first", "predict:second", "advance", "score", "update:first", "update:second"],
        )
        self.assertEqual(len(first.histories), 2)
        self.assertEqual(first.histories[0][1:], first.histories[1][:-1])

    def test_learning_only_changes_enabled_model(self):
        config = WorldConfig(steps_per_episode=8, change_step=4)
        frozen = OnlineLinearPredictor(learning_rate=0.1, update_enabled=False)
        frozen_before = frozen.weights.copy()
        run_episode(MovingDotEnvironment("straight", 4, config), [frozen], learn=True)
        np.testing.assert_array_equal(frozen.weights, frozen_before)

        online = OnlineLinearPredictor(learning_rate=0.1, update_enabled=True)
        online_before = online.weights.copy()
        run_episode(MovingDotEnvironment("straight", 4, config), [online], learn=True)
        self.assertFalse(np.array_equal(online.weights, online_before))

    def test_metadata_is_not_passed_to_predictor(self):
        config = WorldConfig(steps_per_episode=8, change_step=4)
        probe = TracingPredictor("probe", [])
        records = run_episode(
            MovingDotEnvironment("changed", 5, config),
            [probe],
            learn=False,
        )
        self.assertTrue(records)
        self.assertTrue(all(len(history) == 4 for history in probe.histories))
        self.assertFalse(hasattr(probe, "scenario"))
        self.assertFalse(hasattr(probe, "velocity"))

    def test_predictor_comparisons_share_realized_trajectory(self):
        config = WorldConfig(steps_per_episode=20)
        first = run_episode(
            MovingDotEnvironment("bouncing", 8, config),
            [PersistencePredictor()],
            learn=False,
        )
        second = run_episode(
            MovingDotEnvironment("bouncing", 8, config),
            [OnlineLinearPredictor()],
            learn=False,
        )
        self.assertEqual(
            [record.actual_next_position for record in first],
            [record.actual_next_position for record in second],
        )


if __name__ == "__main__":
    unittest.main()
