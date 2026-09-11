"""The strict temporal boundary and the step-record schema."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aaa.config import WorldConfig
from aaa.environment import MovingDotEnvironment
from aaa.experiment import (
    STEP_RECORD_SCHEMA,
    StepRecord,
    run_episode,
    write_step_records,
)
from aaa.predictors import OnlineLinearPredictor, PersistencePredictor, Predictor

from .helpers import identity


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
        self.update_histories = []

    def predict(self, history):
        self.histories.append(tuple(history))
        self.events.append(f"predict:{self.name}")
        return history[-1]

    def update(self, history, target_position):
        self.update_histories.append((tuple(history), target_position))
        self.events.append(f"update:{self.name}")


class TemporalOrderTests(unittest.TestCase):
    def test_prediction_score_update_order(self):
        events = []
        config = WorldConfig(steps_per_episode=5)
        environment = RecordingEnvironment("straight", 1, config, events=events)
        first = TracingPredictor("first", events)
        second = TracingPredictor("second", events)
        run_episode(
            environment,
            [first, second],
            identity(family="straight", scenario="straight"),
            learn=True,
            score_hook=lambda record: events.append("score"),
        )
        self.assertEqual(
            events[3:9],
            ["predict:first", "predict:second", "advance", "score", "update:first", "update:second"],
        )
        self.assertEqual(len(first.histories), 2)
        self.assertEqual(first.histories[0][1:], first.histories[1][:-1])

    def test_update_receives_the_same_window_that_was_predicted_from(self):
        config = WorldConfig(steps_per_episode=8)
        probe = TracingPredictor("probe", [])
        run_episode(
            MovingDotEnvironment("straight", 2, config),
            [probe],
            identity(family="straight", scenario="straight", update_mode="online"),
            learn=True,
        )
        self.assertTrue(probe.update_histories)
        for predicted, (updated, _) in zip(probe.histories, probe.update_histories, strict=True):
            self.assertEqual(predicted, updated)

    def test_update_target_is_the_revealed_next_position(self):
        config = WorldConfig(steps_per_episode=8)
        probe = TracingPredictor("probe", [])
        records = run_episode(
            MovingDotEnvironment("straight", 2, config),
            [probe],
            identity(family="straight", scenario="straight", update_mode="online"),
            learn=True,
        )
        for record, (_, target) in zip(records, probe.update_histories, strict=True):
            self.assertEqual(record.actual_next_position, target)

    def test_learning_only_changes_enabled_model(self):
        config = WorldConfig(steps_per_episode=8, change_step=4)
        frozen = OnlineLinearPredictor(learning_rate=0.1, update_enabled=False)
        frozen_before = frozen.weights.copy()
        run_episode(
            MovingDotEnvironment("straight", 4, config),
            [frozen],
            identity(family="straight", scenario="straight"),
            learn=True,
        )
        np.testing.assert_array_equal(frozen.weights, frozen_before)

        online = OnlineLinearPredictor(learning_rate=0.1, update_enabled=True)
        online_before = online.weights.copy()
        run_episode(
            MovingDotEnvironment("straight", 4, config),
            [online],
            identity(family="straight", scenario="straight", update_mode="online"),
            learn=True,
        )
        self.assertFalse(np.array_equal(online.weights, online_before))

    def test_learn_false_disables_every_update(self):
        config = WorldConfig(steps_per_episode=8)
        online = OnlineLinearPredictor(learning_rate=0.1, update_enabled=True)
        before = online.weights.copy()
        records = run_episode(
            MovingDotEnvironment("straight", 4, config),
            [online],
            identity(family="straight", scenario="straight"),
            learn=False,
        )
        np.testing.assert_array_equal(online.weights, before)
        self.assertTrue(all(record.updates_enabled["linear_online"] is False for record in records))


class LeakageTests(unittest.TestCase):
    def test_metadata_is_not_passed_to_predictor(self):
        config = WorldConfig(steps_per_episode=8, change_step=4)
        probe = TracingPredictor("probe", [])
        records = run_episode(
            MovingDotEnvironment("changed", 5, config),
            [probe],
            identity(family="speed_change", scenario="changed"),
            learn=False,
        )
        self.assertTrue(records)
        self.assertTrue(all(len(history) == 4 for history in probe.histories))
        self.assertFalse(hasattr(probe, "scenario"))
        self.assertFalse(hasattr(probe, "velocity"))

    def test_predictor_inputs_contain_only_past_observations(self):
        config = WorldConfig(steps_per_episode=12, change_step=6)
        probe = TracingPredictor("probe", [])
        records = run_episode(
            MovingDotEnvironment("changed", 11, config),
            [probe],
            identity(family="speed_change", scenario="changed"),
            learn=False,
        )
        for record, history in zip(records, probe.histories, strict=True):
            self.assertEqual(record.history, history)
            self.assertNotIn(record.actual_next_position, history)

    def test_predictor_comparisons_share_realized_trajectory(self):
        config = WorldConfig(steps_per_episode=20)
        first = run_episode(
            MovingDotEnvironment("bouncing", 8, config),
            [PersistencePredictor()],
            identity(family="bouncing", scenario="bouncing"),
            learn=False,
        )
        second = run_episode(
            MovingDotEnvironment("bouncing", 8, config),
            [OnlineLinearPredictor()],
            identity(family="bouncing", scenario="bouncing"),
            learn=False,
        )
        self.assertEqual(
            [record.actual_next_position for record in first],
            [record.actual_next_position for record in second],
        )

    def test_duplicate_predictor_names_are_rejected(self):
        config = WorldConfig(steps_per_episode=6)
        with self.assertRaises(ValueError):
            run_episode(
                MovingDotEnvironment("straight", 1, config),
                [PersistencePredictor(), PersistencePredictor()],
                identity(family="straight", scenario="straight"),
            )


class RecordSchemaTests(unittest.TestCase):
    def _records(self):
        config = WorldConfig(steps_per_episode=10, change_step=5)
        return run_episode(
            MovingDotEnvironment("changed", 3, config),
            [PersistencePredictor()],
            identity(
                trial_id="fam:r00:e000",
                family="speed_change",
                scenario="changed",
                environment_seed=3,
                replica_id=2,
                episode=7,
                stratum="positive-position1-speed2",
                confirmation_batch="batch-x",
                checkpoint_hash="abc123",
                update_mode="online",
            ),
            learn=False,
        )

    def test_identity_fields_are_distinct_and_preserved(self):
        record = self._records()[0]
        value = record.to_dict()
        self.assertEqual(value["schema_version"], STEP_RECORD_SCHEMA)
        self.assertEqual(value["replica_id"], 2)
        self.assertEqual(value["episode"], 7)
        self.assertEqual(value["environment_seed"], 3)
        self.assertEqual(value["confirmation_batch"], "batch-x")
        self.assertEqual(value["checkpoint_hash"], "abc123")
        self.assertEqual(value["update_mode"], "online")
        self.assertNotEqual(value["replica_id"], value["environment_seed"])

    def test_round_trip_through_dict_is_lossless(self):
        for record in self._records():
            restored = StepRecord.from_dict(json.loads(json.dumps(record.to_dict())))
            self.assertEqual(restored.to_dict(), record.to_dict())

    def test_an_unknown_schema_version_is_rejected(self):
        value = self._records()[0].to_dict()
        value["schema_version"] = "aaa.step_record.v1"
        with self.assertRaises(ValueError):
            StepRecord.from_dict(value)

    def test_invalid_identity_values_are_rejected(self):
        with self.assertRaises(ValueError):
            identity(update_mode="whenever")
        with self.assertRaises(ValueError):
            identity(replica_id=-1)
        with self.assertRaises(ValueError):
            identity(episode=True)

    def test_written_records_reload_identically(self):
        records = self._records()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "steps.jsonl"
            write_step_records(records, path, Path(directory) / "steps.csv")
            reloaded = [
                StepRecord.from_dict(json.loads(line))
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual([r.to_dict() for r in reloaded], [r.to_dict() for r in records])


class WarmUpTests(unittest.TestCase):
    def test_warm_up_transitions_are_not_scored(self):
        config = WorldConfig(steps_per_episode=10, history_length=4)
        records = run_episode(
            MovingDotEnvironment("straight", 1, config),
            [PersistencePredictor()],
            identity(family="straight", scenario="straight"),
        )
        self.assertEqual(len(records), config.steps_per_episode - (config.history_length - 1))
        self.assertEqual(records[0].step, config.history_length - 1)

    def test_an_episode_shorter_than_the_warm_up_scores_nothing(self):
        config = WorldConfig(steps_per_episode=2, history_length=4)
        records = run_episode(
            MovingDotEnvironment("straight", 1, config),
            [PersistencePredictor()],
            identity(family="straight", scenario="straight"),
        )
        self.assertEqual(records, [])

    def test_continuation_requires_a_history(self):
        from aaa.experiment import continue_episode

        config = WorldConfig(steps_per_episode=5)
        with self.assertRaises(ValueError):
            continue_episode(
                MovingDotEnvironment("straight", 1, config),
                [PersistencePredictor()],
                [],
                identity(family="straight", scenario="straight"),
            )


if __name__ == "__main__":
    unittest.main()
