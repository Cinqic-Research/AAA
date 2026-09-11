"""Headless rendering, arbitrary bounds, and no pyplot import.

The original module imported ``matplotlib.pyplot`` at module scope and then
tried to switch the backend afterwards, and hardcoded ``[0, 1]`` y-limits.
"""

from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from aaa import visualization
from aaa.config import ExperimentConfig, WorldConfig
from aaa.environment import MovingDotEnvironment
from aaa.experiment import run_episode
from aaa.predictors import ConstantMotionPredictor, PersistencePredictor

from .helpers import identity


def png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssertionError(f"{path} is not a PNG")
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def make_records(config: ExperimentConfig, scenario: str = "changed", episodes: int = 2):
    records = []
    for episode in range(episodes):
        records.extend(
            run_episode(
                MovingDotEnvironment(scenario, 11 + episode, config.world),
                [PersistencePredictor(), ConstantMotionPredictor()],
                identity(family=scenario, scenario=scenario, episode=episode, replica_id=episode % 2),
                learn=False,
            )
        )
    return records


class BackendTests(unittest.TestCase):
    def test_pyplot_is_never_imported_by_the_plotting_module(self):
        source = Path(visualization.__file__).read_text(encoding="utf-8")
        code = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
        self.assertFalse([line for line in code if "pyplot" in line], code)
        self.assertNotIn("matplotlib.use(", source)
        self.assertNotIn("matplotlib.pyplot", sys.modules.get("aaa.visualization").__dict__)

    def test_rendering_works_without_a_display(self):
        config = ExperimentConfig().quick()
        with tempfile.TemporaryDirectory() as directory:
            paths = visualization.write_all_plots(
                Path(directory),
                learning_records=make_records(config),
                frozen_records=make_records(config, "bouncing"),
                adaptation_records=make_records(config),
                config=config,
            )
            self.assertEqual(len(paths), 4)
            for path in paths.values():
                self.assertGreater(png_size(Path(path))[0], 0)


class ArbitraryBoundsTests(unittest.TestCase):
    def test_plots_render_for_a_shifted_wide_interval(self):
        world = WorldConfig(
            lower_bound=-5.0,
            upper_bound=5.0,
            steps_per_episode=70,
            change_step=35,
            speed_min=0.5,
            speed_max=1.2,
            event_margin=1.0,
        )
        config = replace(ExperimentConfig().quick(), world=world)
        with tempfile.TemporaryDirectory() as directory:
            paths = visualization.write_all_plots(
                Path(directory),
                learning_records=make_records(config),
                frozen_records=make_records(config, "bouncing"),
                adaptation_records=make_records(config),
                config=config,
            )
            for path in paths.values():
                self.assertTrue(Path(path).exists())

    def test_axis_limits_come_from_the_configuration(self):
        world = WorldConfig(lower_bound=10.0, upper_bound=30.0, steps_per_episode=70, change_step=35)
        config = replace(ExperimentConfig().quick(), world=world)
        self.assertEqual(visualization._bounds(config), (10.0, 30.0))

    def test_empty_records_are_a_loud_error(self):
        config = ExperimentConfig().quick()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                visualization.plot_representative_predictions([], config, Path(directory) / "x.png")


class BenchmarkPlotTests(unittest.TestCase):
    def summary(self) -> dict:
        return {
            "learning_curve": {
                "budgets": [0, 2, 4],
                "curve": {
                    "0": {"budget": 0, "mean": 1e-2, "replica_episode_values": {"0": [1e-2], "1": [1.1e-2]}},
                    "2": {"budget": 2, "mean": 1e-5, "replica_episode_values": {"0": [1e-5], "1": [1.1e-5]}},
                    "4": {"budget": 4, "mean": 1e-9, "replica_episode_values": {"0": [1e-9], "1": [1.1e-9]}},
                },
            },
            "results": {
                "constant_velocity": {
                    "predictors": {
                        "candidate_frozen": {
                            "strata": {
                                "positive-position0-speed0": {"mae": 1e-9, "p95": 2e-9, "count": 10},
                                "negative-position1-speed2": {"mae": 2e-9, "p95": 3e-9, "count": 12},
                            }
                        }
                    }
                },
                "changed_law:changed": {
                    "predictors": {
                        "online": {"replica_mae": {"0": 0.0015, "1": 0.0013}},
                        "frozen": {"replica_mae": {"0": 0.015, "1": 0.013}},
                    }
                },
            },
            "gates": {
                "gates": [
                    {
                        "name": "recovery",
                        "details": {
                            "status_counts": {"recovered": 40, "unrecovered": 5, "no_measured_shock": 3}
                        },
                    }
                ]
            },
            "checkpoints": {
                "diagnostics": {
                    "0": {"condition_number": 120.0, "min_eigenvalue": 1e-3, "trace": 4.0},
                    "1": {"condition_number": 140.0, "min_eigenvalue": 9e-4, "trace": 4.2},
                }
            },
        }

    def test_all_benchmark_plots_render(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = visualization.write_benchmark_plots(Path(directory), self.summary())
            for name in (
                "learning_curve",
                "stratum_performance",
                "recovery_distribution",
                "paired_scatter",
                "covariance_diagnostics",
            ):
                self.assertIn(name, paths)
            for path in paths.values():
                self.assertGreater(png_size(Path(path))[0], 0)

    def test_missing_optional_sections_are_skipped_not_fatal(self):
        summary = self.summary()
        del summary["gates"]
        del summary["checkpoints"]
        with tempfile.TemporaryDirectory() as directory:
            paths = visualization.write_benchmark_plots(Path(directory), summary)
        self.assertNotIn("recovery_distribution", paths)
        self.assertNotIn("covariance_diagnostics", paths)
        self.assertIn("learning_curve", paths)


if __name__ == "__main__":
    unittest.main()
