"""Environment correctness: determinism, boundaries, and event accounting."""

import unittest

from aaa.config import WorldConfig
from aaa.environment import (
    DampedOscillatorEnvironment,
    MovingDotEnvironment,
    resolve_reflection,
)


class ReflectionTests(unittest.TestCase):
    def test_single_overshoot_reflects_once(self):
        result = resolve_reflection(1.4, 1.0, 0.0, 1.0)
        self.assertAlmostEqual(result.position, 0.6, places=12)
        self.assertEqual(result.walls, ("upper",))
        self.assertEqual(result.bounce_count, 1)
        self.assertLess(result.velocity, 0.0)

    def test_lower_overshoot_reflects_once(self):
        result = resolve_reflection(-0.3, -1.0, 0.0, 1.0)
        self.assertAlmostEqual(result.position, 0.3, places=12)
        self.assertEqual(result.walls, ("lower",))
        self.assertGreater(result.velocity, 0.0)

    def test_exact_upper_contact_moving_outward_reflects(self):
        result = resolve_reflection(1.0, 0.5, 0.0, 1.0)
        self.assertEqual(result.walls, ("upper",))
        self.assertAlmostEqual(result.position, 1.0, places=12)
        self.assertLess(result.velocity, 0.0)

    def test_exact_lower_contact_moving_outward_reflects(self):
        result = resolve_reflection(0.0, -0.5, 0.0, 1.0)
        self.assertEqual(result.walls, ("lower",))
        self.assertGreater(result.velocity, 0.0)

    def test_exact_contact_moving_inward_is_not_a_bounce(self):
        self.assertEqual(resolve_reflection(1.0, -0.5, 0.0, 1.0).walls, ())
        self.assertEqual(resolve_reflection(0.0, 0.5, 0.0, 1.0).walls, ())

    def test_two_wall_overshoot_records_both_walls_in_order(self):
        result = resolve_reflection(2.4, 1.0, 0.0, 1.0)
        self.assertEqual(result.walls, ("upper", "lower"))
        self.assertEqual(result.bounce_count, 2)
        self.assertAlmostEqual(result.position, 0.4, places=12)

    def test_many_wall_overshoot_is_accounted_not_collapsed(self):
        result = resolve_reflection(7.3, 1.0, 0.0, 1.0)
        self.assertGreaterEqual(result.bounce_count, 7)
        self.assertEqual(result.walls[0], "upper")
        self.assertTrue(0.0 <= result.position <= 1.0)

    def test_reflection_is_bounds_relative(self):
        result = resolve_reflection(31.0, 1.0, 10.0, 30.0)
        self.assertAlmostEqual(result.position, 29.0, places=12)
        self.assertEqual(result.walls, ("upper",))

    def test_inside_the_interval_is_identity(self):
        result = resolve_reflection(0.5, 1.0, 0.0, 1.0)
        self.assertEqual(result.walls, ())
        self.assertEqual(result.position, 0.5)
        self.assertEqual(result.velocity, 1.0)


class MovingDotTests(unittest.TestCase):
    def test_reflection_preserves_valid_position_and_overshoot(self):
        config = WorldConfig(dt=2.2, steps_per_episode=2)
        environment = MovingDotEnvironment(
            "bouncing", seed=1, config=config, initial_position=0.2, initial_velocity=1.0
        )
        transition = environment.advance()
        self.assertGreaterEqual(transition.position, 0.0)
        self.assertLessEqual(transition.position, 1.0)
        self.assertTrue(transition.bounced)
        self.assertAlmostEqual(transition.position, 0.4, places=9)
        self.assertGreater(environment.velocity, 0.0)
        self.assertEqual(transition.bounce_count, 2)

    def test_identical_seed_and_configuration_give_identical_trajectory(self):
        config = WorldConfig(steps_per_episode=20, change_step=10)
        first = MovingDotEnvironment("changed", 42, config)
        second = MovingDotEnvironment("changed", 42, config)
        first_values = [first.observe()]
        second_values = [second.observe()]
        first_events = []
        second_events = []
        for _ in range(config.steps_per_episode):
            first_step = first.advance()
            second_step = second.advance()
            first_values.append(first_step.position)
            second_values.append(second_step.position)
            first_events.append((first_step.bounce_walls, first_step.changed))
            second_events.append((second_step.bounce_walls, second_step.changed))
        self.assertEqual(first_values, second_values)
        self.assertEqual(first_events, second_events)

    def test_straight_motion_rejects_crossing_boundary(self):
        config = WorldConfig(dt=1.0, steps_per_episode=2)
        environment = MovingDotEnvironment(
            "straight", seed=3, config=config, initial_position=0.9, initial_velocity=0.2
        )
        with self.assertRaises(RuntimeError):
            environment.advance()

    def test_sampling_is_relative_to_arbitrary_bounds(self):
        config = WorldConfig(
            lower_bound=10.0, upper_bound=30.0, dt=0.1, steps_per_episode=20, speed_min=0.1, speed_max=0.2
        )
        environment = MovingDotEnvironment("bouncing", seed=9, config=config)
        self.assertGreaterEqual(environment.observe(), 10.0)
        self.assertLessEqual(environment.observe(), 30.0)
        for _ in range(config.steps_per_episode):
            transition = environment.advance()
            self.assertGreaterEqual(transition.position, 10.0)
            self.assertLessEqual(transition.position, 30.0)

    def test_exact_boundary_contact_reverses_outward_velocity(self):
        config = WorldConfig(lower_bound=-2.0, upper_bound=3.0, dt=1.0, steps_per_episode=2, history_length=2)
        environment = MovingDotEnvironment(
            "bouncing", seed=1, config=config, initial_position=3.0, initial_velocity=1.0
        )
        transition = environment.advance()
        self.assertTrue(transition.bounced)
        self.assertEqual(transition.position, 2.0)
        self.assertLess(environment.velocity, 0.0)

    def test_change_event_is_flagged_exactly_once(self):
        config = WorldConfig(steps_per_episode=20, change_step=5)
        environment = MovingDotEnvironment("changed", 7, config)
        flags = [environment.advance().changed for _ in range(config.steps_per_episode)]
        self.assertEqual(flags.count(True), 1)
        self.assertTrue(flags[5])

    def test_change_step_equal_to_the_horizon_is_rejected(self):
        with self.assertRaises(ValueError):
            WorldConfig(steps_per_episode=20, change_step=20)

    def test_change_step_zero_is_a_valid_first_transition(self):
        config = WorldConfig(steps_per_episode=20, change_step=0)
        environment = MovingDotEnvironment("changed", 7, config)
        self.assertTrue(environment.advance().changed)

    def test_last_transition_is_a_valid_change_step(self):
        config = WorldConfig(steps_per_episode=20, change_step=19)
        environment = MovingDotEnvironment("changed", 7, config)
        flags = [environment.advance().changed for _ in range(20)]
        self.assertTrue(flags[19])

    def test_no_change_step_means_no_change_event(self):
        config = WorldConfig(steps_per_episode=20, change_step=None)
        environment = MovingDotEnvironment("bouncing", 7, config)
        self.assertIsNone(environment.change_step)
        self.assertFalse(any(environment.advance().changed for _ in range(20)))

    def test_dynamics_change_scenario_is_rejected_by_the_dot_environment(self):
        with self.assertRaises(ValueError):
            MovingDotEnvironment("dynamics_change", 1, WorldConfig(steps_per_episode=5))

    def test_unknown_scenario_is_rejected(self):
        with self.assertRaises(ValueError):
            MovingDotEnvironment("teleport", 1, WorldConfig(steps_per_episode=5))

    def test_partial_initial_state_is_rejected(self):
        with self.assertRaises(ValueError):
            MovingDotEnvironment("bouncing", 1, WorldConfig(steps_per_episode=5), initial_position=0.5)

    def test_initial_position_outside_the_interval_is_rejected(self):
        with self.assertRaises(ValueError):
            MovingDotEnvironment(
                "bouncing",
                1,
                WorldConfig(steps_per_episode=5),
                initial_position=1.5,
                initial_velocity=0.1,
            )


class OscillatorTests(unittest.TestCase):
    def test_identical_seeds_give_identical_trajectories(self):
        config = WorldConfig(steps_per_episode=40, change_step=20)
        first = DampedOscillatorEnvironment(5, config)
        second = DampedOscillatorEnvironment(5, config)
        for _ in range(40):
            self.assertEqual(first.advance().position, second.advance().position)

    def test_coefficients_change_exactly_at_the_declared_transition(self):
        config = WorldConfig(steps_per_episode=40, change_step=20)
        environment = DampedOscillatorEnvironment(5, config)
        flags = [environment.advance().changed for _ in range(40)]
        self.assertEqual(flags.count(True), 1)
        self.assertTrue(flags[20])

    def test_no_change_step_keeps_the_original_law(self):
        config = WorldConfig(steps_per_episode=40, change_step=None)
        unchanged = DampedOscillatorEnvironment(5, config)
        positions = [unchanged.advance().position for _ in range(40)]
        self.assertFalse(any(unchanged.advance().changed for _ in range(0)))
        reference = DampedOscillatorEnvironment(5, config, changed_omega=1.5, changed_damping=0.10)
        self.assertEqual(positions, [reference.advance().position for _ in range(40)])

    def test_positions_stay_inside_the_configured_interval(self):
        config = WorldConfig(lower_bound=-5.0, upper_bound=5.0, steps_per_episode=200, change_step=100)
        environment = DampedOscillatorEnvironment(3, config)
        for _ in range(200):
            position = environment.advance().position
            self.assertGreaterEqual(position, -5.0)
            self.assertLessEqual(position, 5.0)

    def test_non_positive_coefficients_are_rejected(self):
        config = WorldConfig(steps_per_episode=10, change_step=5)
        with self.assertRaises(ValueError):
            DampedOscillatorEnvironment(1, config, omega=0.0)
        with self.assertRaises(ValueError):
            DampedOscillatorEnvironment(1, config, changed_damping=float("nan"))


if __name__ == "__main__":
    unittest.main()
