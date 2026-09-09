import unittest

from aaa.config import WorldConfig
from aaa.environment import MovingDotEnvironment


class EnvironmentTests(unittest.TestCase):
    def test_reflection_preserves_valid_position_and_overshoot(self):
        config = WorldConfig(dt=2.2, steps_per_episode=2)
        environment = MovingDotEnvironment(
            "bouncing",
            seed=1,
            config=config,
            initial_position=0.2,
            initial_velocity=1.0,
        )
        transition = environment.advance()
        self.assertGreaterEqual(transition.position, 0.0)
        self.assertLessEqual(transition.position, 1.0)
        self.assertTrue(transition.bounced)
        self.assertAlmostEqual(transition.position, 0.4, places=9)
        self.assertGreater(environment.velocity, 0.0)

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
            first_events.append((first_step.bounced, first_step.changed))
            second_events.append((second_step.bounced, second_step.changed))
        self.assertEqual(first_values, second_values)
        self.assertEqual(first_events, second_events)

    def test_straight_motion_rejects_crossing_boundary(self):
        config = WorldConfig(dt=1.0, steps_per_episode=2)
        environment = MovingDotEnvironment(
            "straight",
            seed=3,
            config=config,
            initial_position=0.9,
            initial_velocity=0.2,
        )
        with self.assertRaises(RuntimeError):
            environment.advance()


if __name__ == "__main__":
    unittest.main()
