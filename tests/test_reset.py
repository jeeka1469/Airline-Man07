import unittest

from env.environment import AirlineDisruptionEnv


class TestReset(unittest.TestCase):
    def test_reset_easy_task(self):
        env = AirlineDisruptionEnv(task_name="easy")
        observation = env.reset("easy")
        self.assertEqual(observation.airport, "NVA")
        self.assertEqual(observation.weather, "clear")
        self.assertEqual(len(observation.completed_actions), 0)
        self.assertEqual(env.step_count, 0)
        self.assertFalse(env.done)


if __name__ == "__main__":
    unittest.main()
