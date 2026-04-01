import unittest

from env.environment import AirlineDisruptionEnv
from env.models import Action


class TestStep(unittest.TestCase):
    def test_valid_step_changes_state(self):
        env = AirlineDisruptionEnv(task_name="easy")
        env.reset("easy")
        action = Action(action_type="reassign_gate", flight_id="NV102", target_gate="A7")
        response = env.step(action)

        self.assertEqual(response.reward.reason.split(",")[0], "base:reassign_gate")
        self.assertEqual(response.observation.flights[0].gate, "A7")
        self.assertIn("reassign_gate", response.info["actions_taken"])

    def test_invalid_action_penalized(self):
        env = AirlineDisruptionEnv(task_name="easy")
        env.reset("easy")
        action = Action(action_type="teleport_plane", flight_id="NV102")
        response = env.step(action)

        self.assertLess(response.reward.score, 0)
        self.assertIn("invalid_action", response.reward.reason)


if __name__ == "__main__":
    unittest.main()
