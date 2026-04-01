import unittest

from env.models import Action
from env.rewards import calculate_reward


class TestRewards(unittest.TestCase):
    def test_expected_action_gets_bonus(self):
        state = {"completed_actions": [], "backup_crew_count": 1, "matched_expected": 0}
        score, reason = calculate_reward(
            Action(action_type="reassign_gate", flight_id="NV102"),
            state,
            ["reassign_gate", "notify_passengers"],
        )
        self.assertGreater(score, 0.1)
        self.assertIn("next_expected", reason)

    def test_repeat_action_gets_penalty(self):
        state = {"completed_actions": ["notify_passengers"], "backup_crew_count": 1, "matched_expected": 1}
        score, reason = calculate_reward(
            Action(action_type="notify_passengers", flight_id="NV102"),
            state,
            ["reassign_gate", "notify_passengers"],
        )
        self.assertIn("repeated_action", reason)
        self.assertLess(score, 0.15)


if __name__ == "__main__":
    unittest.main()
