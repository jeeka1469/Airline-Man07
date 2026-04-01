import unittest

from env.graders import grade_easy, grade_hard, grade_medium


class TestGraders(unittest.TestCase):
    def test_scores_vary_by_sequence(self):
        score_a = grade_easy(["reassign_gate", "notify_passengers"])
        score_b = grade_easy(["notify_passengers", "reassign_gate"])
        self.assertNotEqual(score_a, score_b)

    def test_scores_are_clamped(self):
        score = grade_hard(["swap_aircraft"] * 50)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_medium_penalizes_cancel(self):
        good = grade_medium(["assign_backup_crew", "hold_connection", "notify_passengers"])
        bad = grade_medium(["assign_backup_crew", "cancel_flight", "notify_passengers"])
        self.assertGreater(good, bad)


if __name__ == "__main__":
    unittest.main()
