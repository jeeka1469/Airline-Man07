from __future__ import annotations


def _clamp_score(score: float) -> float:
    return max(0.0, min(1.0, round(score, 4)))


def _grade_actions(actions_taken: list[str], expected: list[str], weights: dict[str, float]) -> float:
    score = 0.0

    for action_name, weight in weights.items():
        if action_name in actions_taken:
            score += weight

    prefix = 0
    for action in actions_taken:
        if prefix < len(expected) and action == expected[prefix]:
            prefix += 1

    order_bonus = 0.2 * (prefix / max(1, len(expected)))
    duplicate_penalty = 0.02 * max(0, len(actions_taken) - len(set(actions_taken)))
    cancel_penalty = 0.1 if "cancel_flight" in actions_taken else 0.0
    sequence_penalty = 0.0
    for idx, expected_action in enumerate(expected):
        if expected_action in actions_taken:
            actual_idx = actions_taken.index(expected_action)
            if actual_idx > idx:
                sequence_penalty += 0.03

    subtotal = min(1.0, score + order_bonus)
    return _clamp_score(subtotal - duplicate_penalty - cancel_penalty - sequence_penalty)


def grade_easy(actions_taken):
    expected = ["reassign_gate", "notify_passengers"]
    weights = {
        "reassign_gate": 0.5,
        "notify_passengers": 0.5,
    }
    return _grade_actions(actions_taken, expected, weights)


def grade_medium(actions_taken):
    expected = ["assign_backup_crew", "hold_connection", "notify_passengers"]
    weights = {
        "assign_backup_crew": 0.25,
        "hold_connection": 0.25,
        "notify_passengers": 0.25,
        "avoid_cancel": 0.25,
    }
    transformed_actions = list(actions_taken)
    if "cancel_flight" not in transformed_actions:
        transformed_actions.append("avoid_cancel")
    return _grade_actions(transformed_actions, expected + ["avoid_cancel"], weights)


def grade_hard(actions_taken):
    expected = [
        "swap_aircraft",
        "assign_backup_crew",
        "reassign_gate",
        "hold_connection",
        "notify_passengers",
    ]
    weights = {
        "swap_aircraft": 0.2,
        "assign_backup_crew": 0.2,
        "reassign_gate": 0.2,
        "hold_connection": 0.2,
        "notify_passengers": 0.2,
    }
    return _grade_actions(actions_taken, expected, weights)
