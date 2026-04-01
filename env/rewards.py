from __future__ import annotations


REWARD_MAP = {
    "assign_backup_crew": 0.15,
    "reassign_gate": 0.10,
    "notify_passengers": 0.10,
    "hold_connection": 0.15,
    "swap_aircraft": 0.20,
    "prioritize_departure": 0.10,
    "cancel_flight": -0.20,
    "illegal_crew_assignment": -0.30,
    "invalid_action": -0.10,
    "wasted_step": -0.05,
}


def calculate_reward(action, state, expected_solution):
    action_type = action.action_type
    completed_actions = state.get("completed_actions", [])
    backup_crew_count = state.get("backup_crew_count", 0)

    score = REWARD_MAP.get(action_type, REWARD_MAP["invalid_action"])
    reasons = [f"base:{action_type}"]

    if action_type not in REWARD_MAP:
        reasons.append("invalid_action")

    if action_type == "assign_backup_crew" and backup_crew_count <= 0:
        score += REWARD_MAP["illegal_crew_assignment"]
        reasons.append("illegal_crew_assignment")

    if action_type in completed_actions:
        score += 2 * REWARD_MAP["wasted_step"]
        reasons.append("repeated_action")

    if len(completed_actions) >= 2 and completed_actions[-1] == action_type == completed_actions[-2]:
        score += REWARD_MAP["wasted_step"]
        reasons.append("loop_penalty")

    expected_index = state.get("matched_expected", 0)
    if expected_index < len(expected_solution):
        if action_type == expected_solution[expected_index]:
            score += 0.10
            reasons.append("next_expected")
        elif action_type in expected_solution:
            score += 0.02
            reasons.append("useful_but_out_of_order")

    score = max(-1.0, min(1.0, round(score, 4)))
    return score, ",".join(reasons)
