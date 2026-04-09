from __future__ import annotations

import os
from typing import Any

import requests
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
DEFAULT_API_BASE_URL = "http://127.0.0.1:7860"


client = OpenAI(
    api_key=OPENAI_API_KEY or HF_TOKEN or "not-set",
    base_url=API_BASE_URL if API_BASE_URL and API_BASE_URL.startswith("http") else None,
)


TASK_ORDER = ["easy", "medium", "hard"]
TASK_ACTIONS = {
    "easy": ["reassign_gate", "notify_passengers"],
    "medium": ["assign_backup_crew", "hold_connection", "notify_passengers"],
    "hard": ["swap_aircraft", "assign_backup_crew", "reassign_gate", "hold_connection", "notify_passengers"],
}
MIN_SCORE = 1e-3
MAX_SCORE = 1.0 - MIN_SCORE
SUCCESS_SCORE_THRESHOLD = 0.7


def _strict_score(value: float | int | None) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = MIN_SCORE
    return max(MIN_SCORE, min(MAX_SCORE, score))


def _base_url() -> str:
    return (API_BASE_URL or DEFAULT_API_BASE_URL).rstrip("/")


def _post_json(path: str, json_payload: dict[str, Any] | None, timeout: int = 20) -> requests.Response | None:
    base = _base_url()
    urls = [f"{base}{path}", f"{base}{path}/"]
    last_err: Exception | None = None
    for url in urls:
        try:
            return requests.post(url, json=json_payload, timeout=timeout)
        except requests.RequestException as exc:
            last_err = exc
    if last_err:
        print(f"[WARN] request_failed path={path} error={type(last_err).__name__}")
        return None
    print(f"[WARN] request_failed path={path} error=unknown")
    return None


def choose_flight_id(observation: dict[str, Any], action_type: str) -> str | None:
    flights = observation.get("flights", [])
    if action_type == "swap_aircraft":
        for flight in flights:
            if flight.get("maintenance_required"):
                return flight.get("flight_id")
    if action_type == "assign_backup_crew":
        for flight in flights:
            if not flight.get("crew_available"):
                return flight.get("flight_id")
    if action_type in {"reassign_gate", "hold_connection", "notify_passengers"}:
        for flight in flights:
            if flight.get("connection_risk") or flight.get("vip_onboard"):
                return flight.get("flight_id")
    return flights[0].get("flight_id") if flights else None


def choose_gate(observation: dict[str, Any]) -> str | None:
    available = observation.get("available_gates", [])
    return available[0] if available else None


def extract_observation(payload: dict[str, Any]) -> dict[str, Any]:
    observation = payload.get("observation")
    if isinstance(observation, dict):
        return observation
    return payload


def extract_reward(step_data: dict[str, Any]) -> float:
    reward_value = step_data.get("reward", 0.0)
    if isinstance(reward_value, dict):
        reward_value = reward_value.get("score", 0.0)
    try:
        return float(reward_value)
    except (TypeError, ValueError):
        return 0.0


def run_task(task_name: str):
    print(f"[START] task={task_name} env=airline-disruption-env model={MODEL_NAME}")

    reset_resp = _post_json("/reset", {"task_name": task_name}, timeout=20)
    if reset_resp is None or reset_resp.status_code >= 400:
        reset_resp = _post_json("/reset", None, timeout=20)

    if reset_resp is None:
        print("[END] success=false steps=0 rewards= error=reset_failed:connection")
        return MIN_SCORE

    if reset_resp.status_code >= 400:
        print(f"[END] success=false steps=0 rewards= error=reset_failed:http_{reset_resp.status_code}")
        return MIN_SCORE

    try:
        observation = extract_observation(reset_resp.json())
    except Exception as exc:
        print(f"[END] success=false steps=0 rewards= error=reset_failed:{type(exc).__name__}")
        return MIN_SCORE

    rewards: list[float] = []
    for idx, action_type in enumerate(TASK_ACTIONS[task_name], start=1):
        payload = {
            "action_type": action_type,
            "flight_id": choose_flight_id(observation, action_type),
            "target_gate": choose_gate(observation) if action_type == "reassign_gate" else None,
            "target_aircraft": None,
            "target_crew": None,
        }

        step_resp = _post_json("/step", payload, timeout=20)
        if step_resp is None:
            reward = 0.0
            done = True
            error_text = "connection"
        elif step_resp.status_code >= 400:
            reward = 0.0
            done = True
            error_text = f"http_{step_resp.status_code}"
        else:
            try:
                step_data = step_resp.json()
                observation = extract_observation(step_data)
                reward = extract_reward(step_data)
                done = bool(step_data.get("done", False))
                rewards.append(reward)
                error_text = "null"
            except Exception as exc:
                reward = 0.0
                done = True
                error_text = type(exc).__name__

        if error_text != "null":
            reward = 0.0
            done = True

        print(
            f"[STEP] step={idx} action={action_type} "
            f"reward={reward:.2f} done={str(done).lower()} error={error_text}"
        )

        if done:
            break

    score = sum(rewards) / len(rewards) if rewards else MIN_SCORE
    score = _strict_score(score)
    success = score >= SUCCESS_SCORE_THRESHOLD
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} "
        f"steps={len(rewards)} rewards={rewards_str}"
    )
    return score


def main():
    _ = client
    results = {}
    for task_name in TASK_ORDER:
        try:
            results[task_name] = run_task(task_name)
        except Exception as exc:
            print(f"[START] task={task_name} env=airline-disruption-env model={MODEL_NAME}")
            print(f"[END] success=false steps=0 rewards= error=unhandled:{type(exc).__name__}")
            results[task_name] = MIN_SCORE


if __name__ == "__main__":
    main()
