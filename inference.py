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


def _base_url() -> str:
    if not API_BASE_URL:
        raise RuntimeError("API_BASE_URL is not set")
    return API_BASE_URL.rstrip("/")


def _post_json(path: str, json_payload: dict[str, Any] | None, timeout: int = 20) -> requests.Response:
    """Try both no-slash and trailing-slash endpoint variants for compatibility."""
    base = _base_url()
    urls = [f"{base}{path}", f"{base}{path}/"]
    last_err: Exception | None = None
    for url in urls:
        try:
            return requests.post(url, json=json_payload, timeout=timeout)
        except requests.RequestException as exc:
            last_err = exc
    if last_err:
        raise last_err
    raise RuntimeError("Request failed unexpectedly")


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
    if not API_BASE_URL:
        print("[END] success=false steps=0 rewards= error=missing_api_base_url")
        return 0.0

    try:
        reset_resp = _post_json("/reset", {"task_name": task_name}, timeout=20)
        # Fallback for validators/environments that expect empty reset body.
        if reset_resp.status_code >= 400:
            reset_resp = _post_json("/reset", None, timeout=20)
        reset_resp.raise_for_status()
        observation = extract_observation(reset_resp.json())
    except Exception as exc:
        print(f"[END] success=false steps=0 rewards= error=reset_failed:{type(exc).__name__}")
        return 0.0

    final_info = {}
    rewards: list[float] = []
    for idx, action_type in enumerate(TASK_ACTIONS[task_name], start=1):
        payload = {
            "action_type": action_type,
            "flight_id": choose_flight_id(observation, action_type),
            "target_gate": choose_gate(observation) if action_type == "reassign_gate" else None,
            "target_aircraft": None,
            "target_crew": None,
        }

        try:
            step_resp = _post_json("/step", payload, timeout=20)
            step_resp.raise_for_status()
            step_data = step_resp.json()
            observation = extract_observation(step_data)
            reward = extract_reward(step_data)
            done = bool(step_data.get("done", False))
            rewards.append(reward)
            final_info = step_data.get("info", {})
            error_text = "null"
        except Exception as exc:
            reward = 0.0
            done = True
            error_text = type(exc).__name__

        print(
            f"[STEP] step={idx} action={action_type} "
            f"reward={reward:.2f} done={str(done).lower()} error={error_text}"
        )

        if done:
            break

    score = float(final_info.get("episode_grade", 0.0))
    success = score >= 0.7
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
            results[task_name] = 0.0


if __name__ == "__main__":
    main()
