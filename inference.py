from __future__ import annotations

import os
from typing import Any

import requests
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")


client = OpenAI(
    api_key=HF_TOKEN or "not-set",
    base_url=API_BASE_URL if API_BASE_URL and API_BASE_URL.startswith("http") else None,
)


TASK_ORDER = ["easy", "medium", "hard"]
TASK_ACTIONS = {
    "easy": ["reassign_gate", "notify_passengers"],
    "medium": ["assign_backup_crew", "hold_connection", "notify_passengers"],
    "hard": ["swap_aircraft", "assign_backup_crew", "reassign_gate", "hold_connection", "notify_passengers"],
}


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


def run_task(task_name: str):
    if not API_BASE_URL:
        raise RuntimeError("API_BASE_URL is not set")

    reset_resp = requests.post(f"{API_BASE_URL}/reset", json={"task_name": task_name}, timeout=20)
    reset_resp.raise_for_status()
    observation = reset_resp.json()

    print(f"START {task_name}")

    final_info = {}
    for idx, action_type in enumerate(TASK_ACTIONS[task_name], start=1):
        payload = {
            "action_type": action_type,
            "flight_id": choose_flight_id(observation, action_type),
            "target_gate": choose_gate(observation) if action_type == "reassign_gate" else None,
            "target_aircraft": None,
            "target_crew": None,
        }

        print(f"STEP {task_name} {idx} {action_type}")
        step_resp = requests.post(f"{API_BASE_URL}/step", json=payload, timeout=20)
        step_resp.raise_for_status()
        step_data = step_resp.json()
        observation = step_data["observation"]
        final_info = step_data["info"]

        if step_data.get("done"):
            break

    score = final_info.get("episode_grade", 0.0)
    print(f"END {task_name} score={score}")
    return score


def main():
    _ = client
    results = {}
    for task_name in TASK_ORDER:
        results[task_name] = run_task(task_name)

    print("START summary")
    for task_name in TASK_ORDER:
        print(f"STEP summary {task_name} {results[task_name]}")
    print("END summary")


if __name__ == "__main__":
    main()
