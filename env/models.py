from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Flight(BaseModel):
    flight_id: str
    destination: str
    carrier_name: str = "Ops"
    logo_domain: str | None = None
    aircraft_type: str | None = None
    delay_minutes: int = 0
    gate: str
    crew_available: bool
    crew_legal: bool
    maintenance_required: bool
    passenger_count: int
    vip_onboard: bool
    connection_risk: bool
    cancelled: bool = False


class Observation(BaseModel):
    airport: str
    weather: str
    flights: list[Flight]
    available_gates: list[str]
    backup_crew_count: int
    completed_actions: list[str]
    passenger_alerts: list[str]


class Action(BaseModel):
    action_type: str
    flight_id: str | None = None
    target_gate: str | None = None
    target_aircraft: str | None = None
    target_crew: str | None = None


class Reward(BaseModel):
    score: float = Field(ge=-1.0, le=1.0)
    reason: str


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict[str, Any]
