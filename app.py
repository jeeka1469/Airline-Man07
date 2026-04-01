from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from env.environment import AirlineDisruptionEnv
from env.models import Action, Observation, StepResponse


app = FastAPI(title="Airline Disruption Recovery Environment")
env = AirlineDisruptionEnv(task_name="easy")


class ResetRequest(BaseModel):
    task_name: str = "easy"


@app.get("/")
def root():
    return {
        "name": "airline-disruption-env",
        "status": "ok",
        "endpoints": ["/reset", "/step", "/state", "/health", "/metadata", "/schema", "/mcp"],
    }


@app.post("/reset", response_model=Observation)
def reset_environment(payload: ResetRequest):
    try:
        return env.reset(task_name=payload.task_name)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/step", response_model=StepResponse)
def step_environment(action: Action):
    try:
        return env.step(action)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state", response_model=Observation)
def get_state():
    return env.state()


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/metadata")
def metadata():
    return {
        "name": "airline-disruption-env",
        "description": "Airline disruption recovery environment for evaluating AI agents.",
        "version": "1.0.0",
    }


@app.get("/schema")
def schema():
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": Observation.model_json_schema(),
    }


@app.post("/mcp")
def mcp_ping(payload: dict):
    return {
        "jsonrpc": "2.0",
        "id": payload.get("id", 1),
        "result": {"status": "ok"},
    }
