from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from env.environment import AirlineDisruptionEnv
from env.models import Action, Observation, StepResponse


app = FastAPI(title="Airline Disruption Recovery Environment")
env = AirlineDisruptionEnv(task_name="easy")
FRONTEND_FILE = Path(__file__).resolve().parent / "frontend" / "index.html"


class ResetRequest(BaseModel):
    task_name: str = "easy"


@app.get("/")
def root():
    if FRONTEND_FILE.exists():
        return FileResponse(str(FRONTEND_FILE))
    return {
        "name": "airline-disruption-env",
        "status": "ok",
        "endpoints": ["/reset", "/step", "/state", "/health", "/metadata", "/schema", "/mcp"],
    }


@app.get("/api")
def api_info():
    return {
        "name": "airline-disruption-env",
        "status": "ok",
        "endpoints": ["/reset", "/step", "/state", "/health", "/metadata", "/schema", "/mcp"],
    }


@app.post("/reset", response_model=Observation)
@app.post("/reset/", response_model=Observation, include_in_schema=False)
def reset_environment(payload: ResetRequest | None = None):
    try:
        task_name = payload.task_name if payload else "easy"
        return env.reset(task_name=task_name)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/step", response_model=StepResponse)
@app.post("/step/", response_model=StepResponse, include_in_schema=False)
def step_environment(action: Action):
    try:
        return env.step(action)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state", response_model=Observation)
@app.get("/state/", response_model=Observation, include_in_schema=False)
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


def main() -> None:
	uvicorn.run("app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
	main()
