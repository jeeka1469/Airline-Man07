---
title: Airline Disruption Env
emoji: "✈️"
colorFrom: green
colorTo: indigo
sdk: docker
pinned: false
license: apache-2.0
---

# Airline Disruption Recovery Environment

A deterministic OpenEnv-style simulator where an AI agent handles operational disruption in a hub airport. The environment focuses on practical airline recovery decisions such as gate conflicts, crew recovery, maintenance recovery, weather impact handling, passenger communication, and VIP-sensitive prioritization.

## Why This Matters

Disruption recovery is one of the highest-cost airline operations problems. Delays cascade through gates, aircraft, crew legality windows, and passenger connection chains. A lightweight benchmark helps compare agent quality on real operations trade-offs, not just abstract planning.

## Project Structure

```txt
airline-disruption-env/
│── app.py
│── inference.py
│── openenv.yaml
│── Dockerfile
│── README.md
│── requirements.txt
│── .env.example
│── .gitignore
│── env/
│   ├── __init__.py
│   ├── models.py
│   ├── environment.py
│   ├── rewards.py
│   ├── graders.py
│   ├── tasks.py
│── data/
│   ├── easy_task.json
│   ├── medium_task.json
│   ├── hard_task.json
│── tests/
│   ├── test_reset.py
│   ├── test_step.py
│   ├── test_graders.py
│   ├── test_rewards.py
```

## Observation Space

Each observation includes:

- airport
- weather
- flights (with delay, gate, crew status, legality, maintenance flags, passenger load, VIP, connection risk)
- available_gates
- backup_crew_count
- completed_actions
- passenger_alerts

## Action Space

Allowed actions:

- assign_backup_crew
- reassign_gate
- delay_flight
- cancel_flight
- reroute_passengers
- hold_connection
- swap_aircraft
- notify_passengers
- reschedule_maintenance
- prioritize_departure

## Reward System

Rewards are shaped per step (not binary):

- Positive reward for useful operational actions
- Bonus for next expected action in sequence
- Small bonus for useful but out-of-order action
- Penalty for invalid action
- Penalty for repeated actions and loop behavior
- Penalty for illegal crew assignment attempt
- Penalty for cancellation-heavy behavior

The model gives agents incremental guidance while still requiring end-to-end strategy quality for top episode grades.

## Tasks

### Easy

- Single gate conflict with waiting passengers
- Expected sequence: `reassign_gate`, `notify_passengers`

### Medium

- Crew shortage with weather delay and connection risk
- Expected sequence: `assign_backup_crew`, `hold_connection`, `notify_passengers`

### Hard

- Storm pressure, maintenance issue, crew timeout risk, gate conflict, VIP sensitivity, and connection risk
- Expected sequence: `swap_aircraft`, `assign_backup_crew`, `reassign_gate`, `hold_connection`, `notify_passengers`

## API

Start server:

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Reset

```bash
curl -X POST http://127.0.0.1:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name":"easy"}'
```

### Step

```bash
curl -X POST http://127.0.0.1:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"reassign_gate","flight_id":"NV102","target_gate":"A7"}'
```

### State

```bash
curl http://127.0.0.1:7860/state
```

## Inference Runner

`inference.py`:

- Uses OpenAI Python client initialization
- Reads `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
- Runs easy/medium/hard tasks with deterministic action policy
- Prints logs using exact tokens: `START`, `STEP`, `END`

Run:

```bash
python inference.py
```

## Baseline Scores

With deterministic policy matching each expected solution:

- easy: typically 1.0
- medium: typically 1.0
- hard: typically 1.0

Alternative action order or cancellation behavior will reduce scores.

## Docker

Build:

```bash
docker build -t airline-disruption-env .
```

Run:

```bash
docker run -p 7860:7860 airline-disruption-env
```

## Hugging Face Spaces Deployment

1. Create a new Docker Space on Hugging Face.
2. Push this repository.
3. Set optional secrets (`API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`) if running external inference workflows.
4. Space will expose the FastAPI app on port 7860.

## Validation

Local checks:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Manual endpoint checks:

1. POST `/reset` for each task (`easy`, `medium`, `hard`)
2. POST `/step` with valid and invalid actions
3. GET `/state`
4. Run `inference.py` and verify `START/STEP/END` logging format
5. Confirm grader outputs differ across action sequences

OpenEnv checklist items such as runtime limits, memory profile, and endpoint HTTP health can be validated in your CI/deployment pipeline.
