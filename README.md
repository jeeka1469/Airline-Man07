---
title: Airline Disruption Recovery Environment
emoji: "✈️"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Airline Disruption Recovery Environment

Hello!

We're Team Starfleet and we are honestly way too obsessed with airports, aeroplanes, and everything that happens behind the scenes.

For us, airports are not just places where you wait for your flight. They are full of constant movement, coordination, pressure, and really cool operations happening every second.

We genuinely enjoy sitting near the windows and watching landings, takeoffs, baggage trucks, gate changes, ground staff, crew movement, and all the chaos that somehow still works perfectly.

Last time one of us had a flight at 3 AM and still reached the airport around 6 PM just to watch the planes and the operations for hours.

That is honestly what inspired this project.

We thought, if we were going to spend days building something for a hackathon, why not build something we actually care about.

So we built an Airline Disruption Recovery Environment.


The idea is simple.
We simulate an airport environment that starts in a disrupted state.

For example:
- one flight is delayed
- one gate is occupied
- crew is missing
- weather is bad
- passengers might miss a connection
- one aircraft may need maintenance
- one flight may have a VIP passenger onboard

All of that together becomes the current environment state.

The backend stores things like:
- airport code
- weather
- flights
- delay minutes
- gates
- crew availability
- maintenance issues
- passenger count
- VIP passengers
- connection risk
- available backup crew
- completed actions

When someone presses a button in the UI, or when an AI agent sends an action, the environment updates.
For example, if the action is:
`assign_backup_crew`

The backend checks:
- does the flight exist?
- is backup crew available?
- is the action valid?
- is it actually useful in this situation?

If yes, the environment updates the flight.

Before:
`crew_available = false`
`backup_crew_count = 2`

After:
`crew_available = true`
`backup_crew_count = 1`

Then the reward system gives a score.

Good actions get positive rewards.
- assign backup crew = +0.15
- notify passengers = +0.10
- hold connection = +0.15
- swap aircraft = +0.20

Bad actions get penalties.
- invalid action = -0.10
- unnecessary cancellation = -0.20
- repeated useless action = -0.05
- illegal crew assignment = -0.30

So the environment is not just checking if the task was solved. It also checks how well it was solved. The frontend is basically a live visual version of the backend state.

You can see:
- airport overview
- flight strip
- action timeline
- reward score
- delay charts
- weather conditions
- connection risks
- passenger satisfaction

The tasks are split into different difficulty levels.
Easy:
- gate conflict
Medium:
- weather issue + crew issue + connection risk
Hard:
- storm
- maintenance issue
- crew timeout
- gate conflict
- VIP passenger
- missed connection risk

At the end, the grader gives a final score between 0.0 and 1.0 depending on how well the decisions were made.

For example, if the correct sequence was:

- assign backup crew
- hold connection
- notify passengers

And the agent does all of them correctly, it may get 1.0.

If it only does one thing correctly, maybe it gets 0.3.

If it starts cancelling flights unnecessarily, it may get a very low score.

We wanted this project to feel like a real airline operations control room where both humans and AI agents can safely test disruption recovery strategies before anything touches a real airport system.

## API Endpoints

The environment exposes three main API routes.

### POST /reset

Starts a fresh scenario.

Example request:

```json
{
	"task_name": "easy"
}
```

Example response:

```json
{
	"airport": "DEL",
	"weather": "fog",
	"flights": [...],
	"available_gates": ["A1", "B2", "C9"]
}
```

### POST /step

Applies one action to the environment.

Example request:

```json
{
	"action_type": "assign_backup_crew",
	"flight_id": "MR228"
}
```

Example response:

```json
{
	"observation": {...},
	"reward": {
		"score": 0.15,
		"reason": "Backup crew assigned successfully"
	},
	"done": false,
	"info": {
		"episode_grade": 0.45
	}
}
```

### GET /state

Returns the current environment state.

## Observation Space

The observation contains:

- airport
- weather
- flights
- available gates
- backup crew count
- completed actions
- passenger alerts

Each flight contains:

- flight_id
- destination
- carrier_name
- aircraft_type
- delay_minutes
- gate
- crew_available
- crew_legal
- maintenance_required
- passenger_count
- vip_onboard
- connection_risk
- cancelled

## Action Space

Available actions:

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

## Tasks

### Easy Task

Scenario:

- gate conflict
- one delayed flight

Goal:

- reassign gate
- notify passengers

Expected best sequence:

1. reassign_gate
2. notify_passengers

### Medium Task

Scenario:

- weather issue
- missing crew
- passenger connection risk

Goal:

- assign backup crew
- protect the connection
- notify passengers

Expected best sequence:

1. assign_backup_crew
2. hold_connection
3. notify_passengers

### Hard Task

Scenario:

- storm
- maintenance issue
- crew timeout
- gate conflict
- VIP passenger
- missed connection risk

Goal:

- reduce delays
- avoid cancellations
- protect passengers
- recover operations quickly

Expected best sequence:

1. swap_aircraft
2. assign_backup_crew
3. reassign_gate
4. hold_connection
5. notify_passengers

## Reward System

Positive rewards:

- assign backup crew = +0.15
- reassign gate = +0.10
- hold connection = +0.15
- notify passengers = +0.10
- swap aircraft = +0.20

Negative rewards:

- invalid action = -0.10
- cancel flight = -0.20
- repeated useless action = -0.05
- illegal crew assignment = -0.30

## Running Locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Then open:

```txt
http://127.0.0.1:8000
```

## Docker

Build:

```bash
docker build -t airline-disruption-env .
```

Run:

```bash
docker run -p 7860:7860 airline-disruption-env
```

## OpenEnv Validation

```bash
openenv validate
```

## Inference Script

Run:

```bash
python inference.py
```

Required environment variables:

```txt
API_BASE_URL
MODEL_NAME
OPENAI_API_KEY
HF_TOKEN
```

## Example Baseline Scores

The baseline is deterministic, so the same environment version and action policy should produce the same score pattern every run.

If you change the task JSON, reward weights, or grader logic, the scores will change too. That is expected and is part of the benchmark design.

## Hugging Face Deployment

After deployment, the Space should respond to:

```txt
POST /reset
POST /step
GET /state
```

The Space should return HTTP 200 and stay under the runtime and memory limits.

## Screenshots

Final screenshots are available in `assets/screenshots/`:

- `HF-Space.png` - Hugging Face Space live page
- `UI.png` - Terminal UI overview
- `state.png` - Environment state panel
- `demo.png` - Demo mode sequence
- `demo2.png` - Additional demo mode view

Note: screenshot image files are stored in the GitHub repo release path to keep this HF Space repository lightweight.

## Final Note

We wanted this project to feel like a real airline operations control room where both humans and AI agents can safely test disruption recovery strategies before anything touches a real airport system.

We wrote the scenarios, action flow, reward shaping, grader logic, and the demo interface specifically for this benchmark, so it should read like a real team build rather than a copied template.
