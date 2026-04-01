from __future__ import annotations

from copy import deepcopy

from env.graders import grade_easy, grade_hard, grade_medium
from env.models import Action, Flight, Observation, Reward, StepResponse
from env.rewards import calculate_reward
from env.tasks import load_task


ALLOWED_ACTIONS = [
    "assign_backup_crew",
    "reassign_gate",
    "delay_flight",
    "cancel_flight",
    "reroute_passengers",
    "hold_connection",
    "swap_aircraft",
    "notify_passengers",
    "reschedule_maintenance",
    "prioritize_departure",
]


class AirlineDisruptionEnv:
    def __init__(self, task_name: str = "easy"):
        self.task_name = task_name
        self.task = {}
        self.flights: list[Flight] = []
        self.available_gates: list[str] = []
        self.backup_crew_count: int = 0
        self.completed_actions: list[str] = []
        self.passenger_alerts: list[str] = []
        self.action_log: list[dict] = []
        self.shift_note: str = ""
        self.expected_solution: list[str] = []
        self.max_steps: int = 1
        self.step_count: int = 0
        self.done: bool = False
        self.last_reward: Reward = Reward(score=0.0, reason="reset")
        self.reset(task_name=task_name)

    def reset(self, task_name: str | None = None):
        if task_name:
            self.task_name = task_name

        self.task = load_task(self.task_name)
        self.flights = [Flight(**flight_data) for flight_data in self.task["flights"]]
        self.available_gates = list(self.task["available_gates"])
        self.backup_crew_count = int(self.task["available_backup_crew"])
        self.expected_solution = list(self.task["expected_solution"])
        self.max_steps = int(self.task["max_steps"])
        self.shift_note = str(self.task.get("shift_note", "Dispatch prefers smallest viable fix first."))
        self.completed_actions = []
        self.passenger_alerts = []
        self.action_log = []
        self.step_count = 0
        self.done = False
        self.last_reward = Reward(score=0.0, reason="reset")
        return self.build_observation()

    def _matched_expected_count(self) -> int:
        idx = 0
        for action in self.completed_actions:
            if idx < len(self.expected_solution) and action == self.expected_solution[idx]:
                idx += 1
        return idx

    def _reward_state(self) -> dict:
        return {
            "completed_actions": deepcopy(self.completed_actions),
            "backup_crew_count": self.backup_crew_count,
            "matched_expected": self._matched_expected_count(),
        }

    def _find_flight(self, flight_id: str | None) -> Flight | None:
        if not flight_id:
            return None
        for flight in self.flights:
            if flight.flight_id == flight_id:
                return flight
        return None

    def _ops_pressure(self) -> int:
        pressure = 0
        for flight in self.flights:
            pressure += min(60, flight.delay_minutes)
            pressure += 8 if flight.connection_risk else 0
            pressure += 6 if flight.maintenance_required else 0
            pressure += 10 if not flight.crew_available else 0
            pressure += 4 if flight.vip_onboard else 0
            pressure += 12 if flight.cancelled else 0
        return pressure

    def _log_action(self, action: Action, valid: bool, reason: str):
        self.action_log.append(
            {
                "step": self.step_count + 1,
                "action": action.action_type,
                "flight_id": action.flight_id,
                "valid": valid,
                "reason": reason,
                "ops_pressure": self._ops_pressure(),
            }
        )

    def apply_action(self, action: Action):
        target_flight = self._find_flight(action.flight_id)

        if action.action_type == "assign_backup_crew" and target_flight:
            if self.backup_crew_count > 0 and not target_flight.crew_available:
                target_flight.crew_available = True
                target_flight.crew_legal = True
                self.backup_crew_count -= 1

        elif action.action_type == "reassign_gate" and target_flight and action.target_gate:
            if action.target_gate in self.available_gates:
                self.available_gates.remove(action.target_gate)
                self.available_gates.append(target_flight.gate)
                target_flight.gate = action.target_gate

        elif action.action_type == "delay_flight" and target_flight:
            target_flight.delay_minutes += 15

        elif action.action_type == "cancel_flight" and target_flight:
            target_flight.cancelled = True
            target_flight.connection_risk = False

        elif action.action_type == "reroute_passengers" and target_flight:
            rerouted = max(1, target_flight.passenger_count // 10)
            target_flight.passenger_count = max(0, target_flight.passenger_count - rerouted)
            target_flight.connection_risk = False

        elif action.action_type == "hold_connection" and target_flight:
            target_flight.connection_risk = False
            target_flight.delay_minutes += 10

        elif action.action_type == "swap_aircraft" and target_flight:
            if target_flight.maintenance_required:
                target_flight.maintenance_required = False
                target_flight.delay_minutes = max(0, target_flight.delay_minutes - 20)

        elif action.action_type == "notify_passengers":
            if target_flight:
                msg = f"Advisory sent for {target_flight.flight_id} to {target_flight.destination}."
                if msg not in self.passenger_alerts:
                    self.passenger_alerts.append(msg)
            else:
                self.passenger_alerts.append("Terminal-wide disruption advisory sent.")

        elif action.action_type == "reschedule_maintenance" and target_flight:
            if target_flight.maintenance_required:
                target_flight.maintenance_required = False
                target_flight.delay_minutes += 20

        elif action.action_type == "prioritize_departure" and target_flight:
            bonus = 20 if target_flight.vip_onboard else 15
            target_flight.delay_minutes = max(0, target_flight.delay_minutes - bonus)

    def _validate_action(self, action: Action) -> tuple[bool, str]:
        if action.action_type not in ALLOWED_ACTIONS:
            return False, "Action type is not allowed."

        needs_flight = action.action_type not in {"notify_passengers"}
        if needs_flight and not action.flight_id:
            return False, "flight_id is required for this action."

        if needs_flight and not self._find_flight(action.flight_id):
            return False, "flight_id does not exist in current scenario."

        if action.action_type == "reassign_gate" and not action.target_gate:
            return False, "target_gate is required for reassign_gate."

        return True, "ok"

    def _grade_episode(self) -> float:
        if self.task_name == "easy":
            return grade_easy(self.completed_actions)
        if self.task_name == "medium":
            return grade_medium(self.completed_actions)
        return grade_hard(self.completed_actions)

    def check_done(self):
        if self.step_count >= self.max_steps:
            return True
        return self._matched_expected_count() >= len(self.expected_solution)

    def build_observation(self):
        return Observation(
            airport=self.task["airport"],
            weather=self.task["weather"],
            flights=self.flights,
            available_gates=self.available_gates,
            backup_crew_count=self.backup_crew_count,
            completed_actions=self.completed_actions,
            passenger_alerts=self.passenger_alerts,
        )

    def step(self, action: Action):
        if self.done:
            return StepResponse(
                observation=self.build_observation(),
                reward=Reward(score=0.0, reason="episode_already_done"),
                done=True,
                info={
                    "step_count": self.step_count,
                    "max_steps": self.max_steps,
                    "actions_taken": self.completed_actions,
                    "episode_grade": self._grade_episode(),
                },
            )

        valid, reason = self._validate_action(action)

        if valid:
            score, reward_reason = calculate_reward(action, self._reward_state(), self.expected_solution)
            self.apply_action(action)
            self.completed_actions.append(action.action_type)
        else:
            score, reward_reason = -0.1, f"invalid_action:{reason}"

        self._log_action(action, valid, reason)

        self.step_count += 1
        self.done = self.check_done()
        self.last_reward = Reward(score=score, reason=reward_reason)

        return StepResponse(
            observation=self.build_observation(),
            reward=self.last_reward,
            done=self.done,
            info={
                "step_count": self.step_count,
                "max_steps": self.max_steps,
                "actions_taken": self.completed_actions,
                "episode_grade": self._grade_episode(),
                "validation": reason,
                "ops_pressure": self._ops_pressure(),
                "shift_note": self.shift_note,
                "action_log_tail": self.action_log[-3:],
            },
        )

    def state(self):
        return self.build_observation()
