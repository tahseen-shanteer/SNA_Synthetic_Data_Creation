import random
import math
from math import ceil
from typing import Optional

RESTING = 1
STANDING = 2
MOVING = 3
FEEDING = 4
DRINKING = 5
MILKING = 6
SOCIALIZING = 7

from cow_pen import Cow_Pen

class Cow:
    all_cows = {}

    def __init__(self,
                 cow_id: int,
                 tag: str,
                 parity: int,
                 lactating: bool,
                 pen_id: int,
                 assigned_resting_zone):
        self.id = cow_id
        self.tag = tag
        self.parity = parity
        self.lactating = lactating
        self.speed = random.uniform(0.8, 1.2)
        self.friendliness = random.uniform(0.0, 1.0)
        self.aggressiveness = random.uniform(0.0, 1.0)
        self.rest_preference = random.uniform(0.0, 1.0)
        self.pos = None
        self.activity = STANDING
        self.activity_timer = 0
        self.path = []
        self.next_states = []
        self._postponed_path = None
        self._postponed_next_states = None
        self.friendships = {}
        self.interaction_radius = 150
        self.pen = Cow_Pen.pens[pen_id]
        self.home_pen_id = pen_id
        self.is_sick = False
        self.sickness_timer = 0
        self.assigned_resting_zone = assigned_resting_zone
        self.hunger = random.uniform(10, 40)
        self.thirst = random.uniform(10, 40)
        self.fatigue = random.uniform(0, 20)
        self.social_need = random.uniform(20, 50)
        Cow.all_cows[self.id] = self
        self.pen.register_cow(self)
        if self.assigned_resting_zone:
            try:
                self.pos = self.assigned_resting_zone.get_random_point()
            except Exception:
                self.pos = (random.randint(0, self.pen.width-1), random.randint(0, self.pen.height-1))
        else:
            free = self.pen.get_random_point_in_free_area()
            if free:
                self.pos = free
            else:
                self.pos = (random.randint(0, self.pen.width-1), random.randint(0, self.pen.height-1))

    def update(self, current_time: int):
        ts_of_day = current_time % 86400
        if self.is_sick and self.sickness_timer > 0 and self not in self.pen.cows:
            self.sickness_timer -= 1
            if self.sickness_timer <= 0:
                self._recover_from_sickness()
            return
        if self.is_sick:
            if self.activity == MOVING and self.path:
                self._move_along_path_and_check_social(current_time)
            else:
                entrance = self.pen.get_zone_by_name("entrance")
                if entrance and self.pos and \
                   abs(self.pos[0] - entrance.center[0]) < 2 and abs(self.pos[1] - entrance.center[1]) < 2:
                    if self in self.pen.cows:
                        self.pen.cows.remove(self)
                    self.pos = None
                    self.sickness_timer = random.randint(3*86400, 14*86400)
                    self.pen = Cow_Pen.pens[Cow_Pen.SICK_PEN_ID]
            return
        if random.random() < 0.00001:
            self._become_sick()
            return
        if ts_of_day in self.pen.milking_times and self.lactating:
            self._begin_milking(current_time)
        elif ts_of_day in self.pen.feeding_times:
            self._start_activity(FEEDING, current_time)
        else:
            if self.activity == MOVING and self.path:
                self._move_along_path_and_check_social(current_time)
            else:
                if self.activity_timer > 0:
                    self.activity_timer -= 1
                    if self.activity_timer <= 0:
                        self._on_activity_finish(current_time)
                else:
                    self._decide_next_activity(current_time)
        self._baseline_and_activity_needs_tick(current_time)

    def _become_sick(self):
        self.is_sick = True
        self.activity = MOVING
        self.path = []
        entrance = self.pen.get_zone_by_name("entrance")
        if entrance:
            self.path = self.pen.get_path_to_random_point_in_zone(self.pos, entrance)
        else:
            print(f"[Cow {self.id}] No entrance zone to exit for sickness.")

    def _recover_from_sickness(self):
        self.is_sick = False
        self.sickness_timer = 0
        self.pen = Cow_Pen.pens[self.home_pen_id]
        self.pen.cows.append(self)
        entrance = self.pen.get_zone_by_name("entrance")
        self.pos = entrance.center if entrance else (0, 0)
        self.activity = STANDING
        self.activity_timer = random.randint(60, 180)
        print(f"[Cow {self.id}] has recovered and returned to pen {self.home_pen_id}.")

    def _begin_milking(self, current_time: int):
        entrance = self.pen.zones["entrance"]
        if entrance:
            path = self.pen.get_path_to_zone_center(self.pos, entrance)
        else:
            print("[Warning] Pen has no entrance zone.")
            return
        travel_time = 0
        if path:
            travel_time = len(path) // (10.0 * self.speed)
        else:
            print(f"[Warning] Cow{self.id} cannot find path to entrance for milking. {self.pos} -> {entrance.center}")
            return
        self.path = path
        self.activity = MOVING
        self.next_states.insert(0, MILKING)
        self._pending_milking_extra = travel_time

    def _move_along_path_and_check_social(self, current_time: int):
        if not self.path:
            self._on_movement_finish(current_time)
            return
        step_count = max(1, int(10 * self.speed))
        idx = min(len(self.path) - 1, step_count - 1)
        self.pos = self.path[idx]
        self.path = self.path[idx+1:]
        nearby = []
        for other in self.pen.cows:
            if other.id == self.id or other.pos is None:
                continue
            d = math.hypot(self.pos[0] - other.pos[0], self.pos[1] - other.pos[1])
            if d <= self.interaction_radius:
                if other.activity not in (RESTING, DRINKING, MILKING):
                    nearby.append((other, d))
        if nearby:
            scored = []
            for other, d in nearby:
                friend_strength = self.friendships.get(other.id, 0)
                weight = friend_strength + (1.0 / (1.0 + d))
                scored.append((weight, other, d))
            scored.sort(reverse=True, key=lambda x: x[0])
            partner = scored[0][1]
            friend_strength = self.friendships.get(partner.id, 0)
            friend_score = 1 - math.exp(-friend_strength / 3.0)
            initiator_factor = (1.0 - self.aggressiveness * 0.1)
            partner_factor = (1.0 - partner.aggressiveness * 0.4)
            prob = max(0.05, min(0.95, friend_score * self.friendliness * initiator_factor * partner_factor))
            if random.random() < prob:
                self._postponed_path = self.path.copy() if self.path else []
                self._postponed_next_states = self.next_states.copy() if self.next_states else []
                partner._postponed_path = partner.path.copy() if partner.path else []
                partner._postponed_next_states = partner.next_states.copy() if partner.next_states else []
                self.path = []
                self.next_states = []
                partner.path = []
                partner.next_states = []
                dur = random.randint(60, 180)
                self.activity = SOCIALIZING
                partner.activity = SOCIALIZING
                self.activity_timer = dur
                partner.activity_timer = dur
                self.social_need = max(0.0, self.social_need - 15)
                partner.social_need = max(0.0, partner.social_need - 15)
                self.friendships[partner.id] = self.friendships.get(partner.id, 0) + 1
                partner.friendships[self.id] = partner.friendships.get(self.id, 0) + 1
                return
        if not self.path:
            self._on_movement_finish(current_time)

    def _on_movement_finish(self, current_time: int):
        if self.next_states:
            next_act = self.next_states.pop(0)
            if next_act == MILKING and hasattr(self, "_pending_milking_extra"):
                extra = self._pending_milking_extra
                base = random.randint(600, 900)
                self.activity = MILKING
                self.activity_timer = base + extra + self.pen.total_distance_to_and_from_milking // (10 * self.speed)
                delattr(self, "_pending_milking_extra")
                return
            self._start_activity(next_act, current_time)
        else:
            self._start_activity(STANDING, current_time)

    def _start_activity(self, activity: int, current_time: Optional[int] = None):
        self.activity_timer = 0
        self.path = []
        if activity == FEEDING:
            feed_zone = self.pen.assign_feeding_window(self)
            if feed_zone:
                path = self.pen.get_path_to_random_point_in_zone(self.pos, feed_zone)
            else:
                path = []
            if path:
                self.path = path
                self.next_states.insert(0, FEEDING)
                self.activity = MOVING
            else:
                self.activity = FEEDING
                self.activity_timer = random.randint(5400, 9000)
            return
        elif activity == DRINKING:
            water_zone = self.pen.get_zone_by_name("water_trough")
            if water_zone:
                path = self.pen.get_path_to_random_point_in_zone(self.pos, water_zone)
            else:
                path = []
            if path:
                self.path = path
                self.next_states.insert(0, DRINKING)
                self.activity = MOVING
            else:
                self.activity = DRINKING
                self.activity_timer = random.randint(120, 180)
            return
        elif activity == RESTING:
            bunk = self.assigned_resting_zone
            if bunk:
                path = self.pen.get_path_to_random_point_in_zone(self.pos, bunk)
            else:
                path = []
            if path:
                self.path = path
                self.next_states.insert(0, RESTING)
                self.activity = MOVING
            else:
                dur = self._rest_duration(current_time)
                self.activity = RESTING
                self.activity_timer = dur
            return
        elif activity == STANDING:
            free_zone = self.pen.get_random_point_in_free_area()
            if free_zone:
                path = self.pen.get_path_to_point(self.pos, free_zone)
            else:
                path = []
            if path:
                self.path = path
                self.next_states.insert(0, STANDING)
                self.activity = MOVING
            else:
                self.activity = STANDING
                self.activity_timer = random.randint(60, 180)
            return
        elif activity == SOCIALIZING:
            self.activity = SOCIALIZING
            self.activity_timer = random.randint(60, 600)
            return
        elif activity == MOVING:
            self.activity = MOVING
            return

    def _rest_duration(self, current_time: Optional[int]) -> int:
        base_day = 3600
        base_night = 5400
        if current_time is None:
            return random.randint(base_day, base_day + 1800)
        hour = (current_time // 3600) % 24
        if hour >= 20 or hour < 6:
            return random.randint(base_night, base_night + 3600)
        else:
            return random.randint(base_day, base_day + 1800)

    def _on_activity_finish(self, current_time: int):
        finished = self.activity
        self.activity = STANDING
        self.activity_timer = 0
        if finished == FEEDING:
            self.pen.unassign_feeding_window(self)
        if finished == MILKING:
            self.pos = self.pen.zones["entrance"].get_random_point()
            self._decide_next_activity(current_time)
            return
        if finished == SOCIALIZING:
            if self._postponed_path is not None:
                self.path = self._postponed_path
                self.next_states = self._postponed_next_states or []
                self._postponed_path = None
                self._postponed_next_states = None
                self.activity = MOVING
                return
        self._decide_next_activity(current_time)

    def _decide_next_activity(self, current_time: Optional[int] = None):
        if self.next_states:
            next_act = self.next_states.pop(0)
            self._start_activity(next_act, current_time)
            return
        thirst_w = self.thirst * (1.5 if self.lactating else 1.0)
        weights = {
            DRINKING: max(0.1, thirst_w),
            RESTING: max(0.1, self.fatigue),
            SOCIALIZING: max(0.1, self.social_need),
            STANDING: 10.0
        }
        total = sum(weights.values())
        acts, probs = zip(*[(a, w / total) for a, w in weights.items()])
        chosen = random.choices(acts, probs, k=1)[0]
        self._start_activity(chosen, current_time)

    def _baseline_and_activity_needs_tick(self, current_time: Optional[int]):
        baseline_hunger_inc = 0.005
        baseline_thirst_inc = 0.006 if self.lactating else 0.003
        baseline_fatigue_inc = 0.002
        baseline_social_inc = 0.0015
        self.hunger = min(100.0, self.hunger + baseline_hunger_inc)
        self.thirst = min(100.0, self.thirst + baseline_thirst_inc)
        self.fatigue = min(100.0, self.fatigue + baseline_fatigue_inc)
        self.social_need = min(100.0, self.social_need + baseline_social_inc)
        if self.activity == FEEDING:
            self.hunger = max(0.0, self.hunger - 0.05)
            self.thirst = min(100.0, self.thirst + 0.02)
        elif self.activity == DRINKING:
            self.thirst = max(0.0, self.thirst - 0.12)
        elif self.activity == RESTING:
            if current_time is not None:
                hour = (current_time // 3600) % 24
                if hour >= 20 or hour < 6:
                    self.fatigue = max(0.0, self.fatigue - 0.09)
                else:
                    self.fatigue = max(0.0, self.fatigue - 0.06)
            else:
                self.fatigue = max(0.0, self.fatigue - 0.06)
        elif self.activity == MOVING:
            self.hunger = min(100.0, self.hunger + 0.01)
            self.thirst = min(100.0, self.thirst + (0.02 if self.lactating else 0.01))
            self.fatigue = min(100.0, self.fatigue + 0.02)
        elif self.activity == MILKING:
            self.hunger = min(100.0, self.hunger + 0.02)
            self.fatigue = min(100.0, self.fatigue + 0.03)
        elif self.activity == SOCIALIZING:
            self.social_need = max(0.0, self.social_need - 0.03)
        elif self.activity == STANDING:
            self.hunger = min(100.0, self.hunger + 0.001)
            self.thirst = min(100.0, self.thirst + (0.001 if not self.lactating else 0.002))
            self.fatigue = min(100.0, self.fatigue + 0.001)

    def distance_to(self, other) -> float:
        if self.pos is None or other.pos is None:
            return float("inf")
        return math.hypot(self.pos[0] - other.pos[0], self.pos[1] - other.pos[1])

    def __repr__(self):
        return (f"Cow({self.id}, act={self.activity}, pos={self.pos}, "
                f"H={self.hunger:.1f}, T={self.thirst:.1f}, F={self.fatigue:.1f}, S={self.social_need:.1f})")