import random


class ParkingEnvironment:
    def __init__(self, total_slots=10):
        self.total_slots = total_slots
        self.reset()

    def reset(self):
        # 0 = empty slot
        # 1 = occupied slot
        # -1 = temporarily unavailable slot
        self.parking_slots = [0] * self.total_slots
        self.total_cars_parked = 0
        self.last_wait_time = 0

        self.traffic_level = random.choices(
            [0, 1, 2],
            weights=[0.5, 0.35, 0.15],
            k=1
        )[0]

        blocked_count = random.choices(
            [0, 1, 2],
            weights=[0.65, 0.25, 0.10],
            k=1
        )[0]
        blocked_count = min(blocked_count, self.total_slots - 3)

        for slot in random.sample(range(self.total_slots), blocked_count):
            self.parking_slots[slot] = -1

        available_capacity = len(self.available_slots())
        min_arrivals = min(3 + self.traffic_level, available_capacity)
        max_arrivals = min(5 + (self.traffic_level * 2), available_capacity)
        self.incoming_cars = random.randint(min_arrivals, max_arrivals)
        return self.get_state()

    def get_state(self):
        free_slots = self.parking_slots.count(0)
        occupied_slots = self.parking_slots.count(1)
        blocked_slots = self.parking_slots.count(-1)

        return (free_slots, occupied_slots, blocked_slots, self.traffic_level)

    def available_slots(self):
        return [i for i, slot in enumerate(self.parking_slots) if slot == 0]

    def park_car(self, action):
        """
        action = slot index chosen by agent
        """

        reward = 0
        done = False

        if action < 0 or action >= self.total_slots:
            next_state = self.get_state()
            self.last_wait_time = self.total_slots
            return next_state, -30, done

        # If chosen slot is empty
        if self.parking_slots[action] == 0:
            nearest_available_slot = min(self.available_slots())
            self.parking_slots[action] = 1
            self.total_cars_parked += 1

            occupancy_rate = self.total_cars_parked / self.total_slots
            entrance_congestion = self.traffic_level * max(0, 3 - action) * 1.1
            self.last_wait_time = 1 + (action * 0.4) + entrance_congestion

            fast_parking_reward = max(0, self.total_slots - self.last_wait_time) * 2.5
            wait_time_penalty = self.last_wait_time * 2
            congestion_avoidance_reward = (
                self.traffic_level * 2
                if action >= 3
                else 0
            )
            inefficiency_weight = max(0, 0.2 - (self.traffic_level * 0.1))
            inefficient_slot_penalty = (
                action - nearest_available_slot
            ) * inefficiency_weight
            occupancy_reward = occupancy_rate * 2

            # Keep reward aligned with operational goals, not inflated totals.
            reward = 4 + fast_parking_reward + congestion_avoidance_reward
            reward += occupancy_reward
            reward -= wait_time_penalty
            reward -= inefficient_slot_penalty

            # Extra penalty for high waiting time.
            if self.last_wait_time > 4:
                reward -= (self.last_wait_time - 4) * 2

            # Penalize congestion as the lot gets crowded.
            if occupancy_rate > 0.7:
                reward -= (occupancy_rate - 0.7) * 10

        else:
            # Penalty for choosing occupied/full slot
            self.last_wait_time = self.total_slots
            reward = -30

        # Parking lot full
        if self.total_cars_parked == self.incoming_cars or not self.available_slots():
            done = True

        next_state = self.get_state()

        return next_state, reward, done

    def render(self):
        print("\nParking Slots:")
        print(self.parking_slots)
