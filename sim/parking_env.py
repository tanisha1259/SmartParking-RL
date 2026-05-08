import random


class ParkingEnvironment:
    def __init__(self, total_slots=10):
        self.total_slots = total_slots
        self.reset()

    def reset(self):
        # 0 = empty slot
        # 1 = occupied slot
        self.parking_slots = [0] * self.total_slots
        self.total_cars_parked = 0
        self.incoming_cars = random.randint(3, self.total_slots)
        return self.get_state()

    def get_state(self):
        free_slots = self.parking_slots.count(0)
        occupied_slots = self.parking_slots.count(1)

        return (free_slots, occupied_slots)

    def available_slots(self):
        return [i for i, slot in enumerate(self.parking_slots) if slot == 0]

    def park_car(self, action):
        """
        action = slot index chosen by agent
        """

        reward = 0
        done = False

        # If chosen slot is empty
        if self.parking_slots[action] == 0:
            self.parking_slots[action] = 1
            self.total_cars_parked += 1

            # Reward for successful parking
            reward = 10 - action

        else:
            # Penalty for choosing occupied slot
            reward = -10

        # Parking lot full
        if self.total_cars_parked == self.incoming_cars:
            done = True

        next_state = self.get_state()

        return next_state, reward, done

    def render(self):
        print("\nParking Slots:")
        print(self.parking_slots)