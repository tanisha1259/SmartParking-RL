from sim.parking_env import ParkingEnvironment

env = ParkingEnvironment(total_slots=5)

state = env.reset()

print("Initial State:", state)

env.render()

next_state, reward, done = env.park_car(2)

print("\nNext State:", next_state)
print("Reward:", reward)
print("Done:", done)

env.render()