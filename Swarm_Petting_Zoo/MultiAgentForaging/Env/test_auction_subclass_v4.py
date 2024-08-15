from foraging_world_with_auctions_v2 import ForagingEnvironmentWithAuction
import time
import numpy as np

def manhattan_distance(a, b):
    """Calculate the Manhattan distance between two points."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def return_to_base(env, agent):
    """Determine the action to move the agent towards the home base."""
    current_location = env.get_agent_location(agent)
    home_base_location = env.get_home_base_location()
    direction = np.array(home_base_location) - np.array(current_location)
    if abs(direction[0]) > abs(direction[1]):
        return 0 if direction[0] > 0 else 2
    else:
        return 1 if direction[1] > 0 else 3

def foraging_behavior(env, agent):
    """Determine the action for the agent based on its state."""
    if env.get_carrying(agent):
        return return_to_base(env, agent)
    else:
        return env.action_space.sample()

def test_subclass_features(step_limit=10):
    env = ForagingEnvironmentWithAuction(num_agents=2, size=20, num_resources=5, fov=2, render_mode="human")
    env.reset(seed=42)

    step_count = 0
    while step_count < step_limit:
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last(observe=False)
            if termination or truncation:
                env.step(None)
                continue

            action = foraging_behavior(env, agent)
            env.step(action)
            obs = env.observe(agent)

            if obs['battery_level'] <= 0:
                env.terminations[agent] = True
                env.step(None)
                continue

            print(f"Agent {agent} post-step: Location: {env.get_agent_location(agent)}, Carrying: {env.get_carrying(agent)}, Money: {obs['money']}, Battery Level: {obs['battery_level']}")
            env.render()
            time.sleep(1)

            step_count += 1  # Increment the step count after each agent's action
            if step_count >= step_limit:
                print(f"Reached step limit of {step_limit}. Exiting simulation.")
                break

        if step_count >= step_limit:
            break  # Ensure the outer loop exits if the step limit is reached

        if all(env.terminations.values()):
            print("All agents are terminated. Exiting simulation.")
            break

    env.close()

if __name__ == "__main__":
    test_subclass_features()
