from foraging_world_with_auctions import ForagingEnvironmentWithAuction
import time
import numpy as np

def manhattan_distance(a, b):
    """Calculate the Manhattan distance between two points."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def return_to_base(env, agent):
    """Determine the action to move the agent towards the home base."""
    current_location = env.get_agent_location(agent)
    home_base_location = env.get_home_base_location()

    # Calculate the direction towards the home base
    direction = np.array(home_base_location) - np.array(current_location)

    # Convert direction to an action
    if abs(direction[0]) > abs(direction[1]):
        if direction[0] > 0:
            return 0  # Move right
        else:
            return 2  # Move left
    else:
        if direction[1] > 0:
            return 1  # Move down
        else:
            return 3  # Move up

def foraging_behavior(env, agent):
    """Determine the action for the agent based on its state."""
    carrying = env.get_carrying(agent)

    if carrying:
        # If carrying a resource, return to base
        return return_to_base(env, agent)
    else:
        # Otherwise, choose a random action
        return env.action_space.sample()

def test_subclass_features(step_limit=10):
    # Initialize the environment with the auction subclass
    env = ForagingEnvironmentWithAuction(num_agents=2, size=20, num_resources=5, fov=2, render_mode="human")
    env.reset(seed=42)

    # Track initial money and battery levels
    initial_money = {agent: 0 for agent in env.agents}
    initial_battery = {agent: env.observe(agent)['battery_level'] for agent in env.agents}

    step_count = 0

    # Simulation loop
    while step_count < step_limit:
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last(observe=False)

            print(f"Step {step_count}")

            if termination or truncation:
                print(f"Agent {agent} has been terminated or truncated.")
                env.step(None)  # Advance the environment to the next agent
                continue  # Skip processing this agent

            action = foraging_behavior(env, agent)  # Determine the agent's action based on behavior
            env.step(action)
            obs = env.observe(agent)  # Ensure the observation corresponds to the correct agent

            # Handle agent battery depletion
            if obs['battery_level'] <= 0:
                print(f"Agent {agent} battery depleted and is now terminated.")
                env.terminations[agent] = True
                env.step(None)  # Advance to the next agent
                continue  # Skip further processing for this agent

            # Check if agent is carrying a resource and has returned to base
            if obs['battery_level'] < initial_battery[agent]:
                print(f"Agent {agent} used battery charge. Current battery level: {obs['battery_level']}")

            if env.get_carrying(agent) and np.array_equal(env.get_agent_location(agent), env.get_home_base_location()):
                print(f"Agent {agent} returned to base with a resource and earned {reward} money.")
                print(f"Agent {agent} received payment. New balance: {obs['money']}")

            if obs['money'] < initial_money[agent]:
                print(f"Agent {agent} purchased battery charge.")

            # Update the tracked values for the next step
            initial_money[agent] = obs['money']
            initial_battery[agent] = obs['battery_level']

            # Print out observations
            print(f"Agent {agent} post-step: Location: {env.get_agent_location(agent)}, Carrying: {env.get_carrying(agent)}, Money: {obs['money']}, Battery Level: {obs['battery_level']}")
            print("-" * 30)

            # Increment the step count and check if the step limit has been reached
            step_count += 1
            if step_count >= step_limit:
                print(f"Reached step limit of {step_limit}. Exiting simulation.")
                break

        # Render the environment after each step
        env.render()
        time.sleep(1)

        # Check if all agents are terminated
        if all(env.terminations.values()):
            print("All agents are terminated. Exiting simulation.")
            break

    env.close()

if __name__ == "__main__":
    test_subclass_features()
