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

def test_subclass_features():
    # Initialize the environment with the auction subclass
    env = ForagingEnvironmentWithAuction(num_agents=1, size=10, num_resources=5, fov=2, render_mode="human")
    env.reset(seed=42)

    # Test for a few steps to observe agent behaviors
    for step in range(5):
        print(f"Step {step + 1}")
        for agent in env.agents:
            action = foraging_behavior(env, agent)  # Determine the agent's action based on behavior
            env.step(action)
            obs = env.observe(agent)
            
            # Print out observations
            print(f"Agent: {agent}")
            print(f"Nearby Agents: {obs['nearby_agents']}")
            print(f"Carrying: {env.get_carrying(agent)}")
            print(f"Money: {obs['money']}")
            print(f"Battery Level: {obs['battery_level']}")
            print(f"Resources: {obs['resources']}")
            print("-" * 30)
        
        # Render the environment after each step
        env.render()
        time.sleep(1)

    env.close()

if __name__ == "__main__":
    test_subclass_features()
