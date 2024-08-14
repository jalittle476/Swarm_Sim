from foraging_world_with_auctions import ForagingEnvironmentWithAuction
import time
import numpy as np

def test_subclass_features():
    # Initialize the environment with the auction subclass
    env = ForagingEnvironmentWithAuction(num_agents=1, size=10, num_resources=5, fov=2, render_mode="human")
    env.reset(seed=42)

    # Test for a few steps to observe agent behaviors
    for step in range(5):
        print(f"Step {step + 1}")
        for agent in env.agents:
            action = env.action_space.sample()  # Sample a random action for each agent
            env.step(action)
            obs = env.observe(agent)
            
            # Print out observations
            print(f"Agent: {agent}")
            #print(f"Nearby Agents: {obs['nearby_agents']}")
            print(f"Carrying: {env.get_carrying(agent)}")
            print(f"Money: {obs['money']}")
            print(f"Battery Level: {obs['battery_level']}")
            #print(f"Resources: {obs['resources']}")
            print("-" * 30)
        
        # Render the environment after each step
        env.render()
        time.sleep(1)

    env.close()

if __name__ == "__main__":
    test_subclass_features()
