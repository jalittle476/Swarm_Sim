from foraging_world_with_auctions import ForagingEnvironmentWithAuction
import time

def test_observations():
    # Initialize the environment with the auction subclass
    env = ForagingEnvironmentWithAuction(num_agents=3, size=10, num_resources=2, fov=2, render_mode="human")
    env.reset()

    # Run a few steps in the environment
    for step in range(3):
        print(f"Step {step + 1}")
        for agent in env.agents:
            action = env.action_space.sample()  # Sample a random action for each agent
            env.step(action)
            obs = env.observe(agent)
            print(f"Agent: {agent}")
            print(f"Nearby Agents: {obs['nearby_agents']}")
            print("-" * 30)
        
        # Render the environment after each step
        env.render()
        time.sleep(1)  # Add a short delay to better visualize the steps

    env.close()

if __name__ == "__main__":
    test_observations()
