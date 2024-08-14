
from foraging_world_with_communication_v2 import ForagingEnvironmentWithCommunication
import numpy as np

def test_environment():
    # Initialize the environment
    env = ForagingEnvironmentWithCommunication(num_agents=3, size=5, num_resources=2)
    env.reset()

    # Run a few steps in the environment
    for step in range(5):
        print(f"Step {step + 1}")
        for agent in env.agents:
            action = env.action_space.sample()  # Sample a random action for each agent
            env.step(action)
            obs = env.observe(agent)
            print(f"Agent: {agent}")
            print(f"Messages: {obs['messages']}")
            print("-" * 30)

        if all(env.terminated.values()) or all(env.truncated.values()):
            print("All agents are done.")
            break

    env.close()

if __name__ == "__main__":
    test_environment()
