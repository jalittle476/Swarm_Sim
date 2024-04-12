import numpy as np
import time
from coverage_world import CoverageEnvironment

def main():
    # Define parameters
    num_episodes = 10
    max_steps_per_episode = 100

    # Create the environment
    env = CoverageEnvironment(num_agents=2, size=25, fov=2, show_fov=True, show_gridlines=True, render_mode="human")

    for episode in range(num_episodes):
        print(f"Starting episode {episode + 1}")

        # Reset the environment
        obs = env.reset()

        for step in range(max_steps_per_episode):
            print(f"Step {step + 1}")

            for agent in env.agents:
                # Select a random action for each agent
                action = (np.random.choice(4), np.random.choice(2))  # Assuming 4 movement directions and 2 communication actions
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Check if the episode is terminated
                if terminated:
                    break

            # Render the environment
            env.render()

            while env.paused:
                env.render()
                time.sleep(0.1)

            if terminated:
                print(f"Terminated at step {step + 1}")
                break

        print(f"Episode {episode + 1} finished\n")

if __name__ == "__main__":
    main()
