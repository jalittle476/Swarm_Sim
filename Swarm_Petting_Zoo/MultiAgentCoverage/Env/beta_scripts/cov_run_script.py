import pygame
from coverage_world import CoverageEnvironment

# Initialize the environment with desired settings
env = CoverageEnvironment(num_agents=2, max_steps=1000, render_mode='human', size=10, seed=123)

def run_environment(env):
    observations = env.reset()
    done = False
    step_count = 0

    while not done:
        if env.render_mode == 'human':
            env.render()

        for agent in env.agent_iter():  # Iterate through agents
            action = env.action_space.sample()  # Randomly sample an action
            observation, reward, terminated, truncation, info = env.step(action)
            done = terminated or truncation

            print(f"Agent {agent}: Step {step_count}, Action {action}, Reward {reward}")
            if terminated:
                print(f"Agent {agent} terminated.")

            if truncation:
                print("Max steps reached, truncating episode.")

            step_count += 1

            if done:
                break

        if env.paused:
            continue  # If paused, skip the rest until unpaused

    print("Simulation complete.")
    pygame.quit()  # Properly close Pygame

# Run the environment simulation
run_environment(env)
