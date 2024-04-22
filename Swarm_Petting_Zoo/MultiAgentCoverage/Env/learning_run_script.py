import pygame
import numpy as np
from coverage_world import CoverageEnvironment

# Initialize the environment
env = CoverageEnvironment(num_agents=5, max_steps=10000, render_mode='human', size=10, seed=123)

# Q-Learning setup
num_states = env.size * env.size
num_actions = 4  # Assuming 4 possible actions: up, down, left, right
q_tables = {f'agent_{i}': np.zeros((num_states, num_actions)) for i in range(env.num_agents)}
alpha = 0.1
gamma = 0.99
epsilon = 0.1

def choose_action(agent, state):
    if np.random.rand() < epsilon:
        return np.random.randint(num_actions)
    else:
        return np.argmax(q_tables[agent][state])

def update_q_table(agent, state, action, reward, next_state):
    best_next_action = np.argmax(q_tables[agent][next_state])
    td_target = reward + gamma * q_tables[agent][next_state][best_next_action]
    td_error = td_target - q_tables[agent][state][action]
    q_tables[agent][state][action] += alpha * td_error

def get_state(observation):
    # Assuming state is defined by the agent's location for simplicity
    return observation['agent_location'][0] * env.size + observation['agent_location'][1]

def run_environment(env):
    observations = env.reset()
    done = False
    step_count = 0

    while not done:
        if env.render_mode == 'human':
            env.render()

        for agent in env.agent_iter():  # Iterate through agents
            current_state = get_state(observations[agent])
            action = choose_action(agent, current_state)
            next_observation, reward, terminated, truncation, info = env.step(action)
            next_state = get_state(next_observation)

            update_q_table(agent, current_state, action, reward, next_state)
            observations[agent] = next_observation

            print(f"Agent {agent}: Step {step_count}, Action {action}, Reward {reward}")
            done = terminated or truncation

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
