import numpy as np
import pygame
from coordinated_coverage_world import CoverageEnvironment
import matplotlib.pyplot as plt
import os

# Initialize the environment
env = CoverageEnvironment(num_agents=2, max_steps=1000, render_mode='human', size=10, seed=123)

# Q-Learning setup
num_states = env.size * env.size
num_actions = env.action_space.n
q_table = {agent: {} for agent in env.agents}  # Separate Q-table for each agent
alpha = 0.1
gamma = 0.99
num_episodes = 50  # Number of episodes to run
rewards_per_episode = []

epsilon_start = 1.0
epsilon_end = 0.3
epsilon_decay = 0.999
epsilon = epsilon_start

def get_state(observation):
    # Extract the agent's grid position
    agent_pos = observation['agent_location']
    local_map = observation['local_map']
    other_agents_positions = observation['other_agents_positions']

    # Convert local map and other agents positions to a binary format
    explored_tiles_binary = (local_map > 0).astype(int).flatten()
    other_agents_positions_binary = [1 if np.any(pos) else 0 for pos in other_agents_positions.values()]
    # Combine the agent's position, the binary state of the local map, and the binary state of other agents positions into one tuple
    state_features = (agent_pos[0], agent_pos[1]) + tuple(explored_tiles_binary) + tuple(other_agents_positions_binary)

    return state_features

def choose_action(agent, state):
    global epsilon
    # Ensure the current state is in the Q-table for this agent
    if state not in q_table[agent]:
        q_table[agent][state] = np.zeros(num_actions)
    
    # Epsilon-greedy policy for action selection
    if np.random.rand() < epsilon:
        action = np.random.randint(num_actions)
    else:
        action = np.argmax(q_table[agent][state])
    
    # Get the Q-value for the chosen action
    q_value = q_table[agent][state][action]

    epsilon = max(epsilon * epsilon_decay, epsilon_end)  # Decaying epsilon
    return action, q_value

def update_q_table(agent, state, action, reward, next_state):
    # Initialize next state in Q-table for this agent if it doesn't exist
    if next_state not in q_table[agent]:
        q_table[agent][next_state] = np.zeros(num_actions)
    
    # Q-Learning update formula
    best_next_action = np.argmax(q_table[agent][next_state])
    td_target = reward + gamma * q_table[agent][next_state][best_next_action]
    td_error = td_target - q_table[agent][state][action]
    q_table[agent][state][action] += alpha * td_error

def run_environment(env, record=False, episode_num=0, render=False):
    total_reward = 0
    observations = env.reset()
    done = False
    frame_count = 0

    # Create a directory to store the frames
    if record:
        os.makedirs(f'recordings/episode_{episode_num}', exist_ok=True)

    while not done:
        if render and env.render_mode == 'human':
            env.render()

            # Save the screen surface to an image file for the first and last episode
            if record and frame_count % 10 == 0:
                pygame.image.save(env.window, f'recordings/episode_{episode_num}/frame_{frame_count//10}.png')
            frame_count += 1

        actions = {}
        for agent in env.agents:
            state = get_state(observations[agent])
            action, q_value = choose_action(agent, state)  # Get the chosen action and Q-value
            actions[agent] = action
            observations[agent]['action'] = action  # Add action to the observation
            observations[agent]['q_value'] = q_value  # Add Q-value to the observation

        next_observations, rewards, terminated, truncation, info = env.step(actions)

        # Calculate the global reward as the sum of the individual rewards
        global_reward = sum(rewards.values())
        total_reward += global_reward

        for agent in env.agents:
            current_state = get_state(observations[agent])
            action = actions[agent]
            next_state = get_state(next_observations[agent])
            update_q_table(agent, current_state, action, global_reward, next_state)  # Update Q-table with the global reward
            observations[agent] = next_observations[agent]

        done = np.any(terminated) or np.any(truncation)  # Use np.any() to evaluate arrays in a boolean context

        if done:
            break
    return total_reward

# Run the environment
for episode in range(num_episodes):
    if episode == 1 or episode == num_episodes - 1:  # Record and render only the first and last episodes
        total_reward = run_environment(env, record=True, episode_num=episode, render=True)
    else:
        total_reward = run_environment(env)
    rewards_per_episode.append(total_reward)
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

env.close()
