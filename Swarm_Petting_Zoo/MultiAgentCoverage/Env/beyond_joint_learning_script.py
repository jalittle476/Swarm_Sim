import numpy as np
import pygame
from coverage_world_joint import CoverageEnvironment
import matplotlib.pyplot as plt

# Initialize the environment
env = CoverageEnvironment(num_agents=5, max_steps=100, render_mode='human', size=10, seed=123)

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

    # Convert local map to a binary format: 0 for unexplored, 1 for explored
    # Assuming '0' indicates unexplored and '1' indicates explored in your local_map setup
    explored_tiles_binary = (local_map > 0).astype(int).flatten()

    # Combine the agent's position and the binary state of the local map into one tuple
    state_features = (agent_pos[0], agent_pos[1]) + tuple(explored_tiles_binary)

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
    
    epsilon = max(epsilon * epsilon_decay, epsilon_end)  # Decaying epsilon
    return action

def update_q_table(agent, state, action, reward, next_state):
    # Initialize next state in Q-table for this agent if it doesn't exist
    if next_state not in q_table[agent]:
        q_table[agent][next_state] = np.zeros(num_actions)
    
    # Q-Learning update formula
    best_next_action = np.argmax(q_table[agent][next_state])
    td_target = reward + gamma * q_table[agent][next_state][best_next_action]
    td_error = td_target - q_table[agent][state][action]
    q_table[agent][state][action] += alpha * td_error

def run_environment(env):
    # print("Running environment...")  # Debug print
    total_reward = 0
    observations = env.reset()
    done = False
    episode = 0

    while not done:
        # print("Starting loop...")  # Debug print
        if env.render_mode == 'human':
            env.render()

        actions = {agent: choose_action(agent, get_state(observations[agent])) for agent in env.agents}
        # print(f"Calling step function with actions {actions}")  # Debug print
        next_observations, rewards, terminated, truncation, info = env.step(actions)

        for agent in env.agents:
            # print(f"Agent {agent} turn")  # Debug print
            current_state = get_state(observations[agent])
            action = actions[agent]
            reward = rewards[agent]
            next_state = get_state(next_observations[agent])
            update_q_table(agent, current_state, action, reward, next_state)
            observations[agent] = next_observations[agent]

            total_reward += sum(rewards.values())
            done = terminated or truncation

            if done:
                break
    episode = episode + 1                
    return total_reward

# Run the environment
for episode in range(num_episodes):
    total_reward = run_environment(env)
    rewards_per_episode.append(total_reward)
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

pygame.quit()  # Properly close Pygame

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(rewards_per_episode, label='Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Learning Progress Over Episodes')
plt.legend()
plt.show()
