import numpy as np
import pygame
from coverage_world_transitions import CoverageEnvironment
import matplotlib.pyplot as plt

# Initialize the environment
env = CoverageEnvironment(num_agents=2, max_steps=1000, render_mode='human', size=10, seed=123)

# Q-Learning setup
num_states = env.size * env.size
num_actions = env.action_space.n
q_tables = {agent: {} for agent in env.possible_agents}
alpha = 0.1
gamma = 0.99
num_episodes = 50  # Number of episodes to run
rewards_per_episode = []

epsilon_start = 1.0
epsilon_end = 0.3
epsilon_decay = 0.99
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

    # Convert to a hash or another form of unique identifier if needed
    state = hash(state_features)  # Using hash to ensure a unique state identifier

    return state

def choose_action(agent, state):
    global epsilon
    # Ensure the current state is in the Q-table
    if state not in q_tables[agent]:
        q_tables[agent][state] = np.zeros(num_actions)
    
    # Epsilon-greedy policy for action selection
    if np.random.rand() < epsilon:
        action = np.random.randint(num_actions)
    else:
        action = np.argmax(q_tables[agent][state])
    
    epsilon = max(epsilon * epsilon_decay, epsilon_end)  # Decaying epsilon
    return action

def update_q_table(agent, state, action, reward, next_state):
    # Initialize next state in Q-table if it doesn't exist
    if next_state not in q_tables[agent]:
        q_tables[agent][next_state] = np.zeros(num_actions)
    
    # Q-Learning update formula
    best_next_action = np.argmax(q_tables[agent][next_state])
    td_target = reward + gamma * q_tables[agent][next_state][best_next_action]
    td_error = td_target - q_tables[agent][state][action]
    q_tables[agent][state][action] += alpha * td_error

def run_environment(env):
    total_reward = 0
    observations = env.reset()
    done = False

    while not done:
        if env.render_mode == 'human':
            env.render()

        for agent in env.agent_iter():
            current_state = get_state(observations[agent])
            action = choose_action(agent, current_state)
            next_observation, reward, terminated, truncation, info = env.step(action)
            next_state = get_state(next_observation)

            update_q_table(agent, current_state, action, reward, next_state)
            observations[agent] = next_observation

            total_reward += reward
            done = terminated or truncation

            if terminated or truncation:
                break

    return total_reward


# Run multiple episodes
for episode in range(num_episodes):
    episode_reward = run_environment(env)
    rewards_per_episode.append(episode_reward)
    print(f"Episode {episode + 1}: Total Reward = {episode_reward}")
    env.reset()

pygame.quit()  # Properly close Pygame

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(rewards_per_episode, label='Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Learning Progress Over Episodes')
plt.legend()
plt.show()
