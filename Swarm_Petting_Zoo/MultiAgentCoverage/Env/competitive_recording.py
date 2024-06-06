import numpy as np
import pygame
from competitive_coverage_world import CoverageEnvironment
import matplotlib.pyplot as plt
import os

# Initialize the environment
env = CoverageEnvironment(num_agents=2, max_steps=1000, render_mode='human', size=10, seed=123)

# Q-Learning setup
num_states = env.size * env.size
num_actions = env.action_space.n
q_table = {agent: {} for agent in env.agents}
alpha = 0.1
gamma = 0.99
num_episodes = 50
rewards_per_episode = []

epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 0.999
epsilon = epsilon_start

def get_state(observation):
    agent_pos = observation['agent_location']
    local_map = observation['local_map']
    other_agents_positions = observation['other_agents_positions']
    explored_tiles_binary = (local_map > 0).astype(int).flatten()
    other_agents_positions_binary = [1 if np.any(pos) else 0 for pos in other_agents_positions.values()]
    state_features = (agent_pos[0], agent_pos[1]) + tuple(explored_tiles_binary) + tuple(other_agents_positions_binary)
    return state_features

def choose_action(agent, state):
    global epsilon
    if state not in q_table[agent]:
        q_table[agent][state] = np.zeros(num_actions)
    if np.random.rand() < epsilon:
        action = np.random.randint(num_actions)
    else:
        action = np.argmax(q_table[agent][state])
    epsilon = max(epsilon * epsilon_decay, epsilon_end)
    return action

def update_q_table(agent, state, action, reward, next_state):
    if next_state not in q_table[agent]:
        q_table[agent][next_state] = np.zeros(num_actions)
    best_next_action = np.argmax(q_table[agent][next_state])
    td_target = reward + gamma * q_table[agent][next_state][best_next_action]
    td_error = td_target - q_table[agent][state][action]
    q_table[agent][state][action] += alpha * td_error

def run_environment(env, record=False, episode_num=None,render=False):
    total_rewards = {agent: 0 for agent in env.agents}
    observations = env.reset()
    done = False
    frame_count = 0

    while not done:
        if render and env.render_mode == 'human':
            env.render()
            if record and frame_count % 10 == 0:  # Save every 10th frame
                if not os.path.exists(f'recordings/comp_episode_{episode_num}'):
                    os.makedirs(f'recordings/comp_episode_{episode_num}')
                pygame.image.save(env.window, f'recordings/comp_episode_{episode_num}/frame_{frame_count//10}.png')

        actions = {agent: choose_action(agent, get_state(observations[agent])) for agent in env.agents}
        next_observations, rewards, terminated, truncation, info = env.step(actions)

        for agent in env.agents:
            current_state = get_state(observations[agent])
            action = actions[agent]
            reward = rewards[agent]
            next_state = get_state(next_observations[agent])
            update_q_table(agent, current_state, action, reward, next_state)
            observations[agent] = next_observations[agent]
            total_rewards[agent] += reward

        frame_count += 1
        done = np.any(terminated) or np.any(truncation)

    return total_rewards

# Run the environment
for episode in range(num_episodes):
    if episode == 1 or episode == num_episodes - 1:  # Record only the first and last episodes
        total_reward = run_environment(env, record=True, episode_num=episode,render=True)
    else:
        total_reward = run_environment(env)
    rewards_per_episode.append(total_reward)
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

env.close()

# Calculate the total reward for each episode
total_rewards_per_episode = [sum(reward.values()) for reward in rewards_per_episode]

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(total_rewards_per_episode, label='Total Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Learning Progress Over Episodes')
plt.legend()
plt.show()