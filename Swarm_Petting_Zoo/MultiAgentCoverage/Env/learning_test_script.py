import numpy as np
import time
from coverage_world import CoverageEnvironment

def location_to_state(location, size):
    # Converts a 2D grid location to a single integer state
    return location[0] * size + location[1]

# Define parameters
num_episodes = 1000
max_steps_per_episode = 100
num_agents = 2
size = 25
num_actions = 4 * 2  # 4 movement directions and 2 communication actions
epsilon = 0.1
alpha = 0.1
gamma = 0.9

# Create the environment
env = CoverageEnvironment(num_agents=num_agents, size=size, fov=2, show_fov=True, show_gridlines=True, render_mode="human")

# Initialize Q-tables for each agent
num_states = size * size
Q_tables = {agent: np.zeros((num_states, num_actions)) for agent in env.possible_agents}

for episode in range(num_episodes):
    print(f"Starting episode {episode + 1}")
    env.reset()
    
    total_reward = 0

    for step in range(max_steps_per_episode):
        print(f"Step {step + 1}")

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            current_state = location_to_state(observation["agent_location"], size)

             # Debug: Print observations before taking action
            print("Observations before action:", observation)

            # Action selection (epsilon-greedy)
            if np.random.random() < epsilon:
                action = np.random.choice(num_actions)
            else:
                action = np.argmax(Q_tables[agent][current_state])

            # Execute action
            _, reward, terminated, _, _ = env.step(action)  # Update observations
            next_obs = env.observe(agent)  # Get new observations
            next_state = location_to_state(next_obs["agent_location"], size)
            
             # Debug: Print next observations
            print(f"Next observations for {agent}:", next_obs)

            # Update Q-table
            Q_tables[agent][current_state][action] += alpha * (reward + gamma * np.max(Q_tables[agent][next_state]) - Q_tables[agent][current_state][action])
            total_reward += reward

            if terminated:
                break

        env.render()
        while env.paused:
            env.render()
            time.sleep(0.1)

        if terminated:
            print(f"Terminated at step {step + 1}")
            break

    print(f"Episode {episode + 1} finished. Total reward: {total_reward}")
