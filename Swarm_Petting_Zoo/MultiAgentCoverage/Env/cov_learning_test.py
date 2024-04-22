import numpy as np
import time
from coverage_world import CoverageEnvironment

def q_learning(env, num_episodes, max_steps_per_episode):
    # Define Q-learning parameters
    alpha = 0.1  # Learning rate
    gamma = 0.9  # Discount factor
    epsilon = 0.1  # Exploration rate

    # Initialize Q-table
    num_actions = 4  # Assuming 4 movement directions
    num_states = env.size * env.size  # Assuming each cell in the grid is a state
    q_table = np.zeros((num_states, num_actions))

    # Convert agent's location to state index
    def state_to_index(location):
        return location[0] * env.size + location[1]

    # Convert state index to agent's location
    def index_to_state(index):
        row = index // env.size
        col = index % env.size
        return (row, col)

    # Q-learning algorithm
    for episode in range(num_episodes):
        print(f"Starting episode {episode + 1}")

        # Reset the environment
        obs = env.reset()

        for step in range(max_steps_per_episode):
            print(f"Step {step + 1}")

            # Convert agent's location to state index
            state = state_to_index(env.agent_locations[env.agent_selection])

            # Choose an action using epsilon-greedy strategy
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.choice(num_actions)  # Exploration
            else:
                action = np.argmax(q_table[state])  # Exploitation

            # Take action and observe reward
            obs, reward, terminated, truncated, info = env.step(action)

            # Convert next state to state index
            next_state = state_to_index(env.agent_locations[env.agent_selection])

            # Update Q-value
            q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])

            # Update current state
            state = next_state

            # Check if the episode is terminated
            if terminated:
                break

            # Render the environment
            env.render()

            while env.paused:
                env.render()
                time.sleep(0.1)

        print(f"Episode {episode + 1} finished\n")

def main():
    # Define parameters
    num_episodes = 100
    max_steps_per_episode = 100

    # Create the environment
    env = CoverageEnvironment(num_agents=1, max_steps=max_steps_per_episode, size=10, fov=2, show_fov=True, show_gridlines=True, render_mode="human")

    # Run Q-learning
    q_learning(env, num_episodes, max_steps_per_episode)

if __name__ == "__main__":
    main()
