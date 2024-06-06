from Foraging_World import ForagingEnvironment
import numpy as np
import pygame
import random

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def gaussian_sample(mean_direction, std_dev=1.0):
    # `mean_direction` is a 2D vector (dx, dy)
    # Apply Gaussian noise to the mean direction
    sampled_dx = np.random.normal(mean_direction[0], std_dev)
    sampled_dy = np.random.normal(mean_direction[1], std_dev)

    # Convert sampled direction to discrete action
    if abs(sampled_dx) > abs(sampled_dy):
        return 0 if sampled_dx > 0 else 2  # Right or left
    else:
        return 1 if sampled_dy > 0 else 3  # Down or up

def should_return_to_base(battery_level, min_battery_level):
    return battery_level <= min_battery_level

def return_to_base_with_low_battery(agent_location, base_location):
    # Check if the agent is already at the base
    if np.array_equal(agent_location, base_location):
        return None  # No action needed, agent is already at the base

    # Otherwise, calculate the direction towards the base
    direction = np.array(base_location) - np.array(agent_location)
    return gaussian_sample(direction, std_dev=0.1)

def communication_behavior(agent, observation, env):
    # Randomly choose to send a message or not
    # For example, 10% chance to send a message
    communication_action = 1  # Send a message
    return communication_action

def foraging_behavior(env, observation, agent, std_dev=0.5):
    carrying = env.get_carrying(agent)
    visible_resources = observation["resources"]
    agent_location = observation["agent_location"]
    base_location = observation["home_base"]
    base_proximity_threshold = 1  # Define how close is considered 'near' the base

    # Calculate distance to base
    distance_to_base = np.linalg.norm(np.array(base_location) - np.array(agent_location))

    if carrying:
        mean_direction = base_location - agent_location
        return gaussian_sample(mean_direction, std_dev)
    else:
        if visible_resources:
            distances = [manhattan_distance(agent_location, resource) for resource in visible_resources]
            nearest_resource = visible_resources[np.argmin(distances)]
            mean_direction = nearest_resource - agent_location
        else:
            mean_direction = np.random.normal(0, std_dev, 2)

        # Directly avoid the base if near and not carrying
        if distance_to_base <= base_proximity_threshold and not carrying:
            mean_direction = agent_location - base_location  # Direct away from base

        new_action = gaussian_sample(mean_direction, std_dev)
        return new_action
    

env = ForagingEnvironment(num_agents=25, size = 25, render_mode="human", show_fov = True, draw_numbers=False, num_resources=200)
env.reset(seed=42)
battery_safety_margin = 0 # Robot's will not assume perfect knowlege of their battery levels 
# Define the maximum distance to the base as a threshold
min_battery_level = env.size
base_location = [env.size // 2, env.size // 2]  # Assuming the base is at the center
exit_simulation = False  # Flag to indicate whether to exit the simulation



for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                env.paused = not env.paused
            elif event.key == pygame.K_q:  # Check if the 'Q' key is pressed
                exit_simulation = True      # Set the flag to exit the simulation

    if exit_simulation:  # Check the flag before continuing the simulation
        print("Exiting simulation.")
        break
    
    if not env.paused:
        if termination or truncation:
            combined_action = (None, None)
        elif should_return_to_base(observation["battery_level"], min_battery_level):
            movement_action = return_to_base_with_low_battery(observation["agent_location"], base_location)
            communication_action = communication_behavior(agent, observation, env)
            combined_action = (movement_action, communication_action)
        else:
            movement_action = foraging_behavior(env, observation, agent)
            communication_action = communication_behavior(agent, observation, env)
            combined_action = (movement_action, communication_action)

        env.step(combined_action)
        
    # Check if all agents are terminated, then pause
    if all(env.terminations.values()):
        print("All agents are terminated. Press any key to exit.")

        # Refresh/render the screen one last time before pausing
        if env.render_mode == "human":
            env._render()

        # Wait for a key press to exit
        waiting_for_input = True
        while waiting_for_input:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    waiting_for_input = False
                elif event.type == pygame.QUIT:
                    waiting_for_input = False
                    exit_simulation = True

        if exit_simulation:
            break

env.close()

