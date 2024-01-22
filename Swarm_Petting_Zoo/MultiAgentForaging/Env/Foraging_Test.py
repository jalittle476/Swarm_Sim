from Foraging_World import ForagingEnvironment
import numpy as np
import pygame

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

def should_return_to_base(battery_level, max_distance):
    return battery_level <= max_distance

def calculate_action_towards_base(agent_location, base_location):
    # Logic to calculate the action that moves the agent towards the base
    direction = np.array(base_location) - np.array(agent_location)
    return gaussian_sample(direction, std_dev=0.1)  # You might adjust the std_dev as needed

def go_home_FOV(env, observation, agent, std_dev = 0.5):
    carrying = env.get_carrying(agent)
    visible_resources = observation["resources"]
    agent_location = observation["agent_location"]

    if not carrying:
        if visible_resources:
            distances = [manhattan_distance(observation["agent_location"], resource) for resource in visible_resources]
            nearest_resource = visible_resources[np.argmin(distances)]
            
            mean_direction = nearest_resource - agent_location
        else:
            mean_direction = np.array([0,0])
    else:
        mean_direction = observation["home_base"] - agent_location
        
    return gaussian_sample(mean_direction,std_dev)

env = ForagingEnvironment(num_agents=20, size = 25, render_mode="human", show_fov = False, draw_numbers=False, num_resources=200)
env.reset(seed=42)
battery_safety_margin = 5 # Robot's will not assume perfect knowlege of their battery levels 
# Define the maximum distance to the base as a threshold
max_distance_to_base = (env.size // 2) * 2 - battery_safety_margin
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
            action = None
        elif should_return_to_base(observation["battery_level"], max_distance_to_base):
            action = calculate_action_towards_base(observation["agent_location"], base_location)
        else:
            action = go_home_FOV(env, observation, agent)



        env.step(action)
        
    if all(env.terminations.values()):
        print("All agents are terminated, ending simulation.")
        break

env.close()
