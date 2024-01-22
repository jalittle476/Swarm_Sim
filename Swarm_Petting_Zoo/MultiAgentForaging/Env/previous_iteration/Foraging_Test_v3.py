from Foraging_World_v3 import ForagingEnvironment
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
        else:
            action = go_home_FOV(env, observation, agent)
            #print(f"{agent} moved to position: {env.get_agent_location(agent)}")
            #print(observation)


        env.step(action)
        
    if all(env.terminations.values()):
        print("All agents are terminated, ending simulation.")
        break

env.close()
