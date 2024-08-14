from foraging_world_with_auctions import ForagingEnvironmentWithAuction
import numpy as np
import pygame

# Foraging Test with Auction/Trading
# Agents perform basic foraging and retrieval algorithms, with added auction functionality

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def gaussian_sample(mean_direction, std_dev=1.0):
    sampled_dx = np.random.normal(mean_direction[0], std_dev)
    sampled_dy = np.random.normal(mean_direction[1], std_dev)

    if abs(sampled_dx) > abs(sampled_dy):
        return 0 if sampled_dx > 0 else 2  # Right or left
    else:
        return 1 if sampled_dy > 0 else 3  # Down or up

def should_return_to_base(battery_level, min_battery_level):
    return battery_level <= min_battery_level

def return_to_base_with_low_battery(agent_location, base_location):
    if np.array_equal(agent_location, base_location):
        return None
    direction = np.array(base_location) - np.array(agent_location)
    return gaussian_sample(direction, std_dev=0.1)

def foraging_behavior(env, observation, agent, std_dev=0.5):
    carrying = env.get_carrying(agent)
    visible_resources = observation["resources"]
    agent_location = observation["agent_location"]
    base_location = observation["home_base"]
    base_proximity_threshold = 5

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

        if distance_to_base <= base_proximity_threshold and not carrying:
            mean_direction = agent_location - base_location

        return gaussian_sample(mean_direction, std_dev)

env = ForagingEnvironmentWithAuction(num_agents=5, size=20, render_mode="human", show_fov=False, draw_numbers=False, num_resources=200)
env.reset(seed=42)
battery_safety_margin = 0
min_battery_level = env.size
base_location = [env.size // 2, env.size // 2]
exit_simulation = False

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                env.paused = not env.paused
            elif event.key == pygame.K_q:
                exit_simulation = True

    if exit_simulation:
        print("Exiting simulation.")
        break
    
    if not env.paused:
        if termination or truncation:
            action = None
        elif should_return_to_base(observation["battery_level"], min_battery_level):
            action = return_to_base_with_low_battery(observation["agent_location"], base_location)
        else:
            action = foraging_behavior(env, observation, agent)

        env.step(action)
        
    if all(env.terminations.values()):
        print("All agents are terminated. Press any key to exit.")

        if env.render_mode == "human":
            env._render()

        waiting_for_input = True
        while waiting_for_input:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN or event.type == pygame.QUIT:
                    waiting_for_input = False
                    exit_simulation = True

        if exit_simulation:
            break

env.close()
