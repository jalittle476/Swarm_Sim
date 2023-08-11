from Foraging_World import ForagingEnvironment
import numpy as np
import pygame

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def go_home_FOV(env, observation, agent):
    carrying = env.get_carrying(agent)
    visible_resources = observation["resources"]

    if not carrying:
        if visible_resources:
            distances = [manhattan_distance(observation["agent_location"], resource) for resource in visible_resources]
            nearest_resource = visible_resources[np.argmin(distances)]
            dx, dy = nearest_resource - observation["agent_location"]
        else:
            return env.action_space.sample()
    else:
        dx, dy = observation["home_base"] - observation["agent_location"]

    if abs(dx) > abs(dy):
        return 0 if dx > 0 else 2
    else:
        return 1 if dy > 0 else 3

num_agents = 3
env = ForagingEnvironment(num_agents, render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                env.paused = not env.paused

    if not env.paused:
        if termination or truncation:
            action = None
        else:
            action = go_home_FOV(env, observation, agent)
            #print(f"{agent} moved to position: {env.get_agent_location(agent)}")

        env.step(action)

env.close()
