from pettingzoo.classic import rps_v2
import pygame

env = rps_v2.env(render_mode="human")

running = True
while running:
    env.reset(seed=42)
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            action = env.action_space(agent).sample() # this is where you would insert your policy

        env.step(action)
env.close()

