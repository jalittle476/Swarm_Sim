from Foraging_World import ForagingEnvironment

num_agents = 3
env = ForagingEnvironment(num_agents, render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        # You can add action masking logic here if your environment supports it
        # Otherwise, simply sample an action from the environment's action space
        action = env.action_space.sample()


    env.step(action)
env.close()
