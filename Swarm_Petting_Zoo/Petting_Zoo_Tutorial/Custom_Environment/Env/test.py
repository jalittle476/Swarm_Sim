from custom_environment import CustomEnvironment
from pettingzoo.test import parallel_api_test 
import pygame

clock = pygame.time.Clock()
env = CustomEnvironment()

# Reset the environment before starting
observations, _ = env.reset()

# Now we'll do a few steps in the environment
for _ in range(1000):
    
    # Choose actions for each agent (just choosing random actions here as an example)
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    
    # Take a step in the environment
    observations, rewards, terminations, truncations, info = env.step(actions)

    # If the game is over for the prisoner, break the loop
    if terminations["prisoner"]:
        print("Prisoner escaped!")
        break

    # Render the current state of the environment
    env.render()

    # Optionally, add a delay
    #pygame.time.wait(100)  # Pause for 100 milliseconds
    clock.tick(10)

# Close the environment when done
env.close()