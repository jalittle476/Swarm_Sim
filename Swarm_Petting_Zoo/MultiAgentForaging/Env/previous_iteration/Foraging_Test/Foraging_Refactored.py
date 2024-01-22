from Foraging_World import ForagingEnvironment
import numpy as np
import pygame

# Constants for actions
MOVE_RIGHT = 0
MOVE_UP = 1
MOVE_LEFT = 2
MOVE_DOWN = 3

def manhattan_distance(a, b):
    """Calculate the Manhattan distance between two points."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def find_nearest_resource(agent_location, resources):
    """Find the nearest resource to the agent."""
    nearest_resource = None
    min_distance = float('inf')
    for resource in resources:
        distance = manhattan_distance(agent_location, resource)
        if distance < min_distance:
            min_distance = distance
            nearest_resource = resource
    return nearest_resource

def choose_action(env, observation, agent):
    """Decide the action for the agent based on the current observation."""
    carrying = env.get_carrying(agent)
    visible_resources = observation["resources"]

    if carrying:
        return move_towards(observation["home_base"], observation["agent_location"])
    elif visible_resources:
        nearest_resource = find_nearest_resource(observation["agent_location"], visible_resources)
        return move_towards(nearest_resource, observation["agent_location"])
    else:
        return env.action_space.sample()

def move_towards(target, current_location):
    """Determine the action to move from current_location towards target."""
    dx, dy = target - current_location
    if abs(dx) > abs(dy):
        return MOVE_RIGHT if dx > 0 else MOVE_LEFT
    else:
        return MOVE_UP if dy > 0 else MOVE_DOWN

def handle_events(env):
    """Handle Pygame events, including pausing and exiting the simulation."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:  # Allows the window to be closed manually
            return True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                env.paused = not env.paused
            elif event.key == pygame.K_q:
                return True
    return False


def run_simulation():
    """Run the main simulation loop."""
    env = ForagingEnvironment(num_agents=100, size=25, render_mode="human", show_fov=False, draw_numbers=False, num_resources=200)
    env.reset(seed=42)

    while True:
        observation, reward, termination, truncation, info = env.last()
        exit_simulation = handle_events(env)

        if exit_simulation or all(env.terminations.values()):
            break

        if not env.paused and not (termination or truncation):
            action = choose_action(env, observation, env.agent_selection)
            env.step(action)

    print("Ending simulation.")
    env.close()

# Run the simulation
run_simulation()
