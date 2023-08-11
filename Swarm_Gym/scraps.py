def step(self, action):
    # Map the action (element of {0,1,2,3}) to the direction we walk in
    direction = self._action_to_direction[action]
    # We use `np.clip` to make sure we don't leave the grid
    self._agent_location = np.clip(
        self._agent_location + direction, 0, self.size - 1
    )

    # Initialize reward and termination flag
    reward = 0
    terminated = False

    # Check if the agent is on a resource location, and is not carrying a resource
    for i in range(len(self._resources_location)):
        if np.array_equal(self._agent_location, self._resources_location[i]) and not self._carrying:
            self._carrying = True  # Now the agent is carrying a resource
            # Remove the resource from the environment
            self._resources_location = np.delete(self._resources_location, i, axis=0)
            break

    # Check if the agent has returned to the base with a resource
    if np.array_equal(self._agent_location, self._home_base_location) and self._carrying:
        reward = 1  # The agent gets a reward for delivering the resource
        self._carrying = False  # The agent is no longer carrying a resource

    # The episode is done if all resources have been collected
    if len(self._resources_location) == 0 and not self._carrying:
        terminated = True

    observation = self._get_obs()
    info = self._get_info()

    if self.render_mode == "human":
        self._render_frame()

    return observation, reward, terminated, info

# def random_policy(env):
#     return env.action_space.sample()

# def go_home_policy(env):
#     carrying = env.get_carrying()
#     if not carrying:
#         # If the agent is not carrying a resource, act randomly
#         return env.action_space.sample()
#     else:
#         # If the agent is carrying a resource, choose the action that reduces the distance to the home base
#         dx, dy = env.get_home_base_location() - env.get_agent_location()
#         if abs(dx) > abs(dy):
#             # Move in the x direction
#             return 0 if dx > 0 else 2
#         else:
#             # Move in the y direction
#             return 1 if dy > 0 else 3
        
# def go_home_FOV(env, observation):
#     carrying = env.get_carrying()
#     visible_resources = observation["resources"]
#     if not carrying:
#         # If the agent is not carrying a resource, try to find one
#         if visible_resources:
#             # If there are visible resources, choose the action that reduces the distance to the nearest one
#             dx, dy = visible_resources[0] - observation["agent_location"]  # Assumes the first visible resource is the nearest one
#             if abs(dx) > abs(dy):
#                 # Move in the x direction
#                 return 0 if dx > 0 else 2
#             else:
#                 # Move in the y direction
#                 return 1 if dy > 0 else 3
#         else:
#             # If no resources are visible, act randomly
#             return env.action_space.sample()
#     else:
#         # If the agent is carrying a resource, choose the action that reduces the distance to the home base
#         dx, dy = observation["home_base"] - observation["agent_location"]
#         if abs(dx) > abs(dy):
#             # Move in the x direction
#             return 0 if dx > 0 else 2
#         else:
#             # Move in the y direction
#             return 1 if dy > 0 else 3

# def go_home_FOV(env, observation):
#     carrying = env.get_carrying()
#     visible_resources = observation["resources"]
    
#     if not carrying:
#         if visible_resources:
#             # Find the nearest resource by computing the Manhattan distances to each visible resource.
#             distances = [manhattan_distance(observation["agent_location"], resource) for resource in visible_resources]
#             nearest_resource = visible_resources[np.argmin(distances)]
            
#             # Choose the action that reduces the distance to the nearest resource.
#             dx, dy = nearest_resource - observation["agent_location"]
#             if abs(dx) > abs(dy):
#                 # Move in the x direction
#                 return 0 if dx > 0 else 2
#             else:
#                 # Move in the y direction
#                 return 1 if dy > 0 else 3
#         else:
#             # If no resources are visible, act randomly.
#             return env.action_space.sample()
#     else:
#         # If the agent is carrying a resource, choose the action that reduces the distance to the home base.
#         dx, dy = observation["home_base"] - observation["agent_location"]
#         if abs(dx) > abs(dy):
#             # Move in the x direction
#             return 0 if dx > 0 else 2
#         else:
#             # Move in the y direction
#             return 1 if dy > 0 else 3


# observation, _ = env.reset()
# done = False