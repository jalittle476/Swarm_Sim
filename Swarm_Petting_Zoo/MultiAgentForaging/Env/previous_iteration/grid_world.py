import gym_examples
import gym
from gym import spaces
import pygame
import numpy as np


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=3, seed=255, num_resources = 5, fov = 2):
        self.np_random = np.random.default_rng(seed)
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self._carrying = False
        self._battery_level = 100
        self.paused = False
        self.num_resources = num_resources
        self.fov = fov
        

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent_location": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "home_base": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "resources": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "battery_level": spaces.Box(0, size - 1, shape=(2,), dtype=int)
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        # Define the agent's field of view (FOV)
        fov = self.fov

        # Get the coordinates of the top-left and bottom-right corners of the FOV
        # tl_y, tl_x = max(0, self._agent_location[0] - fov), max(0, self._agent_location[1] - fov)
        # br_y, br_x = min(self.size, self._agent_location[0] + fov + 1), min(
        # self.size, self._agent_location[1] + fov + 1)
        # Calculate the boundaries for the FOV
        # Calculate the boundaries for the FOV
        tl_y = max(0, self._agent_location[0] - fov)
        tl_x = max(0, self._agent_location[1] - fov)
        br_y = min(self.size, self._agent_location[0] + fov + 1)
        br_x = min(self.size, self._agent_location[1] + fov + 1)


        # Check each resource
        visible_resources = []
        for resource_location in self._resources_location:
            # If the resource is within the FOV, add it to the list of visible resources
            if tl_y <= resource_location[0] < br_y and tl_x <= resource_location[1] < br_x:
                visible_resources.append(resource_location)

        return {
        "agent_location": self._agent_location,
        "home_base": self._home_base_location,
        "resources": visible_resources,
        "battery_level": self._battery_level}

    def _get_info(self):
        return {
            "carrying": self._carrying,
            "remaining_resources": len(self._resources_location)
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Home base is at the center of the grid
        self._home_base_location = np.array([self.size // 2, self.size // 2])

        self._agent_location = self._home_base_location + np.array([1, 0])

        # Choose the agent's location uniformly at random
        #self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # Resources are generated randomly, ensuring they don't overlap with agent or home base
        self._resources_location = self._generate_resources(self.num_resources)

        # # We will sample the target's location randomly until it does not coincide with the agent's location
        # self._target_location = self._agent_location
        # while np.array_equal(self._target_location, self._agent_location):
        #     self._target_location = self.np_random.integers(
        #         0, self.size, size=2, dtype=int
        #     )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

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

        #print(f"Resources = {self._resources_location}")

        # Check if the agent is on a resource location, and is not carrying a resource
        for i in range(len(self._resources_location)):
            if np.array_equal(self._agent_location, self._resources_location[i]) and not self._carrying:
                self._carrying = True  # Now the agent is carrying a resource
                # Remove the resource from the environment
                self._resources_location = np.delete(
                    self._resources_location, i, axis=0)
                break

      # Check if the agent has returned to the base with a resource. If they do they will recieve a recharge and 1 reward point.
        if np.array_equal(self._agent_location, self._home_base_location) and self._carrying:
            reward = 1  # The agent gets a reward for deliver the resource
            self._carrying = False  # The agent is no longer carrying a resource
            self._battery_level = 100  # The agent is given a free recharge

        # The episode is done if all resources have been collected
        if len(self._resources_location) == 0 and not self._carrying:
            print("Should be done now!")
            terminated = True

        # Setup Battery Logic Use up some battery for each action
        self._battery_level -= 1

        if self._battery_level <= 0:
            terminated = True  # End the episode if the battery is dead
            reward = -100  # Large negative reward for dying
        else:
            # Compute the reward as normal
            pass

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # Draw the home base
        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                pix_square_size * self._home_base_location,
                (pix_square_size, pix_square_size),
            ),
        )

        # Draw the resources
        for resource_location in self._resources_location:
            pygame.draw.rect(
                canvas,
                (255, 0, 0),  # Color of the resources
                pygame.Rect(
                    pix_square_size * resource_location,
                    (pix_square_size, pix_square_size),
                ),
            )

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        # Visualize the FOV
        fov = self.fov  # Adjust this if you've defined FOV elsewhere

        # Get the coordinates of the top-left and bottom-right corners of the FOV
        # Calculate the boundaries for the FOV
        # Calculate the boundaries for the FOV
        tl_x = max(0, self._agent_location[0] - fov)
        tl_y = max(0, self._agent_location[1] - fov)
        br_x = min(self.size, self._agent_location[0] + fov + 1)
        br_y = min(self.size, self._agent_location[1] + fov + 1)

        # Create a semi-transparent surface for the FOV
        fov_surface = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
        fov_color = (100, 100, 255, 80)  # RGBA: semi-transparent blue

        # Fill the cells within the FOV
        for x in range(tl_x, br_x):
            for y in range(tl_y, br_y):
                pygame.draw.rect(fov_surface, fov_color, pygame.Rect(pix_square_size * np.array([x, y]), (pix_square_size, pix_square_size)))

        # Blit the FOV surface onto the main canvas
        canvas.blit(fov_surface, (0, 0))

        # Pausing code
        if self.paused:
            font = pygame.font.SysFont(None, 55)
            pause_surf = font.render('Paused', True, (255, 0, 0))
            pause_rect = pause_surf.get_rect(center=(self.window_size/2, self.window_size/2))
            self.window.blit(pause_surf, pause_rect)
        
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
            
      

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    # def _reflect_at_boundaries(self):
    #     """Reflect the agent at the boundaries of the environment."""
    #     if self._agent_location[0] < 0:
    #         self._agent_location[0] = abs(self._agent_location[0])
    #     elif self._agent_location[0] >= self.size:
    #         self._agent_location[0] = 2*self.size - self._agent_location[0] - 1

    #     if self._agent_location[1] < 0:
    #         self._agent_location[1] = abs(self._agent_location[1])
    #     elif self._agent_location[1] >= self.size:
    #         self._agent_location[1] = 2*self.size - self._agent_location[1] - 1

    def _reflect_at_boundaries(self, pos):
        # """Reflect the agent at the boundaries of the environment."""
        # print(f"Agent location before reflection: {self._agent_location}")

        # # Reflect on the x-axis (horizontal boundaries)
        # if self._agent_location[0] < 0:
        #     self._agent_location[0] = abs(self._agent_location[0])
        #     print(f"Reflected on the left boundary.")
        # elif self._agent_location[0] >= self.size -1:
        #     self._agent_location[0] = 2*self.size - self._agent_location[0] - 1
        #     print(f"Reflected on the right boundary.")

        # # Reflect on the y-axis (vertical boundaries)
        # if self._agent_location[1] < 0:
        #     self._agent_location[1] = abs(self._agent_location[1])
        #     print(f"Reflected on the bottom boundary.")
        # elif self._agent_location[1] >= self.size -1:
        #     self._agent_location[1] = 2*self.size - self._agent_location[1] - 1
        #     print(f"Reflected on the top boundary.")

        # print(f"Agent location after reflection: {self._agent_location}\n")
        return pos
        

    # Below are functions related to the foraging aspects of the simulation

    def _generate_resources(self, num_resources):
        # Generate a list of all locations
        all_locations = [(x, y) for x in range(self.size)
                         for y in range(self.size)]
        print(f"all locations is {all_locations}")
        # Remove the agent's location and the home base location
        all_locations.remove(tuple(self._agent_location))

        all_locations.remove(tuple(self._home_base_location))

        # shuffle the remaing locations
        self.np_random.shuffle(all_locations)

        return np.array(all_locations[:num_resources])

    def get_carrying(self):
        return self._carrying

    def get_home_base_location(self):
        return self._home_base_location

    def get_agent_location(self):
        return self._agent_location

    def get_agent_awareness(self, radius=1):
        # Get the agent's location
        agent_x, agent_y = env._agent_location

        # Initialize an empty list to store the contents of the cells within the agent's area of awareness
        awareness = []

        # Check each cell within the radius of the agent's location
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                x = agent_x + dx
                y = agent_y + dy

                # Check if the cell is within the grid
                if 0 <= x < env.size and 0 <= y < env.size:
                    # If the cell is within the grid, add its contents to the list
                    if (x, y) in env._resources_location:
                        awareness.append('resource')
                    elif (x, y) == tuple(env._home_base_location):
                        awareness.append('home_base')
                    else:
                        awareness.append('empty')

        return awareness

    