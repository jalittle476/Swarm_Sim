import copy

from pettingzoo.utils.env import AECEnv
from gym import spaces
import numpy as np
import pygame

class ForagingEnvironment(AECEnv):
    metadata = {"name": "foraging_environment_v0", "render_fps": 30}

    def __init__(self, num_agents, render_mode=None, size=20, seed=255, num_resources=5, fov=2, show_fov = False):
        self.np_random = np.random.default_rng(seed)
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.window = pygame.display.set_mode((self.window_size, self.window_size))
        self.num_resources = num_resources
        self.fov = fov
        self.show_fov = show_fov
        self.render_mode = render_mode
        self.clock = None
        

        # Initialize the possible agents
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agent_selection = self.possible_agents[0]
        
        self.agents = self.possible_agents.copy()

        # Initialize carrying status and battery level for each agent
        self._carrying = {agent: False for agent in self.possible_agents}
        self._battery_level = {agent: 100 for agent in self.possible_agents}

        # Initialize observation space and action space (as provided earlier)
        self.observation_space = spaces.Dict(
            {
                "agent_location": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "home_base": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "resources": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "battery_level": spaces.Box(0, size - 1, shape=(2,), dtype=int)
            }
        )
        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        self.paused = False
        
        # assert render_mode is None or render_mode in self.metadata["render_modes"]
        # self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        


    def reset(self, seed=None, options=None):
        # If you're extending another class, call its reset method (if needed)
        #super().reset(seed=seed)
        self.active_agents = set(self.possible_agents)
        
        # Home base is at the center of the grid
        self._home_base_location = np.array([self.size // 2, self.size // 2])
        
        # Initialize agent locations, ensuring they don't overlap with home base
        self._agent_locations = {agent: self._home_base_location + np.array([1, 0]) for agent in self.possible_agents}

        # Resources are generated randomly, ensuring they don't overlap with agents or home base
        self._resources_location = self._generate_resources(self.num_resources)

        # Reset carrying status and battery level for each agent
        self._carrying = {agent: False for agent in self.possible_agents}
        self._battery_level = {agent: 100 for agent in self.possible_agents}

        # Set the initial agent selection
        self.agent_selection = self.possible_agents[0]
        
        self.agents = self.possible_agents.copy()
        
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # Return observation for the first agent and general info
        return self._get_obs(self.agent_selection), self._get_info(self.agent_selection)

    def step(self, action):
        agent = self.agent_selection  # Get the current agent
        
        # Initialize reward, termination, and truncation
        reward = 0
        terminated = False
        truncation = False
        
        # Check if the agent's battery level is zero
        if self._battery_level[agent] == 0 or action is None:
            # Remove the agent from the active agents set
            self.active_agents.discard(agent)
            # Skip the agent's turn if battery is zero
            self._update_agent_selection()
            
            # Check if all agents have a battery level of zero (i.e., no active agents)
            if not self.active_agents:
                terminated = True
                reward = -100  # Adjust this based on your reward scheme
            # Update termination status for all agents
                self.terminations = {agent: True for agent in self.possible_agents}
                observation, _, _, truncation, info = self.last()
                return observation, reward, terminated, truncation, info

        direction = self._action_to_direction[action]

        # Move the agent within the grid
        self._agent_locations[agent] = np.clip(
            self._agent_locations[agent] + direction, 0, self.size - 1
        )
        
        # Reduce battery level
        self._battery_level[agent] -= 1

        # Check if the agent is on a resource location
        for i in range(len(self._resources_location)):
            if np.array_equal(self._agent_locations[agent], self._resources_location[i]) and not self._carrying[agent]:
                self._carrying[agent] = True
                self._resources_location = np.delete(self._resources_location, i, axis=0)
                break

        # Check if the agent has returned to the base with a resource
        if np.array_equal(self._agent_locations[agent], self._home_base_location) and self._carrying[agent]:
            reward = 1
            self._carrying[agent] = False
            self._battery_level[agent] = 100

       # Check termination conditions
        if len(self._resources_location) == 0 and not any(self._carrying.values()):
            terminated = True

        observation = self._get_obs(agent)
        info = self._get_info(agent)

        # Update selected agent for the next step
        self._update_agent_selection()
        
        if not self.active_agents:
            terminated = True

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncation, info

    def _update_agent_selection(self):
        
        if not self.active_agents:
            return
        
        current_idx = self.possible_agents.index(self.agent_selection)
        next_idx = (current_idx + 1) % len(self.active_agents)
        self.agent_selection = self.possible_agents[next_idx]

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
                (255, 0, 0),
                pygame.Rect(
                    pix_square_size * resource_location,
                    (pix_square_size, pix_square_size),
                ),
            )

        # Now we draw all the agents
        for agent, location in self._agent_locations.items():
            pygame.draw.circle(
                canvas,
                (0, 0, 255),
                (location + 0.5) * pix_square_size,
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

        # Get the location of the currently selected agent
        current_agent_location = self._agent_locations[self.agent_selection]

        # Get the coordinates of the top-left and bottom-right corners of the FOV
        # Calculate the boundaries for the FOV
        tl_x = max(0, current_agent_location[0] - fov)
        tl_y = max(0, current_agent_location[1] - fov)
        br_x = min(self.size, current_agent_location[0] + fov + 1)
        br_y = min(self.size, current_agent_location[1] + fov + 1)


        # Create a semi-transparent surface for the FOV
        fov_surface = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
        fov_color = (100, 100, 255, 80)  # RGBA: semi-transparent blue

        # Fill the cells within the FOV
        for x in range(tl_x, br_x):
            for y in range(tl_y, br_y):
                pygame.draw.rect(fov_surface, fov_color, pygame.Rect(pix_square_size * np.array([x, y]), (pix_square_size, pix_square_size)))

        # Blit the FOV surface onto the main canvas
        canvas.blit(fov_surface, (0, 0))

        # # Visualize the FOV (for the currently selected agent if needed)
        # if self.show_fov:
        #     agent_location = self._agent_locations[self.agent_selection]
        #     # Code to render FOV based on agent_location ...

        # ... (rest of the code, including pausing and display update, stays the same) ...

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


    def _get_obs(self, agent):
        # Define the agent's field of view (FOV)
        fov = self.fov

        # Get the coordinates of the top-left and bottom-right corners of the FOV
        tl_y = max(0, self._agent_locations[agent][0] - fov)
        tl_x = max(0, self._agent_locations[agent][1] - fov)
        br_y = min(self.size, self._agent_locations[agent][0] + fov + 1)
        br_x = min(self.size, self._agent_locations[agent][1] + fov + 1)

        # Check each resource
        visible_resources = []
        for resource_location in self._resources_location:
            # If the resource is within the FOV, add it to the list of visible resources
            if tl_y <= resource_location[0] < br_y and tl_x <= resource_location[1] < br_x:
                visible_resources.append(resource_location)

        return {
            "agent_location": self._agent_locations[agent],
            "home_base": self._home_base_location,
            "resources": visible_resources,
            "battery_level": self._battery_level[agent]
        }

    def _get_info(self, agent):
        return {
            "carrying": self._carrying[agent],
            "remaining_resources": len(self._resources_location)
    }

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        return self._get_obs(agent)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _generate_resources(self, num_resources):
        # Generate a list of all locations
        all_locations = {(x, y) for x in range(self.size) for y in range(self.size)}

        # Remove agent locations and the home base location
        for agent_location in self._agent_locations.values():
            all_locations.discard(tuple(agent_location))
        all_locations.discard(tuple(self._home_base_location))

        # Convert to a list and shuffle the remaining locations
        all_locations = list(all_locations)
        self.np_random.shuffle(all_locations)

        return np.array(all_locations[:num_resources])
    

    # Below are functions related to the foraging aspects of the simulation

    def get_carrying(self, agent):
        return self._carrying[agent]

    def get_home_base_location(self):
        return self._home_base_location

    def get_agent_location(self, agent):
        return self._agent_locations[agent]

    def get_agent_awareness(self, agent, radius=1):
        # Get the agent's location
        agent_x, agent_y = self._agent_locations[agent]

        # Initialize an empty list to store the contents of the cells within the agent's area of awareness
        awareness = []

        # Check each cell within the radius of the agent's location
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                x = agent_x + dx
                y = agent_y + dy

                # Check if the cell is within the grid
                if 0 <= x < self.size and 0 <= y < self.size:
                    if (x, y) in self._resources_location:
                        awareness.append('resource')
                    elif (x, y) == tuple(self._home_base_location):
                        awareness.append('home_base')
                    else:
                        awareness.append('empty')

        return awareness
