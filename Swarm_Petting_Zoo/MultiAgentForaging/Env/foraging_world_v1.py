import copy

from pettingzoo.utils.env import AECEnv
from pettingzoo.utils import agent_selector
from gym import spaces
from foraging_config import ForagingConfig
import numpy as np
import pygame
import random 

class ForagingEnvironment(AECEnv):
    metadata = {"name": "foraging_environment_v0", "render_fps": 1000}

    def __init__(self, config: ForagingConfig):

        # Directly set the class attributes using the config dictionary
        self.__dict__.update(config.__dict__)

        self.np_random = np.random.default_rng(self.seed)  # Now self.seed is available
        self.paused = False
        self.frame_count = 0
      
        pygame.init()
        self.window = None
        self.clock = None
        self.window_size = 1024  # The size of the PyGame window
        
        # Initialize the grid
        self.grid = np.zeros((self.size, self.size), dtype=int)
        
        #Initialize the home base location
        self._home_base_location = np.array([self.size // 2, self.size // 2])
        
        # Initialize the possible agents
        self.possible_agents = [f"agent_{i}" for i in range(config.num_agents)]
        self.agent_selection = self.possible_agents[0]
        self.agents = self.possible_agents.copy()
        
        # Initialize the cache for agent colors
        self.agent_color_cache = {}  # A dictionary to store the colors of agents

        self.occupied_locations = {}  # Dictionary to store sets of agents at each location


        # Initialize agent colors for all agents
        for agent_id in range(self.num_agents):
            agent_name = f"agent_{agent_id}"
            self.agent_color_cache[agent_name] = (0, 0, 255)  # Default color: Blue

        # Initialize observation space and action space 
        self.observation_space = spaces.Dict(
            {
                "agent_location": spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
                "home_base": spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
                "resources": spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
                "battery_level": spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
            }
        )
        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            None: np.array([0, 0]),
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        
    def reset(self, seed=None, options=None):
         # Reset the grid
        self.grid.fill(0)
        
         # Home base is at the center of the grid
        self.grid[self._home_base_location[0], self._home_base_location[1]] = -1  # Mark the home base
        
       # Initialize agent locations and update the grid
        self._agent_locations = {}
        for agent in self.possible_agents:
            self._agent_locations[agent] = self._home_base_location + np.array([1, 0])
            self.grid[self._agent_locations[agent][0], self._agent_locations[agent][1]] = 1  # Mark the agent
        
        # Generate resources and update the grid
        generated_resources = self._generate_resources(self.num_resources, self.distribution_type)
        self._resources_location = set(map(tuple, generated_resources))

        # Update the grid to mark resource locations
        for resource_location in self._resources_location:
            self.grid[resource_location[0], resource_location[1]] = 2  # Mark the resource


        # Reset carrying status and battery level for each agent
        self._carrying = {agent: False for agent in self.possible_agents}
        self._battery_level = {agent: self.full_battery_charge for agent in self.possible_agents}

        # Set the initial agent selection
        
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        agent = self.agent_selection  # Get the current agent

        # Initialize reward, termination, and truncation
        reward = 0
        terminated = False
        truncation = False
        observation = self.observe(agent)
        info = self._get_info(agent)

        # Handle cases where the agent's battery is depleted, agent is terminated
        if self._battery_level[agent] == 0:
            terminated = True  # Mark the agent as terminated
            self.terminations[agent] = True  # Mark the agent as terminated
            
            # Check if all agents are terminated
            if all(self.terminations.values()):
                terminated = True
                reward = -100  # Adjust this based on your reward scheme

                return observation, reward, terminated, truncation, info
            
            # selects the next agent.
            self.agent_selection = self._agent_selector.next()
            return observation, reward, terminated, truncation, info # If terminated, end the step here without doing anything else

        # Process the action: Determine the new direction and location
        direction = self._action_to_direction[action]
        new_location = self._agent_locations[agent] + direction

        # Validate the new location and possibly avoid collisions
        if not self._is_location_valid(agent, new_location):
            new_location = self._simple_avoidance(agent, direction)
        
         # Update the agent's location 
        self.update_agent_location(agent, new_location)
        
        # Handle resource collection if the agent is on a resource location
        agent_location = tuple(self._agent_locations[agent])  # Convert agent location to a tuple

        if agent_location in self._resources_location and not self._carrying[agent]:
            self._carrying[agent] = True
            self.grid[agent_location[0], agent_location[1]] = 0  # Remove resource from grid
            self._resources_location.remove(agent_location)  # Remove resource from the set


        # Check if the agent has returned to the base with a resource
        if np.array_equal(self._agent_locations[agent], self._home_base_location) and self._carrying[agent]:
            reward = 1
            self._carrying[agent] = False

        # If no resources remain, terminate the environment
        if len(self._resources_location) == 0 and not any(self._carrying.values()):
            terminated = True
            self.terminations = {agent: True for agent in self.agents}   

        observation = self.observe(agent)
        info = self._get_info(agent)
        
        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
            
        if self.render_mode == "human":
            self._render()

        return observation, reward, terminated, truncation, info

    def _is_location_valid(self, agent, location):
        """Check if a location is valid for an agent to move to."""
        # Ensure the location is within the grid boundaries
        if not (0 <= location[0] < self.size and 0 <= location[1] < self.size):
            return False  # The location is outside the grid

        # Check if the location is the home base
        if self._is_home_base(location):
            return True  # The home base can hold any number of agents

        # Check if the location is occupied by another agent
        if self._is_occupied_by_agent(location, exclude_agent=agent):
            return False  # Location is occupied by another agent

        return True  # Location is valid and unoccupied
    
    def _simple_avoidance(self, agent, direction):
        # Check alternative directions for avoidance
        alternative_directions = [
            np.array([direction[1], direction[0]]),   # Turn right
            np.array([-direction[1], -direction[0]]), # Turn left
            -direction,                               # Move backward
            np.array([0, 0]),                         # Stay in place
        ]

        # Shuffle the alternative directions to randomize the choice order
        random.shuffle(alternative_directions)

        for alt_dir in alternative_directions:
            new_location = self._agent_locations[agent] + alt_dir
            if self._is_location_valid(agent, new_location):
                return new_location

        return self._agent_locations[agent]  # Stay in place if no valid move is found

    def _render(self):
        
        # If render_mode is None, skip rendering entirely
        if self.render_mode is None:
            return
        
        if self.window is None and self.render_mode == "human":
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size  # The size of a single grid square in pixels
        
        # Initialize Pygame font
        pygame.font.init()
        font = pygame.font.SysFont('Arial', 12)  # Adjust the font size as needed

        # Draw the home base
        home_base_pos = (int(pix_square_size * self._home_base_location[1]), int(pix_square_size * self._home_base_location[0]))
        pygame.draw.rect(
            canvas,
            (102, 51, 0),
            pygame.Rect(
                home_base_pos,
                (pix_square_size, pix_square_size),
            ),
        )

        # Draw the resources
        for resource_location in self._resources_location:
            # Scale each coordinate separately and convert to integers
            scaled_location = (int(pix_square_size * resource_location[0]), int(pix_square_size * resource_location[1]))
            
            pygame.draw.rect(
                canvas,
                (0, 102, 0),
                pygame.Rect(
                    scaled_location,
                    (int(pix_square_size), int(pix_square_size)),
                ),
            )

            # Display grid addresses on the simulation grid
        if self.show_grid_addresses:
            for x in range(self.size):
                for y in range(self.size):
                    # Compute the position for the text
                    text_location = (int(pix_square_size * x), int(pix_square_size * y))
                    
                    # Render the grid coordinates as text
                    text_surface = font.render(f'({x}, {y})', True, (0,0,0))  # White text
                    
                    # Blit (draw) the text surface onto the canvas
                    canvas.blit(text_surface, text_location)

        # Draw the agents
        for agent, location in self._agent_locations.items():
            # Use cached color
            agent_color = self.agent_color_cache[agent]
            agent_pos = ((location + 0.5) * pix_square_size).astype(int)
            pygame.draw.circle(
                canvas,
                agent_color,
                agent_pos,
                pix_square_size / 3,
            )

            if self.draw_numbers:
                text_surface = font.render(str(agent), True, (0, 0, 0))
                text_position = (
                    (location[0] + 0.3) * pix_square_size,
                    (location[1] - 0.2) * pix_square_size,
                )
                canvas.blit(text_surface, text_position)

        # Optionally draw gridlines
        if self.show_gridlines:
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
        if self.show_fov:
            fov_surface = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
            fov_color = (100, 100, 255, 80)
            
            current_agent_location = self._agent_locations[self.agent_to_visualize]
            tl_x = max(0, current_agent_location[0] - self.fov)
            tl_y = max(0, current_agent_location[1] - self.fov)
            br_x = min(self.size, current_agent_location[0] + self.fov + 1)
            br_y = min(self.size, current_agent_location[1] + self.fov + 1)

            
            
            pygame.draw.rect(
                fov_surface,
                fov_color,
                pygame.Rect(
                    (tl_x * pix_square_size, tl_y * pix_square_size),
                    ((br_x - tl_x) * pix_square_size, (br_y - tl_y) * pix_square_size),
                ),
            )

            canvas.blit(fov_surface, (0, 0))

        # Display number of active agents
        num_active_agents = sum(not terminated for terminated in self.terminations.values())
        text_surface = font.render(f'Active Agents: {num_active_agents}', True, (0, 0, 0))
        text_position = (10, 10)
        canvas.blit(text_surface, text_position)

        # Render the key
        x_offset, y_offset = self.window_size + 10, 10
        line_height = 30
        key_title_surface = font.render("Key:", True, (0, 0, 0))
        self.window.blit(key_title_surface, (x_offset, y_offset))
        y_offset += line_height

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

            if self.record_sim:
                frame_filename = f'frames/frame_{self.frame_count:05d}.png'
                pygame.image.save(self.window, frame_filename)
                self.frame_count += 1
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def _update_agent_color(self, agent):
        """
        Update the color of the agent based on its current state.
        This function is called only when the agent's state changes.
        """
        is_carrying_resource = self._carrying[agent]
        battery_level = self._battery_level[agent]

        if battery_level == 0:
            self.agent_color_cache[agent] = (0, 0, 0)  # Black for zero battery
        elif battery_level < self.size:
            self.agent_color_cache[agent] = (255, 0, 0)  # Red for low battery
        elif is_carrying_resource:
            self.agent_color_cache[agent] = (0, 102, 0)  # Green if carrying a resource
        else:
            self.agent_color_cache[agent] = (0, 0, 255)  # Blue otherwise
    
    def _get_obs(self, agent):
        """Get observations about the agent's surroundings, including visible resources and FOV coordinates."""

        # Get the agent's current location
        agent_location = self._agent_locations[agent]

        # Get the FOV corners
        tl_y, tl_x, br_y, br_x = self.get_fov_corners(agent_location, self.fov)

        # Slice the grid to get the agent's FOV
        fov_slice = self.grid[tl_y:br_y, tl_x:br_x]

        # Convert the grid slice into a list of resource locations within the FOV
        visible_resources = np.argwhere(fov_slice == 2)

        # Adjust resource coordinates relative to the global grid
        visible_resources = [
            (y + tl_y, x + tl_x) for y, x in visible_resources
        ]

        # Calculate FOV coordinates (all grid cells within the FOV)
        fov_coordinates = [
            (y, x) for y in range(tl_y, br_y) for x in range(tl_x, br_x)
        ]

        observation = {
            "agent_location": self._agent_locations[agent],
            "home_base": self._home_base_location,
            "resources": visible_resources,
            "battery_level": self._battery_level[agent],
            "fov_coordinates": fov_coordinates  # Include FOV coordinates in the observation
        }
        
        return observation

    def _get_info(self, agent):
        return {
            "carrying": self._carrying[agent],
            "remaining_resources": len(self._resources_location)}

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
        """Generate resource locations on the grid, avoiding agent locations and the home base."""
        # Generate a set of all possible locations on the grid
        all_locations = {(x, y) for x in range(self.size) for y in range(self.size)}

        # Exclude agent locations and the home base location
        excluded_locations = set(tuple(loc) for loc in self._agent_locations.values())
        excluded_locations.add(tuple(self._home_base_location))

        # Get available locations by subtracting the excluded ones
        available_locations = list(all_locations - excluded_locations)

        # Shuffle available locations to randomize resource placement
        self.np_random.shuffle(available_locations)

        # Select the desired number of resource locations
        selected_resource_locations = np.array(available_locations[:num_resources])

        return selected_resource_locations
    
    def _generate_resources(self, num_resources, distribution_type='uniform'):
        """Generate resource locations on the grid with different distribution types.

        Args:
            num_resources (int): Number of resources to generate.
            distribution_type (str): Type of distribution to use ('uniform' or 'clustered').
        
        Returns:
            np.ndarray: Array of generated resource locations.
        """
        # Generate a set of all possible locations on the grid
        all_locations = {(y, x) for y in range(self.size) for x in range(self.size)}

        # Exclude agent locations and the home base location
        excluded_locations = set(tuple(loc) for loc in self._agent_locations.values())
        excluded_locations.add(tuple(self._home_base_location))

        # Get available locations by subtracting the excluded ones
        available_locations = list(all_locations - excluded_locations)

        if distribution_type == 'uniform':
            # Uniform Distribution: Randomly select locations from all available locations
            self.np_random.shuffle(available_locations)
            selected_resource_locations = np.array(available_locations[:num_resources])
            return selected_resource_locations

        elif distribution_type == 'clustered':
            # Determine number of clusters based on grid area
            num_clusters = max(1, int(self.size ** 2 / 50))  # Example: 1 cluster per 50 cells
            num_clusters = min(num_clusters, len(available_locations))  # Limit to available locations

            # Determine approximate resources per cluster
            cluster_sizes = [num_resources // num_clusters] * num_clusters
            for i in range(num_resources % num_clusters):  # Distribute any remaining resources
                cluster_sizes[i] += 1

            # Shuffle available locations and select random cluster centers
            self.np_random.shuffle(available_locations)
            cluster_centers = available_locations[:num_clusters]

            # Initialize a list to store resource locations
            selected_resource_locations = []

            # Define a fixed radius for each cluster
            cluster_radius = 2  # You can adjust this to control the spread of each cluster

            # Generate resources around each cluster center
            for cluster_index, center in enumerate(cluster_centers):
                center_y, center_x = center
                cluster_size = cluster_sizes[cluster_index]

                for _ in range(cluster_size):
                    # Generate resources within a fixed radius around the cluster center
                    offset_y = self.np_random.integers(-cluster_radius, cluster_radius + 1)
                    offset_x = self.np_random.integers(-cluster_radius, cluster_radius + 1)
                    new_location = (center_y + offset_y, center_x + offset_x)

                    # Ensure the new location is within grid bounds and not excluded
                    if (0 <= new_location[0] < self.size and
                        0 <= new_location[1] < self.size and
                        new_location not in excluded_locations):
                        selected_resource_locations.append(new_location)
                        excluded_locations.add(new_location)  # Prevent double-adding

                        # Stop if we reach the desired number of resources
                        if len(selected_resource_locations) >= num_resources:
                            return np.array(selected_resource_locations)

            return np.array(selected_resource_locations)

        else:
            raise ValueError("Unsupported distribution type. Choose 'uniform' or 'clustered'.")



    # Below are functions related to the foraging aspects of the simulation

    def get_carrying(self, agent):
        return self._carrying[agent]

    def get_home_base_location(self):
        return self._home_base_location

    def get_agent_location(self, agent):
        return self._agent_locations[agent]

    def get_fov_corners(self, location, fov):
        """
        Calculate the top-left and bottom-right corners of the FOV centered around a given location.

        Parameters:
        - location: The (y, x) coordinates of the center location (e.g., the agent's location).
        - fov: The radius of the field of view.

        Returns:
        - (tl_y, tl_x, br_y, br_x): The coordinates of the top-left and bottom-right corners of the FOV.
        """
        # Get the coordinates of the top-left corner of the FOV
        tl_y = max(0, location[0] - fov)
        tl_x = max(0, location[1] - fov)
        
        # Get the coordinates of the bottom-right corner of the FOV
        br_y = min(self.size, location[0] + fov + 1)
        br_x = min(self.size, location[1] + fov + 1)
        
        return tl_y, tl_x, br_y, br_x

    def _is_home_base(self, location):
        """Check if the given location is the home base."""
        return np.array_equal(location, self._home_base_location)

    def _is_occupied_by_agent(self, location, exclude_agent=None):
        """Check if the location is occupied by any active agent, excluding a specific agent if needed."""

        location = tuple(location)  # Ensure the location is a tuple

        # Fast check using the dictionary
        if location in self.occupied_locations:
            for agent in self.occupied_locations[location]:
                # Skip the excluded agent and any terminated agents
                if agent == exclude_agent or self.terminations.get(agent, False):
                    continue
                return True  # Found an active agent occupying the location

        return False  # Location is not occupied by any active agent

    
    def update_agent_location(self, agent, new_location):
        """Update the location of an agent and manage the dictionary of occupied locations."""
        old_location = tuple(self._agent_locations[agent])  # Convert old location to tuple

        # Remove the agent from the old location
        if old_location in self.occupied_locations:
            self.occupied_locations[old_location].discard(agent)  # Use discard to safely remove without KeyError
            # If no agents remain at the old location, remove the entry from the dictionary
            if not self.occupied_locations[old_location]:
                del self.occupied_locations[old_location]

        # Update the agent's location
        self._agent_locations[agent] = new_location
        new_location = tuple(new_location)

        # Add the agent to the new location
        if new_location not in self.occupied_locations:
            self.occupied_locations[new_location] = set()
        self.occupied_locations[new_location].add(agent)
        
    

