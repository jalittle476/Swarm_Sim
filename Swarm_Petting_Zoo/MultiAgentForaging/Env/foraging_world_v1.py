import copy

from pettingzoo.utils.env import AECEnv
from gym import spaces
import numpy as np
import pygame

## Foraging World Without Communication

class ForagingEnvironment(AECEnv):
    metadata = {"name": "foraging_environment_v0", "render_fps": 1000}

    def __init__(self, num_agents, render_mode=None, size=20, seed=255, num_resources=5, fov=2, show_fov = False, show_gridlines = False, draw_numbers = False, record_sim = False):
        self.np_random = np.random.default_rng(seed)
        self.size = size  # The size of the square grid
        self.num_resources = num_resources
        self.fov = fov
        self.show_fov = show_fov 
        self.render_mode = render_mode
        self.agent_to_visualize = "agent_0"
        self.show_gridlines = show_gridlines
        self.draw_numbers = draw_numbers
        self.paused = False
        self.record_sim = record_sim
        self.frame_count = 0
        self.full_battery_charge = 4 * size # They could explore the perimeter of the space 
        
        pygame.init()
        self.window = None
        self.clock = None
        self.window_size = 1024  # The size of the PyGame window
        
        # Initialize the possible agents
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agent_selection = self.possible_agents[0]
        self.agents = self.possible_agents.copy()

        # Initialize observation space and action space 
        self.observation_space = spaces.Dict(
            {
                "agent_location": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "home_base": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "resources": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "battery_level": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )
        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
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
        # If you're extending another class, call its reset method (if needed)
        #super().reset(seed=seed)
        # self.active_agents = set(self.possible_agents)
        
        # Home base is at the center of the grid
        self._home_base_location = np.array([self.size // 2, self.size // 2])
        
        # Initialize agent locations, ensuring they don't overlap with home base
        self._agent_locations = {agent: self._home_base_location + np.array([1, 0]) for agent in self.possible_agents}

        # Resources are generated randomly, ensuring they don't overlap with agents or home base
        self._resources_location = self._generate_resources(self.num_resources)

        # Reset carrying status and battery level for each agent
        self._carrying = {agent: False for agent in self.possible_agents}
        
        self._battery_level = {agent: self.full_battery_charge for agent in self.possible_agents}

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

        # Check if the agent's battery level is zero or if the agent is already terminated
        if self._battery_level[agent] == 0 or self.terminations[agent] or action is None:
            self.terminations[agent] = True  # Mark the agent as terminated

            # Check if all agents are terminated
            if all(self.terminations.values()):
                terminated = True
                reward = -100  # Adjust this based on your reward scheme

                observation, _, _, truncation, info = self.last()
                return observation, reward, terminated, truncation, info

            # Skip the rest of the step for this agent
            self._update_agent_selection()
            return

        direction = self._action_to_direction[action]
        
        # Calculate the potential new location
        new_location = self._agent_locations[agent] + direction
        
        # Check if the new location is not occupied by another agent
        if not self._is_location_valid(agent, new_location):
            # Try to avoid other agent
            new_location = self._simple_avoidance(agent,direction)
        
        # Move the agent to available space
        self._agent_locations[agent] = np.clip(
            new_location,           0, self.size - 1
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
            self._battery_level[agent] = self.full_battery_charge

       # Check termination conditions
        if len(self._resources_location) == 0 and not any(self._carrying.values()):
            terminated = True

        observation = self._get_obs(agent)
        info = self._get_info(agent)

        # Update selected agent for the next step
        self._update_agent_selection()
        
        if all(self.terminations.values()):
            terminated = True
            
        if self.render_mode == "human":
            self._render()

        return observation, reward, terminated, truncation, info
    
    def _is_location_valid(self, agent, location):
        # Check if the location is the home base
        if np.array_equal(location, self._home_base_location):
            return True  # The home base can hold any number of agents

        # If the agent is carrying a resource, check if the location is occupied by another resource
        # if self._carrying[agent]:
        #     for resource_location in self._resources_location:
        #         if np.array_equal(location, resource_location):
        #             return False  # Location is occupied by a resource

        # For other locations, check if they are occupied by an agent
        for other_agent, agent_location in self._agent_locations.items():
            if np.array_equal(location, agent_location):
                return False  # Location is occupied by an agent

        return True  # Location is not occupied and is not the home base

    
    def _simple_avoidance(self, agent, direction):
        # Check alternative directions for avoidance
        alternative_directions = [
            np.array([direction[1], direction[0]]),   # Turn right
            np.array([-direction[1], -direction[0]]), # Turn left
            -direction                                # Move backward
        ]

        for alt_dir in alternative_directions:
            new_location = self._agent_locations[agent] + alt_dir
            if self._is_location_valid(agent, new_location):
                return new_location

        return self._agent_locations[agent]  # Stay in place if no valid move is found

    def _update_agent_selection(self):
        current_idx = self.agents.index(self.agent_selection)
        next_idx = (current_idx + 1) % len(self.agents)

        # Loop through the agents starting from the next index, looking for the next non-terminated agent
        for i in range(len(self.agents)):
            candidate_idx = (next_idx + i) % len(self.agents)
            candidate_agent = self.agents[candidate_idx]

            if not self.terminations[candidate_agent]:
                self.agent_selection = candidate_agent
                return

        # If all agents are terminated, you can handle it as you see fit (e.g., set agent_selection to None)
        self.agent_selection = None

    def _render(self):
        
        if self.window is None and self.render_mode == "human":
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
            (102, 51, 0),
            pygame.Rect(
                pix_square_size * self._home_base_location,
                (pix_square_size, pix_square_size),
            ),
        )

        # Draw the resources
        for resource_location in self._resources_location:
            pygame.draw.rect(
                canvas,
                (0, 102, 0),
                pygame.Rect(
                    pix_square_size * resource_location,
                    (pix_square_size, pix_square_size),
                ),
            )

        if self.draw_numbers: 
             # Now we draw all the agents
            for agent, location in self._agent_locations.items():
                # Check if the agent is carrying a resource
                is_carrying_resource = self._carrying[agent]
                
                # Check if the agent's battery level is zero
                is_battery_depleted = self._battery_level[agent] == 0
                
                # Determine the agent's color based on its status
                if is_battery_depleted:
                    agent_color = (0, 0, 0)  # Black color for zero battery
                elif is_carrying_resource:
                    agent_color = (0, 102, 0)  # Green color if carrying a resource
                else:
                    agent_color = (0, 0, 255)  # Blue color otherwise

                pygame.draw.circle(
                    canvas,
                    agent_color,  # Use the determined color
                    (location + 0.5) * pix_square_size,
                    pix_square_size / 3,
                )

                # Draw the agent's index number
                font = pygame.font.SysFont(None, 20)  # Choose an appropriate font size
                text_surface = font.render(str(idx), True, (0, 0, 0))  # White text
                # Position the text above the agent
                text_position = ((location[0] + 0.3) * pix_square_size, (location[1] - 0.2) * pix_square_size)  # Adjust this position as needed
                canvas.blit(text_surface, text_position)
        
        else:
            
            # Now we draw all the agents
            for agent, location in self._agent_locations.items():
                # Check if the agent is carrying a resource
                is_carrying_resource = self._carrying[agent]
                
                # Check if the agent's battery is low
                is_battery_low = self._battery_level[agent] < self.size
                
                # Check if the agent's battery level is zero
                is_battery_depleted = self._battery_level[agent] == 0

                # Determine the agent's color based on its status
                if is_battery_depleted:
                    agent_color = (0, 0, 0)  # Black color for zero battery
                elif is_battery_low:
                    agent_color = (255,0,0) # Red color if battery is low 
                elif is_carrying_resource:
                    agent_color = (0, 255, 0)  # Green color if carrying a resource
                else:
                    agent_color = (0, 0, 255)  # Blue color otherwise

                pygame.draw.circle(
                    canvas,
                    agent_color,  # Use the determined color
                    (location + 0.5) * pix_square_size,
                    pix_square_size / 3,
                )
            
        
        
        if self.show_gridlines:
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
        fov = self.fov  

        if self.show_fov:
            
            # Get the location of the currently selected agent
            current_agent_location = self._agent_locations[self.agent_to_visualize]

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

          # Pausing code
        if self.paused:
            font = pygame.font.SysFont(None, 55)
            pause_surf = font.render('Paused', True, (255, 0, 0))
            pause_rect = pause_surf.get_rect(center=(self.window_size/2, self.window_size/2))
            self.window.blit(pause_surf, pause_rect)
        
        
        # Determine the number of active agents based on the terminations attribute
        num_active_agents = sum(not terminated for terminated in self.terminations.values())

        # Define the font and size
        font = pygame.font.SysFont(None, 24)

        # Create a surface containing the text
        text_surface = font.render(f'Active Agents: {num_active_agents}', True, (0, 0, 0))

        # Define the position where the text will be drawn (you can adjust this as needed)
        text_position = (10, 10)

        # Draw the text on the canvas
        canvas.blit(text_surface, text_position)
        
        # Initialize position for the key; adjust x_offset to place the key on the right
        x_offset, y_offset = self.window_size + 10, 10  
        line_height = 30  # Space between lines

        # Initialize the font
        font = pygame.font.SysFont(None, 24)

        # Render text for the key
        key_title_surface = font.render("Key:", True, (0, 0, 0))
        self.window.blit(key_title_surface, (x_offset, y_offset))
        y_offset += line_height
        
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
            
            if self.record_sim:
                 # Save the current frame as an image
                frame_filename = f'frames/frame_{self.frame_count:05d}.png'
                pygame.image.save(self.window, frame_filename)
                self.frame_count += 1
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
        # Generate a list of all locations
        all_locations = {(x, y) for x in range(self.size) for y in range(self.size)}

        # Remove agent locations and the home base location
        for agent_location in self._agent_locations.values():
            all_locations.discard(tuple(agent_location))
        
        all_locations.discard(tuple(self._home_base_location))

        # Convert to a list and shuffle the remaining locations
        all_locations = list(all_locations)
        self.np_random.shuffle(all_locations)
        
        
        food_block = {(x,y) for x in range(10) for y in range(10)}
        food_block_list = list(food_block)
        
        #print(food_block)
        #return np.array(all_locations[:num_resources])    
        
        return np.array(food_block_list)

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
