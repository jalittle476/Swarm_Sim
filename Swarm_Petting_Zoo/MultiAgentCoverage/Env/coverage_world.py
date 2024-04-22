import copy

from pettingzoo.utils.env import AECEnv
from gym import spaces
import numpy as np
import pygame

## TODO Implement Communication Methods

class CoverageEnvironment(AECEnv):
    metadata = {"name": "coverage_environment_v0", "render_fps": 1000}

    def __init__(self, num_agents, max_steps, render_mode=None, size=20, seed=255, fov=2, show_fov = False, show_gridlines = False, draw_numbers = False, record_sim = False):
        self.np_random = np.random.default_rng(seed)
        self.size = size  # The size of the square grid
        self.fov = fov
        self.show_fov = show_fov 
        self.render_mode = render_mode
        self.agent_to_visualize = "agent_0"
        self.show_gridlines = show_gridlines
        self.draw_numbers = draw_numbers
        self.paused = False
        self.record_sim = record_sim
        self.frame_count = 0
        self.max_steps = max_steps  # Maximum steps per episode
        self.current_step = 0  # Current step counter
        
        pygame.init()
        self.window = None
        self.clock = None
        self.window_size = 1024  # The size of the PyGame window
        
        # Initialize the possible agents
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agent_selection = self.possible_agents[0]
        self.agents = self.possible_agents.copy()
        
        # Communication related initialization
        #self.agent_messages = {agent: [] for agent in self.possible_agents}  # Stores incoming messages for each agent
        self.agent_locations = {agent: np.array([0, 0]) for agent in self.possible_agents}  # Initial locations of agents

        # Initialize observation space and action space (as provided earlier)
        self.observation_space = spaces.Dict(
            {
                "agent_location": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "local_map": spaces.Box(0, 1, shape=(fov*2+1, fov*2+1), dtype=int),  # Adjust size based on FOV
                #"messages": spaces.Box(0, 255, shape=(size, size), dtype=int)

            }
        )
        
        #Grid to track coverage, each cell inititally set to 0
        self.coverage_grid = np.zeros((self.size,self.size), dtype = int) 
        
        #self.communication_action_space = spaces.Discrete(2)  # Two options: Send or Acknowledge
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
         # Optional: Reset random seed
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        # Initialize or reset agent locations
        # For example, randomize starting locations or set them to specific points
        self._agent_locations = self._initialize_agent_locations()

        # Reset the coverage grid to all zeros
        self.coverage_grid = np.zeros((self.size, self.size), dtype=int)

        # Set the initial agent selection
        self.agent_selection = self.possible_agents[0]
        self.agents = self.possible_agents.copy()
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.current_step = 0  # Reset step count at the start of an episode


        # Return observation for the first agent
        return self._get_obs(self.agent_selection)
    
    def _initialize_agent_locations(self):
        locations = {}
        for agent in self.possible_agents:
            while True:
                location = tuple(self.np_random.integers(0, self.size, size=2))
                if location not in locations.values():  # Ensure no overlap
                    locations[agent] = location
                    break
        return locations

    def step(self, action):
        self.current_step += 1  # Increment the step counter
        agent = self.agent_selection  # Get the current agent
        movement_action = action  # Unpack the action

        # Initialize reward, termination, and truncation
        reward = 0
        terminated = False
        truncation = self.current_step >= self.max_steps  # Check if current step exceeds max steps

        # Process movement
        direction = self._action_to_direction[movement_action]
        new_location = self._agent_locations[agent] + direction
        new_location = np.clip(new_location, 0, self.size - 1)  # Ensure within bounds

        # Check if the new location is valid
        if self._is_location_valid(agent, new_location):
            self._agent_locations[agent] = new_location
            self.coverage_grid[new_location[0], new_location[1]] = 1  # Update coverage
            reward = self._calculate_coverage_reward(agent)  # Calculate reward

        # Check coverage completion
        if self._check_coverage_completion():
            terminated = True

        # Determine termination based on coverage or step count
        if truncation:
            terminated = True

        # Fetch new observation
        observation = self._get_obs(agent)
        info = {'step_count': self.current_step}

        # Update agent selection for next step
        self._update_agent_selection()

        # Rendering and debug
        if self.render_mode == "human":
            self.render()

        print(f"Step: {self.current_step}, Terminated: {terminated}, Truncation: {truncation}")

        return observation, reward, terminated, truncation, info


    def _calculate_coverage_reward(self, agent):
        # Get the agent's current location
        current_location = self._agent_locations[agent]

        # Check if the current location has already been covered
        if self.coverage_grid[current_location[0], current_location[1]] == 1:
            # Penalize slightly for revisiting a covered area
            return -0.1
        else:
            # Reward more for covering a new area
            return 1

        # Note: The reward values can be adjusted based on your simulation's needs.


    def _check_coverage_completion(self):
        # Calculate the total number of cells in the grid
        total_cells = self.size * self.size

        # Count the number of cells that have been covered
        covered_cells = np.sum(self.coverage_grid)

        # Calculate the percentage of the grid that's been covered
        coverage_percentage = (covered_cells / total_cells) * 100

        # Check if the coverage is at least 95%
        return coverage_percentage >= 95

    def _is_location_valid(self, agent, location):

        # For other locations, check if they are occupied by an agent
        for other_agent, agent_location in self._agent_locations.items():
            if np.array_equal(location, agent_location):
                return False  # Location is occupied by an agent

        return True  # Location is not occupied and is not the home base

    
    def _process_communication(self, agent, communication_action):
        print(f"Processing communication for {agent}, Action: {communication_action}")
        for other_agent in self.agents:
            if other_agent != agent and self._is_within_fov(agent, other_agent):
                message = f"Message from {agent}"
                self.agent_messages[other_agent].append(message)
                print(f"{agent} sent a message to {other_agent}: '{message}'")

    def _is_within_fov(self, agent, other_agent):
        # Convert tuples back to numpy arrays for subtraction
        agent_location = np.array(self._agent_locations[agent])
        other_agent_location = np.array(self._agent_locations[other_agent])

        # Calculate the distance
        distance = np.linalg.norm(agent_location - other_agent_location)

        # Check if the distance is within the field of view
        within_fov = distance <= self.fov
        # if within_fov:
        #      print(f"Checking FOV: {agent} -> {other_agent}, Distance: {distance}, Within FOV: {within_fov}")
        return within_fov


    
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

    def render(self):
        
        # Only proceed with rendering if the render mode is set to "human"
        if self.render_mode != "human":
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:  # Press 'p' to toggle pause
                    self.paused = not self.paused
                    
        # Skip the rest of the rendering if paused
        if self.paused:
            # You can optionally add a pause message display here
            return

        # Initialize the window and clock if they haven't been already
        if self.window is None:
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (self.window_size / self.size)  # The size of a single grid square in pixels

        # Draw the coverage grid
        for x in range(self.size):
            for y in range(self.size):
                if self.coverage_grid[x, y] == 1:
                    pygame.draw.rect(canvas, (0, 255, 0),  # Green for covered areas
                                    pygame.Rect(x * pix_square_size, y * pix_square_size, pix_square_size, pix_square_size))

        # Draw agents
        for agent, location in self._agent_locations.items():
            pygame.draw.circle(canvas, (0, 0, 255),  # Blue color for agents
                            (int((location[0] + 0.5) * pix_square_size), int((location[1] + 0.5) * pix_square_size)),
                            int(pix_square_size / 3))
            
        
        
    # Optional: Draw gridlines
        if self.show_gridlines:
            for x in range(self.size + 1):
                pygame.draw.line(canvas, (0, 0, 0),  # Black lines for grid
                                (x * pix_square_size, 0), (x * pix_square_size, self.window_size), 1)
                pygame.draw.line(canvas, (0, 0, 0),
                                (0, x * pix_square_size), (self.window_size, x * pix_square_size), 1)

       # Visualize the FOV
        fov = self.fov  # Adjust this if you've defined FOV elsewhere

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
        # Centered around the agent's location, extract the local map
        # print(f"Agent: {agent}")
        # print(f"Agent Locations: {self._agent_locations}")
        #print(f"Agent Messages: {self.agent_messages}")
        local_map_center = self.agent_locations[agent]
        local_map = self._extract_local_map(local_map_center)

        return {
            "agent_location": self._agent_locations[agent],
            "local_map" : local_map
            #"messages": self.agent_messages[agent]
        }

        
    def _extract_local_map(self, center):
        # Define the range for the local map
        top_left = np.maximum(center - self.fov, 0)
        bottom_right = np.minimum(center + self.fov + 1, self.size)

        # Extract the local map from the coverage grid
        local_map = np.zeros((2 * self.fov + 1, 2 * self.fov + 1), dtype=int)
        local_map_section = self.coverage_grid[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

        # Position the section in the center of the local map
        start_idx = self.fov - center + top_left
        end_idx = start_idx + local_map_section.shape

        local_map[start_idx[0]:end_idx[0], start_idx[1]:end_idx[1]] = local_map_section
        return local_map

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        return self._get_obs(agent)

    # Below are functions related to the coverage aspects of the simulation

    def get_agent_location(self, agent):
        return self._agent_locations[agent]
