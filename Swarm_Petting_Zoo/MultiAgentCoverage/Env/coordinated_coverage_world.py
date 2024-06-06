import copy

from pettingzoo.utils.env import AECEnv
from gym import spaces
import numpy as np
import pygame

#Made to include other agents positions in the observation space

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
        self.penalty_cap = -2  # Maximum penalty for revisits
        
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
            }
        )
        
        #Grid to track coverage, each cell inititally set to 0
        self.coverage_grid = np.zeros((self.size,self.size), dtype = int) 
        self.reward_grid = np.zeros((size, size), dtype=float)  # Tracks rewards/penalties
        
         # Initialize the dictionary to track awarded thresholds
        self.awarded_thresholds = {}

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
        self._agent_locations = self._initialize_agent_locations()

        # Reset the coverage and reward grids
        self.coverage_grid = np.zeros((self.size, self.size), dtype=int)
        self.reward_grid = np.zeros((self.size, self.size), dtype=float)

        # Reset tracking for awarded thresholds
        self.awarded_thresholds = {}

        # Reset the initial agent selection and state
        self.agent_selection = self.possible_agents[0]
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.current_step = 0  # Reset step count at the start of an episode

        # Generate initial observations for each agent
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        return observations
    
    def _initialize_agent_locations(self):
        locations = {}
        for agent in self.possible_agents:
            while True:
                location = tuple(self.np_random.integers(0, self.size, size=2))
                if location not in locations.values():  # Ensure no overlap
                    locations[agent] = location
                    break
        return locations
    
    def step(self, actions):
        self.current_step += 1
        rewards = {agent: 0 for agent in self.agents}
        terminated = False

        for agent in self.agents:
            action = actions[agent]
            total_visits = np.sum(self.coverage_grid > 0)
            movement_success_probability = 0.9  # Probability of successful movement
            sensor_success_probability = 0.8  # Probability of successful sensor mapping

            direction = self._action_to_direction[action]
            new_location = np.clip(self._agent_locations[agent] + direction, 0, self.size - 1)

            # Determine the actual movement
            if np.random.rand() < movement_success_probability:
                # Check if the new location is valid
                if self._is_location_valid(agent, new_location):
                    # Move as intended
                    self._agent_locations[agent] = new_location

                    # Check if any other agent is within one square
                    for other_agent in self.agents:
                        if other_agent != agent:
                            if np.linalg.norm(np.array(self._agent_locations[agent]) - np.array(self._agent_locations[other_agent])) <= 1:
                                rewards[agent] -= 0.01  # Small penalty for moving within one square of another agent

                else:
                    # Try to move in a random direction
                    for _ in range(4):  # Try up to 4 different directions
                        random_action = np.random.randint(4)  # Choose a random action
                        random_direction = self._action_to_direction[random_action]
                        random_new_location = np.clip(self._agent_locations[agent] + random_direction, 0, self.size - 1)
                        if self._is_location_valid(agent, random_new_location):
                            self._agent_locations[agent] = random_new_location
                            break
                    else:
                        # Penalize the agent for trying to move to an occupied cell
                        rewards[agent] -= 0.05

            visits = self.coverage_grid[new_location[0], new_location[1]]

            # Sensor mapping success
            if np.random.rand() < sensor_success_probability:
                self.coverage_grid[new_location[0], new_location[1]] += 1
                immediate_reward = self.calculate_discovery_reward(visits, total_visits)
            else:
                immediate_reward = -0.1  # Sensor failure penalty

            # Add noise to the reward
            noise = np.random.normal(0, 0.1)  # Gaussian noise with mean 0 and standard deviation 0.1
            immediate_reward += noise

            # Add immediate reward to the existing value in the reward grid
            self.reward_grid[new_location[0], new_location[1]] += immediate_reward
            rewards[agent] += immediate_reward

            rewards[agent] += self.check_and_award_completion_bonus()
            terminated = self._check_coverage_completion() or self.current_step >= self.max_steps

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        self._update_agent_selection()

        if self.render_mode == "human":
            self.render()

        info = {'step_count': self.current_step}
        return observations, rewards, terminated, self.current_step >= self.max_steps, info

    def calculate_discovery_reward(self, visits, total_visits):
        coverage_ratio = total_visits / (self.size * self.size)
        if visits == 0:
            # Encourage discovery
            return 2 + 0.5 * coverage_ratio
        else:
            # Apply penalties more severely as more of the grid is covered
            penalty = -0.005 * visits * (1 + 2 * coverage_ratio)
            return max(penalty, self.penalty_cap)  # Ensure the penalty does not go below -5

    def check_and_award_completion_bonus(self):
        total_cells = self.size * self.size
        covered_cells = np.sum(self.coverage_grid > 0)
        coverage_percentage = (covered_cells / total_cells) * 100

        # Define coverage thresholds and corresponding rewards
        thresholds = {75: 750, 85: 850, 95: 950, 100: 1000}  # Example thresholds and their bonuses
        for threshold, bonus in thresholds.items():
            if coverage_percentage >= threshold and not self.awarded_thresholds.get(threshold, False):
                self.awarded_thresholds[threshold] = True
                return bonus
        return 0
    
    
    def _check_coverage_completion(self):
        # Count the number of cells that have been visited at least once
        covered_cells = np.sum(self.coverage_grid > 0)  # Count cells that have been visited at least once

        total_cells = self.size * self.size
        coverage_percentage = (covered_cells / total_cells) * 100

        # Check if the coverage is at least 99%
        return coverage_percentage == 100

    def _is_location_valid(self, agent, location):

        # For other locations, check if they are occupied by an agent
        for other_agent, agent_location in self._agent_locations.items():
            if np.array_equal(location, agent_location):
                return False  # Location is occupied by an agent

        return True  

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
        if self.render_mode != "human":
            return

        if self.window is None:
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))  # Background color
        pix_square_size = self.window_size / self.size
        font = pygame.font.Font(None, int(pix_square_size / 3))  # Font size based on cell size

        for x in range(self.size):
            for y in range(self.size):
                visits = self.coverage_grid[x, y]
                if visits > 0:  # Only color cells that have been visited
                    color_intensity = 255 - int(255 * (visits / np.max(self.coverage_grid) if np.max(self.coverage_grid) > 0 else 1))
                    cell_color = (0, color_intensity, 0)
                    pygame.draw.rect(canvas, cell_color, 
                                    pygame.Rect(x * pix_square_size, y * pix_square_size, pix_square_size, pix_square_size))

                    # Displaying the immediate reward value
                    text_surface = font.render(f"{self.reward_grid[x, y]:.2f}", True, (255, 255, 255))
                    text_rect = text_surface.get_rect(center=(x * pix_square_size + pix_square_size / 2,
                                                            y * pix_square_size + pix_square_size / 2))
                    canvas.blit(text_surface, text_rect)

        # Draw agents after cells to ensure they are visible on top
        for agent, location in self._agent_locations.items():
            pygame.draw.circle(canvas, (255, 0, 0),  # Using red color for better visibility
                            (int((location[0] + 0.5) * pix_square_size), int((location[1] + 0.5) * pix_square_size)),
                            int(pix_square_size / 4))  # Adjust size as needed

        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

        
    # def _get_obs(self, agent):
    #     """
    #     Generate the observation for a given agent, including the agent's location
    #     and a local map centered around the agent's current position.
    #     """
    #     # Use consistent variable access for agent locations
    #     agent_location = self._agent_locations[agent]
    #     local_map = self._extract_local_map(agent_location)

    #     return {
    #         "agent_location": agent_location,  # Current location of the agent
    #         "local_map": local_map  # Local view of the grid around the agent
    #     }
    
    def _get_obs(self, agent):
        """
        Generate the observation for a given agent, including the agent's location,
        a local map centered around the agent's current position, and the relative
        positions and states of other agents within the agent's field of view.
        """
        # Use consistent variable access for agent locations
        agent_location = self._agent_locations[agent]
        local_map = self._extract_local_map(agent_location)

        # Get the relative positions of other agents within the FOV
        other_agents_positions = self._get_other_agents_positions(agent, agent_location)


        return {
            "agent_location": agent_location,  # Current location of the agent
            "local_map": local_map,  # Local view of the grid around the agent
            "other_agents_positions": other_agents_positions,  # Relative positions of other agents within the FOV
        }

    def _get_agent_state(self, agent):
        """
        Get the state of a given agent. This can be defined as needed based on the specific requirements of your environment.
        For example, it could include the agent's location, the local map around the agent, etc.
        """
        agent_location = self._agent_locations[agent]
        local_map = self._extract_local_map(agent_location)
        return {
            "agent_location": agent_location,
            "local_map": local_map
        }

    def _extract_local_map(self, center):
        center = np.array(center)  # Ensure center is an array for element-wise operations.
        fov = self.fov
        grid_size = self.size

        # Define grid bounds to handle edge cases
        top_left = np.maximum(center - fov, 0)
        bottom_right = np.minimum(center + fov + 1, grid_size)

        # Determine the actual size of the grid section to be extracted
        actual_size = bottom_right - top_left

        # Initialize the local map with zeros
        local_map = np.zeros((2 * fov + 1, 2 * fov + 1), dtype=int)

        # Calculate where to place the grid section within the local map
        start_idx = fov - (center - top_left)
        end_idx = start_idx + actual_size

        # Extract the grid section
        grid_section = self.coverage_grid[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

        # Place the grid section into the local map
        local_map[start_idx[0]:end_idx[0], start_idx[1]:end_idx[1]] = grid_section

        return local_map
    
    def _get_other_agents_positions(self, observing_agent, observing_agent_location):
        other_agents_positions = {}

        for other_agent, other_agent_location in self._agent_locations.items():
            if other_agent != observing_agent and self._is_within_fov(observing_agent, other_agent):
                relative_position = tuple(a - b for a, b in zip(other_agent_location, observing_agent_location))
                other_agents_positions[other_agent] = relative_position

        return other_agents_positions


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

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None