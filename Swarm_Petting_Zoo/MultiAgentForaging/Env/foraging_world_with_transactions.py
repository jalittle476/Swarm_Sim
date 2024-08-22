from foraging_world_v1 import ForagingEnvironment  # Importing your base environment
import numpy as np
import random
from foraging_config import ForagingConfig

class ForagingEnvironmentWithTransactions(ForagingEnvironment):
    def __init__(self, config: ForagingConfig):
        super().__init__(config)  # Initialize the base class with the provided config
        self.__dict__.update(config.__dict__)

        # Additional initialization for the subclass
        self.initialize_agents()  # Initialize agent-specific attributes
        self.rng = np.random.default_rng(self.seed)  # Initialize the Generator

    def initialize_agents(self):
        self._money = {agent: self.initial_money for agent in self.agents}
        self._resource_reward = self.resource_reward
        self._battery_usage_rate = self.battery_usage_rate
        self._battery_charge_cost = self.battery_charge_cost
        self._battery_charge_amount = self.battery_charge_amount
        self._min_battery_level = self.min_battery_level
        self._battery_recharge_threshold = self.battery_recharge_threshold
        
        # Default standard deviations for different behaviors
        self.std_dev_base_return = 0.8
        self.std_dev_foraging = 0.5
        
        # Initialize the state of each agent
        self.agent_states = {agent: "Foraging" for agent in self.agents}  # Default state is "Foraging"

    def calculate_direction(self, start, end):
        """Calculate the direction vector from start to end."""
        return np.array(end) - np.array(start)

    def adjust_base_proximity_threshold(self, agent, base_threshold=0, max_threshold=5):
        """Adjust the base proximity threshold based on local agent density."""
        local_density = self.calculate_local_density(agent)
        # Simple linear scaling of threshold based on local density
        adjusted_threshold = min(base_threshold + local_density, max_threshold)
        return adjusted_threshold

    def gaussian_sample(self, direction, std_dev, no_movement_prob=0.1):
        """Sample a discrete action based on the direction vector with added Gaussian noise."""

        # Introduce a probability for no movement
        if self.rng.random() < no_movement_prob:
            if self.debug:
                print(f"Agent chooses not to move. (No movement with probability {no_movement_prob})")
            return None  # No movement action (could return a specific "no movement" code if needed)

        # Normalize the direction vector
        norm = np.linalg.norm(direction)
        if norm != 0:
            direction = direction / norm
        
        # Apply Gaussian noise to the direction
        sampled_direction = self.rng.normal(direction, std_dev)
        if self.debug:
            print(f"Original direction: {direction}, Sampled direction: {sampled_direction}")
        
        # Determine the action based on the sampled direction
        if abs(sampled_direction[0]) > abs(sampled_direction[1]):
            return 0 if sampled_direction[0] > 0 else 2  # Move right or left
        else:
            return 1 if sampled_direction[1] > 0 else 3  # Move down or up

    def should_return_to_base(self, battery_level, _min_battery_level_level):
        """Check if the agent should return to the base based on its battery level."""
        return battery_level <= self._min_battery_level

    def return_to_base(self, agent_location, base_location):
        """Generate an action to return the agent to the base."""
        if np.array_equal(agent_location, base_location):
            return None  # No action needed, agent is already at the base
        direction_to_base = self.calculate_direction(agent_location, base_location)
        action = self.gaussian_sample(direction_to_base, self.std_dev_base_return)
        
        # Handle the case where the agent might choose not to move
        if action is None:
            if self.debug:
                print(f"Agent at {agent_location} chose not to move towards base at {base_location}.")
            return None
        
        return action

    def foraging_behavior(self, agent, observation, search_pattern):
        """Determine the agent's action based on its state and environment."""
        carrying = self.get_carrying(agent)
        visible_resources = observation["resources"]
        agent_location = observation["agent_location"]
        base_location = observation["home_base"]

        # # Dynamically adjust the base proximity threshold to avoid base if not carrying
        # base_proximity_threshold = self.adjust_base_proximity_threshold(agent)

        # # Calculate the distance to the base
        # distance_to_base = self.manhattan_distance(agent_location, base_location)

        if carrying:
            # If carrying a resource, return to base
            return self.return_to_base(agent_location, base_location)
        else:
            if visible_resources:
                # If resources are visible, move towards the nearest one
                nearest_resource = min(visible_resources, key=lambda r: self.manhattan_distance(agent_location, r))
                direction_to_resource = self.calculate_direction(agent_location, nearest_resource)
            elif search_pattern == "levy_walk":
                # If no resources are visible, explore randomly
                direction_to_resource = self.levy_walk_direction(agent_location)
            else:
                # Default search pattern: Move towards the base
                direction_to_resource = self.calculate_direction(agent_location, base_location)

            # # Avoid the base if near and not carrying a resource
            # if distance_to_base <= base_proximity_threshold:
            #     direction_to_resource = -self.calculate_direction(agent_location, base_location)  # Direct away from base

            # Sample the direction with added Gaussian noise for imperfect localization
            return self.gaussian_sample(direction_to_resource, self.std_dev_foraging)
        
    def levy_walk_direction(self, current_location):
        """Generate a direction based on a Lévy walk."""
        # Lévy flight parameters
        beta = self.beta  # Lévy exponent (1 < beta <= 2)
        step_length = self.rng.pareto(beta)  # Lévy distributed step length

        # Randomly choose a direction for the Lévy walk
        angle = self.rng.uniform(0, 2 * np.pi)
        direction = np.array([np.cos(angle), np.sin(angle)]) * step_length

        # Normalize and scale direction
        direction = direction / np.linalg.norm(direction) * min(step_length, self.size // 2)
        return direction


    def manhattan_distance(self, a, b):
        """Calculate the Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def decide_action(self, agent):
        """Decide on an action for the agent based on its state and log the state."""
        observation = self.observe(agent)
        battery_level = observation['battery_level']
        _min_battery_level_level = 20  # Threshold for returning to base
        carrying = self.get_carrying(agent)

        # Determine state based on conditions
        if carrying or self.should_return_to_base(battery_level, _min_battery_level_level):
            state = "Returning to Base"
            action = self.return_to_base(self.get_agent_location(agent), self.get_home_base_location())
            if action is not None:
                self.log_agent_state(agent, observation, state)
                return action

        # Default state: Foraging
        state = "Foraging"
        action = self.foraging_behavior(agent, observation, self.search_pattern)

        self.log_agent_state(agent, observation, state)
        return action
    
    def log_agent_state(self, agent, observation, state):
        """Log the agent's state, location, and other important details."""
        if self.debug:
            log_msg = (
                f"----------------------------------------\n"
                f"Agent {agent} post-step:\n"
                f"- State: {state}\n"
                f"- Location: {self.get_agent_location(agent)}\n"
                f"- Carrying: {self.get_carrying(agent)}\n"
                f"- Money: {observation['money']}\n"
                f"- Battery Level: {observation['battery_level']}\n"
                f"----------------------------------------"
            )
            print(log_msg)

    def step(self, action):
        """Extend the step function to handle purchases and auction functionality."""
        # Call the base class's step function to maintain existing functionality
        agent = self.agent_selection  # Get the current agent
        observation, reward, terminated, truncation, info = super().step(action)

        if terminated or truncation:
            return observation, reward, terminated, truncation, info

        # Decrement battery after each step
        self._decrement_battery(agent)

        # Check if the agent has received a reward (i.e., returned a resource)
        if reward > 0:
            self._money[agent] += self._resource_reward
            if self.debug:
                print(f"Agent {agent} returned a resource and earned {self._resource_reward} money. Total Money: {self._money[agent]}.")

        # Battery threshold for recharging
        if self._battery_level[agent] < self.full_battery_charge * self._battery_recharge_threshold:  # Only recharge if below 50%
            # Automatically purchase battery charges with available money if at home base
            if np.array_equal(self.get_agent_location(agent), self.get_home_base_location()):
                self.purchase_battery_charge(agent)

        # Ensure all observations and updates are consistent
        new_observation = self.observe(agent)
        
        return new_observation, reward, terminated, truncation, info

    def _decrement_battery(self, agent):
        """Decrement the battery level of an agent."""
        if self._battery_level[agent] > 0:
            self._battery_level[agent] -= self._battery_usage_rate
            if self.debug:
                print(f"Agent {agent} used battery charge. Current battery level: {self._battery_level[agent]}")

    def purchase_battery_charge(self, agent):
        """Purchase battery charge using the agent's money if at the home base, with a cap at full battery charge."""
        if self.debug:            
            print(f"Agent {agent} - Initial Money: {self._money[agent]}, Initial Battery: {self._battery_level[agent]}")
        
        while self._money[agent] >= self._battery_charge_cost and self._battery_level[agent] < self.full_battery_charge:
            charge_needed = self.full_battery_charge - self._battery_level[agent]
            charge_to_purchase = min(self._battery_charge_amount, charge_needed)

            # Debug: Check values before purchasing
            if self.debug:
                print(f"Attempting purchase: Charge Needed: {charge_needed}, Charge to Purchase: {charge_to_purchase}, Current Battery: {self._battery_level[agent]}, Money: {self._money[agent]}")
            
            # Deduct the cost and increase the battery level
            self._money[agent] -= self._battery_charge_cost
            self._battery_level[agent] += charge_to_purchase

            # Debug: Check values after purchasing
            if self.debug:
                print(f"Agent {agent} purchased {charge_to_purchase} battery charge for {self._battery_charge_cost} money. Remaining Money: {self._money[agent]}, New Battery Level: {self._battery_level[agent]}")

            if self._battery_level[agent] >= self.full_battery_charge:
                self._battery_level[agent] = self.full_battery_charge  # Ensure it doesn't exceed the max
                if self.debug:
                    print(f"Agent {agent} has reached full battery capacity: {self._battery_level[agent]}.")
                break

        if self.debug:
            print(f"Agent {agent} - Final Money: {self._money[agent]}, Final Battery: {self._battery_level[agent]}")

    def observe(self, agent):
        """Extend observation to include nearby agents' ID and position."""
        observation = super().observe(agent)

        # Get nearby agents
        nearby_agents = self.get_nearby_agents(agent)

        # Include nearby agents' ID and positions in the observation
        agent_info = []
        for other_agent in nearby_agents:
            agent_pos = self.get_agent_location(other_agent)
            agent_info.append({
                'id': other_agent,
                'position': agent_pos,
            })

        # Add nearby agents' information to the observation
        observation['nearby_agents'] = agent_info

        # Include money in the observation
        observation['money'] = self.get_money(agent)  # Assuming get_money is a method that returns the agent's money
        
        return observation
    
    def get_nearby_agents(self, agent):
        nearby_agents = []
        agent_pos = self.get_agent_location(agent)
        
        # Get FOV corners
        tl_y, tl_x, br_y, br_x = self.get_fov_corners(agent_pos, self.fov)

        # Iterate through the grid slice and find nearby agents
        for y in range(tl_y, br_y):
            for x in range(tl_x, br_x):
                for other_agent, other_agent_pos in self._agent_locations.items():
                    if other_agent != agent and np.array_equal(other_agent_pos, [y, x]):
                        nearby_agents.append(other_agent)

        return nearby_agents
 
    def is_in_local_range(self, agent_pos, other_agent_pos):
        """Check if another agent is within the local interaction range (FOV) using grid-based lookup."""
        tl_y, tl_x, br_y, br_x = self.get_fov_corners(agent_pos, self.local_interaction_range)

        # Check if the other agent's position falls within the FOV boundaries
        return tl_y <= other_agent_pos[0] < br_y and tl_x <= other_agent_pos[1] < br_x


    def render(self, mode="human"):
        """Call the existing _render method to avoid NotImplementedError."""
        return self._render()  # This will call the existing _render function in the base class
  
    def get_money(self, agent):
        """Retrieve the money of an agent."""
        return self._money[agent]  # Adjust this based on your actual implementation

    def calculate_local_density(self, agent):
        """Calculate the number of agents within a local interaction range using grid-based lookup."""
        agent_location = self.get_agent_location(agent)
        local_density = 0

        # Get FOV corners for the agent
        tl_y, tl_x, br_y, br_x = self.get_fov_corners(agent_location, self.local_interaction_range)

        # Iterate through the grid slice to count nearby agents
        for y in range(tl_y, br_y):
            for x in range(tl_x, br_x):
                for other_agent, other_agent_location in self._agent_locations.items():
                    if other_agent != agent and np.array_equal(other_agent_location, [y, x]):
                        local_density += 1

        return local_density
