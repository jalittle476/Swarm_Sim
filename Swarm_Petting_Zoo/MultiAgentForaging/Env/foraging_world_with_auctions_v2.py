from foraging_world_v1 import ForagingEnvironment  # Importing your base environment
import numpy as np

class ForagingEnvironmentWithAuction(ForagingEnvironment):
    def __init__(self, num_agents, render_mode=None, size=20, seed=255, num_resources=5, fov=5, show_fov=False, show_gridlines=False, draw_numbers=False, record_sim=False):
        super().__init__(num_agents, render_mode, size, seed, num_resources, fov, show_fov, show_gridlines, draw_numbers, record_sim)
        self.local_interaction_range = fov  # Set the interaction range equal to FOV
        self.auction_history = []  # To track past auctions if needed
        self.initialize_agents()  # Initialize agent-specific attributes


        # Default standard deviations for different behaviors
        self.std_dev_base_return = 0.5
        self.std_dev_foraging = 0.1

    def initialize_agents(self):
        self._money = {agent: 100 for agent in self.agents}
        self._resource_reward = 50
        self._battery_usage_rate = 1
        self._battery_charge_cost = 10
        self._battery_charge_amount = 10
        
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

    def gaussian_sample(self, direction, std_dev):
        """Sample a discrete action based on the direction vector with added Gaussian noise."""
        # Normalize the direction vector
        norm = np.linalg.norm(direction)
        if norm != 0:
            direction = direction / norm
        
        sampled_direction = np.random.normal(direction, std_dev)
        print(f"Original direction: {direction}, Sampled direction: {sampled_direction}")
        
        if abs(sampled_direction[0]) > abs(sampled_direction[1]):
            return 0 if sampled_direction[0] > 0 else 2  # Move right or left
        else:
            return 1 if sampled_direction[1] > 0 else 3  # Move down or up

    def should_return_to_base(self, battery_level, min_battery_level):
        """Check if the agent should return to the base based on its battery level."""
        return battery_level <= min_battery_level

    def return_to_base(self, agent_location, base_location):
        """Generate an action to return the agent to the base."""
        if np.array_equal(agent_location, base_location):
            return None  # No action needed, agent is already at the base
        direction_to_base = self.calculate_direction(agent_location, base_location)
        return self.gaussian_sample(direction_to_base, self.std_dev_base_return)

    def foraging_behavior(self, agent, observation):
        """Determine the agent's action based on its state and environment."""
        carrying = self.get_carrying(agent)
        visible_resources = observation["resources"]
        agent_location = observation["agent_location"]
        base_location = observation["home_base"]
       # Dynamically adjust the base proximity threshold to avoid base if not carrying
        base_proximity_threshold = self.adjust_base_proximity_threshold(agent)

        distance_to_base = np.linalg.norm(self.calculate_direction(agent_location, base_location))

        if carrying:
            return self.return_to_base(agent_location, base_location)
        else:
            if visible_resources:
                nearest_resource = min(visible_resources, key=lambda r: self.manhattan_distance(agent_location, r))
                direction_to_resource = self.calculate_direction(agent_location, nearest_resource)
            else:
                direction_to_resource = np.random.normal(0, self.std_dev_foraging, 2)  # Random exploration direction

            # # Avoid base if near and not carrying a resource
            # if distance_to_base <= base_proximity_threshold:
            #     direction_to_resource = -self.calculate_direction(agent_location, base_location)  # Direct away from base

            return self.gaussian_sample(direction_to_resource, self.std_dev_foraging)

    def manhattan_distance(self, a, b):
        """Calculate the Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def decide_action(self, agent):
        """Decide on an action for the agent based on its state and log the state."""
        observation = self.observe(agent)
        battery_level = observation['battery_level']
        min_battery_level = 10  # Threshold for returning to base
        carrying = self.get_carrying(agent)

        # Determine state based on conditions
        if carrying or self.should_return_to_base(battery_level, min_battery_level):
            state = "Returning to Base"
            action = self.return_to_base(self.get_agent_location(agent), self.get_home_base_location())
            if action is not None:
                self.log_agent_state(agent, observation, state)
                return action

        # Default state: Foraging
        state = "Foraging"
        action = self.foraging_behavior(agent, observation)

        self.log_agent_state(agent, observation, state)
        return action

    def check_agent_state(self, agent, observation):
        if observation['battery_level'] <= 0:
            self.terminations[agent] = True
            print(f"Agent {agent} battery depleted and is now terminated.")
            return True  # Indicate that the agent should be terminated
        return False

    def log_agent_state(self, agent, observation, state):
        """Log the agent's state, location, and other important details."""
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
        _, reward, terminated, truncation, info = super().step(action)

        # Decrement battery after each step
        self._decrement_battery(agent)

        # Check if the agent has received a reward (i.e., returned a resource)
        if reward > 0:
            self._money[agent] += self._resource_reward
            print(f"Agent {agent} returned a resource and earned {self._resource_reward} money. Total Money: {self._money[agent]}.")

        # Automatically purchase battery charges with available money if at home base
        if np.array_equal(self.get_agent_location(agent), self.get_home_base_location()):
            self.purchase_battery_charge(agent)

        # Ensure all observations and updates are consistent
        new_observation = self.observe(agent)

        # Log post-step status for debugging
       #self.log_agent_state(agent, new_observation,self.agent_states[agent])

        return new_observation, reward, terminated, truncation, info

    def _decrement_battery(self, agent):
        """Decrement the battery level of an agent."""
        if self._battery_level[agent] > 0:
            self._battery_level[agent] -= self._battery_usage_rate
            print(f"Agent {agent} used battery charge. Current battery level: {self._battery_level[agent]}")
        if self._battery_level[agent] <= 0:
            self.terminations[agent] = True  # Terminate agent if battery is depleted
            print(f"Agent {agent} battery depleted and is now terminated.")

    def purchase_battery_charge(self, agent):
        """Purchase battery charge using the agent's money if at the home base, with a cap at full battery charge."""
        print(f"Agent {agent} - Initial Money: {self._money[agent]}, Initial Battery: {self._battery_level[agent]}")
        
        while self._money[agent] >= self._battery_charge_cost and self._battery_level[agent] < self.full_battery_charge:
            charge_needed = self.full_battery_charge - self._battery_level[agent]
            charge_to_purchase = min(self._battery_charge_amount, charge_needed)

            # Debug: Check values before purchasing
            print(f"Attempting purchase: Charge Needed: {charge_needed}, Charge to Purchase: {charge_to_purchase}, Current Battery: {self._battery_level[agent]}, Money: {self._money[agent]}")
            
            # Deduct the cost and increase the battery level
            self._money[agent] -= self._battery_charge_cost
            self._battery_level[agent] += charge_to_purchase

            # Debug: Check values after purchasing
            print(f"Agent {agent} purchased {charge_to_purchase} battery charge for {self._battery_charge_cost} money. Remaining Money: {self._money[agent]}, New Battery Level: {self._battery_level[agent]}")

            if self._battery_level[agent] >= self.full_battery_charge:
                self._battery_level[agent] = self.full_battery_charge  # Ensure it doesn't exceed the max
                print(f"Agent {agent} has reached full battery capacity: {self._battery_level[agent]}.")
                break

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
        """Detect agents within the local interaction range (FOV)."""
        nearby_agents = []
        agent_pos = self.get_agent_location(agent)
        for other_agent in self.agents:
            if other_agent != agent:
                other_agent_pos = self.get_agent_location(other_agent)
                if self.is_in_local_range(agent_pos, other_agent_pos):
                    nearby_agents.append(other_agent)
        return nearby_agents

    def is_in_local_range(self, agent_pos, other_agent_pos):
        """Check if another agent is within the local interaction range."""
        distance = self.calculate_distance(agent_pos, other_agent_pos)
        return distance <= self.local_interaction_range

    def calculate_distance(self, pos1, pos2):
        """Calculate the Euclidean distance between two positions."""
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def render(self, mode="human"):
        """Call the existing _render method to avoid NotImplementedError."""
        return self._render()  # This will call the existing _render function in the base class
  
    def get_money(self, agent):
        """Retrieve the money of an agent."""
        return self._money[agent]  # Adjust this based on your actual implementation

    def calculate_local_density(self, agent):
        """Calculate the number of agents within a local interaction range."""
        agent_location = self.get_agent_location(agent)
        local_density = 0
        
        for other_agent in self.agents:
            if other_agent != agent:  # Don't count the agent itself
                other_agent_location = self.get_agent_location(other_agent)
                distance_to_other_agent = np.linalg.norm(self.calculate_direction(agent_location, other_agent_location))
                if distance_to_other_agent <= self.local_interaction_range:
                    local_density += 1
        
        return local_density