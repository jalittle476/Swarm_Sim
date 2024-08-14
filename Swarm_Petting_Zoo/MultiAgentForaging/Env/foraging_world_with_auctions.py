

 #Extending the ForagingEnvironment to include auction capabilities


from foraging_world_v1 import ForagingEnvironment  # Importing your base environment
import numpy as np

class ForagingEnvironmentWithAuction(ForagingEnvironment):
    def __init__(self, num_agents, render_mode=None, size=20, seed=255, num_resources=5, fov=2, show_fov=False, show_gridlines=False, draw_numbers=False, record_sim=False):
        super().__init__(num_agents, render_mode, size, seed, num_resources, fov, show_fov, show_gridlines, draw_numbers, record_sim)
        self.local_interaction_range = fov  # Set the interaction range equal to FOV
        self.auction_history = []  # To track past auctions if needed

        # Initialize money for each agent
        self._money = {agent: 100 for agent in self.agents}  # Starting money can be adjusted
        self._resource_reward = 50  # Amount paid for returning a resource
        self._battery_charge_cost = 10  # Cost of one battery charge
        self._battery_charge_amount = 20  # Amount of battery gained per purchase

    # def step(self, action):
    #     """Extend the step function to include auction functionality."""
    #     #Call the base class's step function to maintain existing functionality
    #     _, reward, terminated, truncation, info = super().step(action)
    #     agent = self.agent_selection  # Get the current agent

    #     #check if the agent has returned to the base with a resource 
    #     if np.array_equal(self.get_agent_location(agent), self.get_home_base_location()) and self.get_carrying(agent):
    #         self._money[agent] += self._resource_reward
    #         print(f"Agent {agent} returned a resource and received a reward of {self._resource_reward}.")
            
    #  # Automatically purchase battery charges with available money
    #     if np.array_equal(self.get_agent_location(agent), self.get_home_base_location()):
    #         self.purchase_battery_charge(agent)

    #     new_observation = self.observe(agent)

    #     return new_observation, reward, terminated, truncation, info
    
    def step(self, action):
        """Extend the step function to include auction functionality."""
        # Call the base class's step function to maintain existing functionality
        observation, reward, terminated, truncation, info = super().step(action)
        agent = self.agent_selection  # Get the current agent

        # Calculate the intended new location based on the action
        direction = self._action_to_direction[action]
        intended_location = np.clip(self.get_agent_location(agent) + direction, 0, self.size - 1)

        # Print the action and intended movement
        print(f"Agent {agent} is taking action: {action}, Current Location: {self.get_agent_location(agent)}, Intended Location: {intended_location}")

        # Check if the move is valid
        if self._is_location_valid(agent, intended_location):
            print(f"Move is valid. Agent {agent} will move to {intended_location}.")
        else:
            print(f"Move is invalid. Agent {agent} remains at {self.get_agent_location(agent)}.")

        # Continue with the base step logic
        if np.array_equal(self.get_agent_location(agent), self.get_home_base_location()) and self.get_carrying(agent):
            self._money[agent] += self._resource_reward
            print(f"Agent {agent} returned a resource and received a reward of {self._resource_reward}.")

        if np.array_equal(self.get_agent_location(agent), self.get_home_base_location()):
            self.purchase_battery_charge(agent)

        new_observation = self.observe(agent)

        # Print post-step status
        print(f"Agent {agent} post-step: Location: {self.get_agent_location(agent)}, Carrying: {self.get_carrying(agent)}, Money: {self._money[agent]}, Battery Level: {self._battery_level[agent]}")

        return new_observation, reward, terminated, truncation, info




    def purchase_battery_charge(self, agent):
        """Purchase battery charge using the agent's money if at the home base, with a cap at full battery charge."""
        while self._money[agent] >= self._battery_charge_cost and self._battery_level[agent] < self.full_battery_charge:
            # Calculate how much more battery the agent can purchase without exceeding the full charge
            charge_needed = self.full_battery_charge - self._battery_level[agent]
            charge_to_purchase = min(self._battery_charge_amount, charge_needed)

            # Deduct the cost and increase the battery level
            self._money[agent] -= self._battery_charge_cost
            self._battery_level[agent] += charge_to_purchase
            print(f"Agent {agent} purchased {charge_to_purchase} battery charge for {self._battery_charge_cost} money.")
            
            # Stop if the battery is full
            if self._battery_level[agent] >= self.full_battery_charge:
                self._battery_level[agent] = self.full_battery_charge  # Ensure it doesn't exceed the max
                print(f"Agent {agent} has reached full battery capacity: {self._battery_level[agent]}.")
                break



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
        # Assuming money is tracked in a dictionary like self._money[agent]
        return self._money[agent]  # Adjust this based on your actual implementation