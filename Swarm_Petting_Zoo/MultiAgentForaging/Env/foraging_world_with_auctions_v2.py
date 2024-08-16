from foraging_world_v1 import ForagingEnvironment  # Importing your base environment
import numpy as np

class ForagingEnvironmentWithAuction(ForagingEnvironment):
    def __init__(self, num_agents, render_mode=None, size=20, seed=255, num_resources=5, fov=2, show_fov=False, show_gridlines=False, draw_numbers=False, record_sim=False):
        super().__init__(num_agents, render_mode, size, seed, num_resources, fov, show_fov, show_gridlines, draw_numbers, record_sim)
        self.local_interaction_range = fov  # Set the interaction range equal to FOV
        self.auction_history = []  # To track past auctions if needed
        self.initialize_agents()  # Initialize agent-specific attributes

    def initialize_agents(self):
        self._money = {agent: 100 for agent in self.agents}  # Starting money can be adjusted
        self._resource_reward = 50  # Amount paid for returning a resource
        self._battery_usage_rate = 1  # Example battery usage rate per step
        self._battery_charge_cost = 10  # Cost to charge battery
        self._battery_charge_amount = 10  # Amount of charge per purchase

    def move_towards_base(self, agent):
        current_location = self.get_agent_location(agent)
        home_base_location = self.get_home_base_location()
        direction = np.array(home_base_location) - np.array(current_location)
        if abs(direction[0]) > abs(direction[1]):
            return 0 if direction[0] > 0 else 2  # Move right or left
        else:
            return 1 if direction[1] > 0 else 3  # Move down or up

    def decide_action(self, agent):
        if self.get_carrying(agent):
            return self.move_towards_base(agent)
        else:
            return self.action_space.sample()

    def check_agent_state(self, agent, observation):
        if observation['battery_level'] <= 0:
            self.terminations[agent] = True
            print(f"Agent {agent} battery depleted and is now terminated.")
            return True  # Indicate that the agent should be terminated
        return False

    def log_agent_state(self, agent, observation):
        log_msg = (
            f"----------------------------------------\n"
            f"Agent {agent} post-step:\n"
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
        _, reward, terminated, truncation, info = super().step(action)
        agent = self.agent_selection  # Get the current agent

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
        self.log_agent_state(agent, new_observation)

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