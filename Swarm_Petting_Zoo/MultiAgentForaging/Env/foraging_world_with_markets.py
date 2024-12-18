from foraging_world_v1 import ForagingEnvironment  # Importing your base environment
import numpy as np
import pandas as pd
import random
from foraging_config import ForagingConfig

class ForagingEnvironmentWithMarkets(ForagingEnvironment):
    def __init__(self, config: ForagingConfig):
        super().__init__(config)  # Initialize the base class with the provided config
        
        # Additional initialization for the subclass
        self.rng = np.random.default_rng(self.seed)  # Initialize the Generator
        self.initialize_agents()  # Initialize agent-specific attributes
        
        # Initialize the data log to store all records
        self.log_data = []  # This will collect each step's data

        self.__dict__.update(config.__dict__)

    def initialize_agents(self):
        
        self._money = {agent: self.initial_money for agent in self.agents}
        
        # Initialize the state of each agent
        self.agent_states = {agent: "Foraging" for agent in self.agents}  # Default state is "Foraging"
        self.exchange_count = {agent: 0 for agent in self.agents}  # Track exchanges per agent

        
        # Tracking resources gathered by each agent and total resources
        self.resources_gathered_per_agent = {agent: 0 for agent in self.agents}
        self.total_resources_gathered = 0
        
          # Initialize auction-related attributes
        self._exchange_seller = None
        self._exchange_buyer = None
        self._exchange_bid = None
         # Initialize the recent trades dictionary
        self.recent_trades = {agent: set() for agent in self.agents}
        
        # Initialize target locations for exchange
        self._target_location = {}  # Empty dictionary to store target locations for each agent during exchanges
        
        """Initialize direction and steps remaining for each agent."""
        grid_directions = [np.array([0, 1]), np.array([0, -1]), np.array([-1, 0]), np.array([1, 0])]  # Up, Down, Left, Right

        self.current_direction = {}  # Store the current direction for each agent
        self.steps_remaining_in_direction = {}  # Store the number of steps remaining in the current direction for each agent
    
        for agent in self.agents:
            # Start each agent with a random direction
            self.current_direction[agent] = self.rng.choice(grid_directions)
            self.steps_remaining_in_direction[agent] = 0  # Start with 0 remaining steps; will be set on first Lévy walk
        
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
        
        # Ensure direction is a 2D vector
        if not isinstance(direction, np.ndarray) or direction.shape != (2,):
            raise ValueError(f"Direction must be a 2D vector. Received: {direction}")

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
    
    def move_towards_location(self, agent_location, target_location):
        """Generate an action to move the agent towards a specified target location."""
        if np.array_equal(agent_location, target_location):
            return None  # No movement needed, agent is already at the target location

        direction_to_target = self.calculate_direction(agent_location, target_location)
        action = self.gaussian_sample(direction_to_target, self.std_dev_move)

        # If the action results in no movement, handle it here
        if action is None:
            if self.debug:
                print(f"Agent at {agent_location} chose not to move towards target at {target_location}.")
            return None

        return action

    def should_return_to_base(self, battery_level):
        """Check if the agent should return to the base based on its battery level."""
        min_battery_level = self.get_manhattan_distance_to_base(self.agent_selection)
        return battery_level <= min_battery_level

    def return_to_base(self, agent_location, base_location):
        """Generate an action to return the agent to the base."""
        if np.array_equal(agent_location, base_location):
            return None  # No action needed, agent is already at the base
        
        direction_to_base = self.calculate_direction(agent_location, base_location)
        action = self.gaussian_sample(direction_to_base, self.std_dev_base_return)
        
        return action

    def foraging_behavior(self, agent, observation, search_pattern):
        """Determine the agent's action based on its state and environment."""
        carrying = self.get_carrying(agent)
        visible_resources = observation["resources"]
        agent_location = observation["agent_location"]
        base_location = observation["home_base"]

        # Determine the current direction for the agent
        if search_pattern == "levy_walk" and not visible_resources:
            next_location = agent_location + self.current_direction[agent]
            if self.steps_remaining_in_direction[agent] > 0 and self._is_location_valid(agent, next_location):
                # Continue in the current direction if steps remain
                current_direction = self.current_direction[agent]
                self.steps_remaining_in_direction[agent] -= 1
            else:
                # Choose a new direction using the Lévy walk
                current_direction = self.levy_walk_direction(agent,self.current_direction[agent])
                self.current_direction[agent] = current_direction
                self.steps_remaining_in_direction[agent] = int(self.rng.pareto(self.beta))  # Update steps to persist

        elif visible_resources:
            # If resources are visible, move towards the nearest one
            nearest_resource = min(visible_resources, key=lambda r: self.manhattan_distance(agent_location, r))
            current_direction = self.calculate_direction(agent_location, nearest_resource)

        else:
            # Default search pattern: Move towards the base
            current_direction = self.calculate_direction(agent_location, base_location)

        # Decide the next action based on the agent's state
        if carrying:
            # If carrying a resource, return to base
            direction_to_resource = self.return_to_base(agent_location, base_location)
        else:
            # If not carrying, use the current direction decided above
            direction_to_resource = current_direction

        # Sample the direction with added Gaussian noise for imperfect localization
        return self.gaussian_sample(direction_to_resource, self.std_dev_foraging)
  
    def levy_walk_direction(self, agent, current_direction):
        """Generate a direction for a Lévy walk in a grid world with a bias towards forward movement.
        
        Args:
            current_direction (np.array): The current direction in which the agent is moving.
            agent: The agent for which to compute the new direction.
        
        Returns:
            np.array: The next direction for the agent to move in.
        """
        
        # Lévy flight parameters
        beta = self.beta  # Lévy exponent (1 < beta <= 2)
        step_length = int(self.rng.pareto(beta))  # Lévy distributed step length, converted to an integer

        # Possible directions in the grid (up, down, left, right)
        grid_directions = [
            np.array([0, 1]),   # up
            np.array([0, -1]),  # down
            np.array([-1, 0]),  # left
            np.array([1, 0])    # right
        ]

        # Bias parameter: probability of continuing in the same direction
        forward_bias = 0.7  # Adjust this value between 0 and 1 for more or less bias

        # Decide whether to continue in the current direction or choose a new one
        if self.rng.uniform(0, 1) < forward_bias:
            # Continue in the current direction
            new_direction = current_direction
        else:
            # Choose a new random direction different from the current one
            new_direction = self.rng.choice([d for d in grid_directions if not np.array_equal(d, current_direction)])

        # Store the number of steps to persist in this direction for the specific agent
        self.steps_remaining_in_direction[agent] = step_length  # Persist for 'step_length' moves

        return new_direction

    def manhattan_distance(self, a, b):
        """Calculate the Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def decide_action(self, agent):
        """Decide on an action for the agent based on its state and log the state."""
        observation = self.observe(agent)
        battery_level = observation['battery_level']
        carrying = self.get_carrying(agent)

        # Update the agent color whenever a decision is made
        self._update_agent_color(agent)
        
     
        if agent in [self._exchange_seller, self._exchange_buyer]:
            # Agent is in an auction exchange state
            self.agent_states[agent] = "Exchanging"
            action = self.execute_exchange(agent)  # Move toward the other agent
            
        elif not carrying and self.should_return_to_base(battery_level):
            # Agent needs to return to base due to low battery
            self.agent_states[agent] = "Returning to Base"
            action = self.return_to_base(self.get_agent_location(agent), self.get_home_base_location())
            
        elif carrying:
            # Agent is carrying a resource and returning to base
            action = self.return_to_base(self.get_agent_location(agent), self.get_home_base_location())
            if agent not in [self._exchange_seller, self._exchange_buyer]:
                self.agent_states[agent] = "Returning to Base"
                self.initiate_auction(agent)  # Initiate an auction process for the agent
                
        else:
            # Default state: Foraging
            self.agent_states[agent] = "Foraging"
            action = self.foraging_behavior(agent, observation, self.search_pattern)

        # Log the agent's state after deciding the action
        self.log_agent_data(agent)
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
        """Extend the step function to handle purchases, auction functionality, and dead agents returning to base."""
        agent = self.agent_selection  # Get the current agent
        observation, reward, terminated, truncation, info = super().step(action)

        # Check if the agent is dead and needs to return to the base
        if self._battery_level[agent] == 0:
            self.agent_states[agent] = "Dead"
            
            if not np.array_equal(self.get_agent_location(agent), self.get_home_base_location()):
                # Agent is dead and not at the base, so prevent termination to allow movement
                self.terminations[agent] = False
                action = self.return_to_base(self.get_agent_location(agent), self.get_home_base_location())
                self.update_agent_location(agent, self._agent_locations[agent] + self._action_to_direction[action])
                observation = self.observe(agent)  # Update observation after movement
            else:
                # Agent has reached the base, finalize termination
                self.terminations[agent] = True
                action = None  # No further actions required

        # Additional logic for handling living agents
        elif not terminated and not truncation:
            # Decrement battery after each step
            self._decrement_battery(agent)

            # Check if the agent has returned a resource and received a reward
            if reward > 0:
                self._money[agent] += self.resource_reward
                reward = 0  # Reset the reward to prevent double counting
                self.resources_gathered_per_agent[agent] += 1
                self.total_resources_gathered += 1
                if self.debug:
                    print(f"Agent {agent} returned a resource and earned {self.resource_reward} money. Total Money: {self._money[agent]}.")

            # Handle recharging if at the base and low on battery
            if self._battery_level[agent] < self.full_battery_charge * self.battery_recharge_threshold:
                if np.array_equal(self.get_agent_location(agent), self.get_home_base_location()):
                    self.purchase_battery_charge(agent)

        # Ensure all observations and updates are consistent
        new_observation = self.observe(agent)

        return new_observation, reward, terminated, truncation, info

    
    def adjust_currency(self, agent, amount):
        """Adjust the agent's currency by the specified amount."""
        if agent not in self._money:
            self._money[agent] = 0  # Initialize the agent's money if not already done

        # Update the agent's currency
        self._money[agent] += amount

        # Ensure currency doesn't drop below zero if that rule is desired
        if self._money[agent] < 0:
            self._money[agent] = 0  # Prevent negative currency balance

        print(f"Agent {agent} currency adjusted by {amount}. New balance: {self._money[agent]}")

    def _decrement_battery(self, agent):
        """Decrement the battery level of an agent."""
        if self._battery_level[agent] > 0:
            self._battery_level[agent] -= self.battery_usage_rate
            if self.debug:
                print(f"Agent {agent} used battery charge. Current battery level: {self._battery_level[agent]}")

    def purchase_battery_charge(self, agent):
        """Purchase battery charge using the agent's money if at the home base, with a cap at full battery charge."""
        if self.debug:            
            print(f"Agent {agent} - Initial Money: {self._money[agent]}, Initial Battery: {self._battery_level[agent]}")
        
        while self._money[agent] >= self.battery_charge_cost and self._battery_level[agent] < self.full_battery_charge:
            charge_needed = self.full_battery_charge - self._battery_level[agent]
            charge_to_purchase = min(self.battery_charge_amount, charge_needed)

            # Debug: Check values before purchasing
            if self.debug:
                print(f"Attempting purchase: Charge Needed: {charge_needed}, Charge to Purchase: {charge_to_purchase}, Current Battery: {self._battery_level[agent]}, Money: {self._money[agent]}")
            
            # Deduct the cost and increase the battery level
            self._money[agent] -= self.battery_charge_cost
            self._battery_level[agent] += charge_to_purchase

            # Debug: Check values after purchasing
            if self.debug:
                print(f"Agent {agent} purchased {charge_to_purchase} battery charge for {self.battery_charge_cost} money. Remaining Money: {self._money[agent]}, New Battery Level: {self._battery_level[agent]}")

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

        # Iterate over other agents to find nearby agents
        for other_agent, other_agent_pos in self._agent_locations.items():
            if other_agent != agent:  # Exclude the current agent itself
                # Check if the other agent is within the FOV bounds
                if tl_y <= other_agent_pos[0] < br_y and tl_x <= other_agent_pos[1] < br_x:
                    nearby_agents.append(other_agent)

        return nearby_agents
 
    def render(self, mode="human"):
        """Call the existing _render method to avoid NotImplementedError."""
        return self._render()  # This will call the existing _render function in the base class
  
    def get_money(self, agent):
        """Retrieve the money of an agent."""
        return self._money[agent]  # Adjust this based on your actual implementation

    def calculate_local_density(self, agent):
        """Calculate the number of agents within the agent's FOV."""
        return len(self.get_nearby_agents(agent))

    def initiate_auction(self, seller_agent):
        """Initiate an auction and prepare for resource exchange."""
        # Ensure no exchange is currently in progress
        if self._exchange_seller is not None or self._exchange_buyer is not None:
            #print(f"Auction blocked for {seller_agent} because an exchange is in progress.")
            return False

        # Generate the reserve price based on the seller's utility function
        reserve_price = self.calculate_reserve_price(seller_agent)

        # Get agents within the seller's FOV
        nearby_agents = self.get_nearby_agents(seller_agent)

        # # Collect bids from nearby agents
        # bids = {}
        # for agent in nearby_agents:
        #     bid = self.calculate_bid(agent)
        #     if bid > reserve_price:
        #         bids[agent] = bid
        
          # Collect bids from nearby agents, excluding recent trading partners
        bids = {}
        for agent in nearby_agents:
            if agent not in self.recent_trades[seller_agent]:  # Exclude recent trading partners
                bid = self.calculate_bid(agent)
                if bid > reserve_price:
                    bids[agent] = bid

        # Determine the highest bid
        if bids:
            winning_agent = max(bids, key=bids.get)
            winning_bid = bids[winning_agent]
            self._exchange_seller = seller_agent
            self._exchange_buyer = winning_agent
            self._exchange_bid = winning_bid
            self.agent_states[seller_agent] = "Exchanging"
            self.agent_states[winning_agent] = "Exchanging"
            print(f"{winning_agent} wins the auction with a bid of {winning_bid}. Preparing for exchange.")
            
            
            # Update recent trades
            self.recent_trades[seller_agent].add(winning_agent)
            self.recent_trades[winning_agent].add(seller_agent)
            
            return True
        else:
            # No bids meet the reserve price; auction fails
            #print(f"No bids higher than the reserve price of {reserve_price}. Auction failed.")
            return False


    def execute_exchange(self, agent):
        """Handle the agent's movement toward the other agent and complete the exchange if they are adjacent."""
       
        if agent == self._exchange_seller:
            self._carrying[agent] = False  # Treat seller as if they are not carrying a resource
        
        # Identify the target agent for the exchange
        target_agent = self._exchange_buyer if agent == self._exchange_seller else self._exchange_seller
        target_location = self.get_agent_location(target_agent)
        agent_location = self.get_agent_location(agent)

        # Calculate the direction towards the target agent
        direction_to_target = self.calculate_direction(agent_location, target_location)
        action = self.gaussian_sample(direction_to_target, self.std_dev_move)

        # Check if agents are adjacent (Manhattan distance of 1)
        manhattan_distance = self.manhattan_distance(agent_location, target_location)
        if manhattan_distance == 1:
            if agent == self._exchange_seller:
                self._carrying[agent] = True
            self.complete_exchange()  # Call a function to complete the exchange
            # Reset the exchange state after completion
            self._exchange_seller = None
            self._exchange_buyer = None
            self._exchange_bid = None

        return action
    
    def complete_exchange(self):
        """Handle the resource and currency exchange between the seller and buyer."""
        # Transfer resource from seller to buyer
        self._carrying[self._exchange_buyer] = True
        self._carrying[self._exchange_seller] = False

        # Increment exchange counts for both agents
        self.exchange_count[self._exchange_buyer] += 1
        self.exchange_count[self._exchange_seller] += 1

        # Adjust currency for the transaction
        self.adjust_currency(self._exchange_buyer, -self._exchange_bid)
        self.adjust_currency(self._exchange_seller, self._exchange_bid)
        
        # Record the exchange as a separate row
        exchange_location = self.get_agent_location(self._exchange_seller)  # Use seller's location as exchange point
        # Log exchange for buyer
        self.log_data.append({
            "row_type": "exchange",
            "agent_id": self._exchange_buyer,
            "x": exchange_location[0],
            "y": exchange_location[1],
            "exchange_count": 1,
        })

        # Log exchange for seller
        self.log_data.append({
            "row_type": "exchange",
            "agent_id": self._exchange_seller,
            "x": exchange_location[0],
            "y": exchange_location[1],
            "exchange_count": 1,
        })

        print(f"Exchange completed between {self._exchange_seller} and {self._exchange_buyer}.")

    def calculate_reserve_price(self, agent):
        """Calculate the minimum selling price based on the agent's utility and the home base resource reward."""
        opportunity_cost = self.calculate_opportunity_cost(agent, selling=True)
        
        return opportunity_cost  # Ensure reserve price reflects resource value

    def calculate_bid(self, agent):
        """Calculate the bid price based on the agent's utility and the potential profit at the home base,
        constrained by the agent's available currency."""
        
        opportunity_cost = self.calculate_opportunity_cost(agent, selling=False)
        available_money = self._money.get(agent, 0)  # Get the agent's available money, default to 0 if not found

        # Bid should be the minimum of the calculated opportunity cost and available money
        bid = min(opportunity_cost, available_money)
        
       # print(f"Agent {agent} calculated bid: {bid} (Opportunity Cost: {opportunity_cost}, Available Money: {available_money})")
        return bid

    
    def calculate_opportunity_cost(self, agent, selling):
        """Estimate opportunity cost based on current state."""
        # Check if the agent is dead
        if self.agent_states[agent] == "Dead" or self.get_battery_level(agent) <= 0:
            return 0  # Dead agents have zero opportunity cost

        battery_level = self.get_battery_level(agent)
        min_battery_level = self.get_manhattan_distance_to_base(agent)

        # Handle edge case: at the base
        if  min_battery_level == 0:
            if selling:
                return 0  # No urgency to sell when already at the base
            else:
                return float('inf')  # No opportunity cost for buying since the agent can recharge freely

        # Avoid division by zero or nonsensical values for low battery
        if battery_level <= min_battery_level:
            return 0  # Minimal opportunity cost when critically low

        if selling:  # Calculate opportunity cost for selling
            opportunity_cost_selling = (
                self.resource_reward
                * (min_battery_level / battery_level)
            )
            return opportunity_cost_selling

        else:  # Calculate opportunity cost for buying
            opportunity_cost_buying = (
                self.battery_charge_cost
                * (1 - min_battery_level / battery_level)
            )
            return opportunity_cost_buying


    def get_battery_level(self, agent):
        return self._battery_level[agent]

    def get_manhattan_distance_to_base(self, agent):
        agent_pos = self.get_agent_location(agent)
        base_pos = self.get_home_base_location()
        return sum(abs(a - b) for a, b in zip(agent_pos, base_pos))

    def get_manhattan_distance_to_visible_resources(self, agent, observation):
        agent_pos = self.get_agent_location(agent)
        visible_resources = observation["resources"]  # Use resources from observation
        if visible_resources:
            distances = [sum(abs(a - b) for a, b in zip(agent_pos, res)) for res in visible_resources]
            return min(distances)  # Closest resource
        else:
            return float('inf')  # No visible resources

    def _update_agent_color(self, agent):
        """
        Update the color of the agent based on its current state.
        This function is called only when the agent's state changes.
        """
        # Retrieve the agent's current state
        agent_state = self.agent_states[agent]

        if agent_state == "Exchanging":
            self.agent_color_cache[agent] = (255, 105, 180)  # Pink for exchange state
        elif agent_state == "Dead" or self._battery_level[agent] <= 0:
            self.agent_color_cache[agent] = (0, 0, 0)  # Black for zero battery while returning
        elif agent_state == "Returning to Base" and self._battery_level[agent] < self.size:
            self.agent_color_cache[agent] = (255, 0, 0)  # Red for low battery while returning
        elif agent_state == "Returning to Base" and self._carrying[agent]:
            self.agent_color_cache[agent] = (0, 102, 0)  # Green if carrying a resource while returning
        else:
            self.agent_color_cache[agent] = (0, 0, 255)  # Blue for foraging or other default state

    def log_agent_data(self, agent):
        """Log relevant data for the agent for each step."""
        agent_data = {
            "row_type": "agent",  # Identify this row as an agent activity row
            "agent_id": agent,
            "state": self.agent_states[agent],
            "location": self.get_agent_location(agent),
            "battery_level": self.get_battery_level(agent),
            "money": self.get_money(agent),
            "carrying": self.get_carrying(agent),
            "nearby_agents": len(self.get_nearby_agents(agent)),  # Example: number of nearby agents
            "exchange_count": self.exchange_count[agent],  # Number of exchanges
            "resources_gathered": self.resources_gathered_per_agent.get(agent, 0),  # Add resources gathered

        }
        self.log_data.append(agent_data)  # Append to the log data

    def save_logs(self, filename="simulation_logs.csv"):
            """Save the log data to a CSV file for analysis."""
            df = pd.DataFrame(self.log_data)
            df.to_csv(filename, index=False)
            print(f"Logs saved to {filename}")

    def generate_summary_table(self, filename="summary_table.csv", file_format="csv"):
        """Generate and save a summary table for all agents."""
        df = pd.DataFrame(self.log_data)

        # Group by agent_id for a summary
        summary_df = df.groupby("agent_id").agg({
            "state": "last",  # Get the last recorded state for each agent
            "battery_level": "mean",  # Average battery level
            "money": "mean",  # Average money
            "carrying": "last",  # Current carrying status
            "nearby_agents": "mean",  # Average number of nearby agents
            "exchange_count": "last",  # Total exchanges
            "resources_gathered": "last"  # Total resources gathered

        }).reset_index()

        # Rename columns for clarity
        summary_df.rename(columns={
            "state": "Final State",
            "battery_level": "Avg Battery Level",
            "money": "Avg Money",
            "carrying": "Current Carrying",
            "nearby_agents": "Avg Nearby Agents",
            "exchange_count": "Total Exchanges",
            "resourches_gathered": "Total Resources Gathered"
        }, inplace=True)

        #print(summary_df)  # Display the summary table in the console

        # Save to file based on file format
        if file_format == "csv":
            summary_df.to_csv(filename, index=False)
            print(f"Summary table saved to '{filename}'")
        elif file_format == "excel":
            summary_df.to_excel(filename, index=False)
            print(f"Summary table saved to '{filename}'")
        else:
            print(f"Unsupported file format: {file_format}. Please use 'csv' or 'excel'.")




