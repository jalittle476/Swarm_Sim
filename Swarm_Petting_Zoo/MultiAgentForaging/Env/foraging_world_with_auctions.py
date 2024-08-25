from foraging_world_with_transactions import ForagingEnvironmentWithTransactions

class ForagingEnvironmentWithAuctions(ForagingEnvironmentWithTransactions):
    def __init__(self, config):
        super().__init__(config)
        self._exchange_seller = None
        self._exchange_buyer = None
        self._exchange_bid = None

    def initiate_auction(self, seller_agent):
        """Initiate an auction and prepare for resource exchange."""
        # Generate the reserve price based on the seller's utility function
        reserve_price = self.calculate_reserve_price(seller_agent)

        # Get agents within the seller's FOV
        nearby_agents = self.get_nearby_agents(seller_agent)

        # Collect bids from nearby agents
        bids = {}
        for agent in nearby_agents:
            bid = self.calculate_bid(agent, seller_agent)
            if bid > reserve_price:
                bids[agent] = bid

        # Determine the highest bid
        if bids:
            winning_agent = max(bids, key=bids.get)
            winning_bid = bids[winning_agent]
            self._exchange_seller = seller_agent
            self._exchange_buyer = winning_agent
            self._exchange_bid = winning_bid
            print(f"{winning_agent} wins the auction with a bid of {winning_bid}. Preparing for exchange.")
        else:
            print(f"No bids higher than the reserve price. Auction failed.")


    def calculate_reserve_price(self, agent):
        """Calculate the minimum selling price based on the agent's utility and the home base resource reward."""
        observation = self.observe(agent)
        battery_level = observation["battery_level"]
        distance_to_base = self.get_manhattan_distance_to_base(agent)
        local_density = len(observation["nearby_agents"])  # Use nearby agents from observation
        opportunity_cost = self.calculate_opportunity_cost(agent, selling=True)
        
        # Factor in the resource reward at the home base into the reserve price
        base_value = self.resource_reward  # The going rate for resources at the home base
        
        # Example formula combining these factors
        reserve_price = (battery_level * 0.5) + (1 / (distance_to_base + 1)) * 10 + (local_density * 2) + opportunity_cost + base_value * 0.5
        
        return max(1, reserve_price)  # Ensure reserve price is at least 1

    def calculate_bid(self, agent, seller_agent):
        """Calculate the bid price based on the agent's utility and the potential profit at the home base."""
        observation = self.observe(agent)
        battery_level = observation["battery_level"]
        distance_to_resources = self.get_manhattan_distance_to_visible_resources(agent, observation)
        local_density = len(observation["nearby_agents"])  # Use nearby agents from observation
        opportunity_cost = self.calculate_opportunity_cost(agent, selling=False)
        
        # Factor in the potential profit from selling the resource at the home base
        potential_profit = self.resource_reward - (battery_level * 0.1)  # Example of adjusting profit based on battery
        
        # Example formula combining these factors
        bid_price = (battery_level * 0.3) + (1 / (distance_to_resources + 1)) * 8 + (local_density * 1.5) + opportunity_cost + potential_profit * 0.5
        
        return max(1, bid_price)  # Ensure bid price is at least 1

    def calculate_opportunity_cost(self, agent, selling):
        """Estimate opportunity cost based on current state."""
        battery_level = self.get_battery_level(agent)
        if selling:
            return battery_level * 0.2
        else:
            return battery_level * 0.1

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

    def execute_transaction(self, seller_agent, buyer_agent, bid):
        # Placeholder for transaction logic
        print(f"{buyer_agent} wins the auction with a bid of {bid}.")
