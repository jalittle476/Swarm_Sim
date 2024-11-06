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