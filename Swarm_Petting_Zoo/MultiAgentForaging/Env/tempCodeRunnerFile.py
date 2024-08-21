def step(self, action):
        agent = self.agent_selection  # Get the current agent

        # Initialize reward, termination, and truncation
        reward = 0
        terminated = False
        truncation = False
        observation = self.observe(agent)
        info = self._get_info(agent)

        # Handle cases where the agent's battery is depleted, agent is terminated
        if self._battery_level[agent] == 0 or self.terminations[agent]:
            self.terminations[agent] = True  # Mark the agent as terminated
            
            # Check if all agents are terminated
            if all(self.terminations.values()):
                terminated = True
                reward = -100  # Adjust this based on your reward scheme

                return observation, reward, terminated, truncation, info
            
            # selects the next agent.
            self.agent_selection = self._agent_selector.next()
            return observation, reward, terminated, truncation, info # If terminated, end the step here without doing anything else

        # Process the action: Determine the new direction and location
        direction = self._action_to_direction[action]
        new_location = self._agent_locations[agent] + direction

        # Validate the new location and possibly avoid collisions
        if not self._is_location_valid(agent, new_location):
            new_location = self._simple_avoidance(agent, direction)
        
        self._agent_locations[agent] = np.clip(new_location, 0, self.size - 1)

        # Handle resource collection if the agent is on a resource location
        for i in range(len(self._resources_location)):
            if np.array_equal(self._agent_locations[agent], self._resources_location[i]) and not self._carrying[agent]:
                self._carrying[agent] = True
                self._resources_location = np.delete(self._resources_location, i, axis=0)
                break

        # Check if the agent has returned to the base with a resource
        if np.array_equal(self._agent_locations[agent], self._home_base_location) and self._carrying[agent]:
            reward = 1
            self._carrying[agent] = False

        # If no resources remain, terminate the environment
        if len(self._resources_location) == 0 and not any(self._carrying.values()):
            terminated = True

        observation = self.observe(agent)
        info = self._get_info(agent)
        
        # selects the next agent.
        self.agent_selection = self._agent_selector.next()

        if all(self.terminations.values()):
            terminated = True
            
        if self.render_mode == "human":
            self._render()

        return observation, reward, terminated, truncation, info
