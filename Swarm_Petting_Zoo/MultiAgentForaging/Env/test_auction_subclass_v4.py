from foraging_world_with_auctions_v2 import ForagingEnvironmentWithAuction
import time

def test_subclass_features(step_limit=5):
    # Initialize the environment with the auction subclass
    env = ForagingEnvironmentWithAuction(num_agents=2, size=20, num_resources=5, fov=5, render_mode="human")
    env.reset(seed=42)

    for agent in env.agent_iter(max_iter=step_limit):

        observation, reward, termination, truncation, info = env.last(observe=False)

        # Print the step count header before any actions are taken
        #print(f"\n=== Step {step_count + 1} ===")  # Enhanced step count header

        if termination or truncation:
            action = None
        
        # Decide and execute the action
        # action = env.decide_action(agent)
        action = env.action_space.sample()
        env.step(action)
     
        
        #Observe and Log agent state
        obs = env.observe(agent)
        env.log_agent_state(agent, obs, env.agent_states[agent])

        # Check agent state and handle battery depletion or other state changes
        if env.check_agent_state(agent, obs):
            env.step(None)
            continue


        env.render()
        time.sleep(1)

    env.close()

if __name__ == "__main__":
    test_subclass_features()
