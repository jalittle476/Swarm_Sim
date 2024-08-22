from foraging_world_with_transactions import ForagingEnvironmentWithTransactions
import time

def test_subclass_features(step_limit=5):
    # Initialize the environment with the auction subclass
    env = ForagingEnvironmentWithTransactions(num_agents=20, size=20, num_resources=200, fov=2, render_mode="human", debug=False)
    env.reset(seed=42)

    for agent in env.agent_iter():

        observation, reward, termination, truncation, info = env.last(agent)

        if termination or truncation:
            action = None
        else:
            # Decide and execute the action
            #print(f"Agent {agent} is deciding on an action...")
            action = env.decide_action(agent)
            #action = env.action_space.sample()
            
        env.step(action)
     
        
        #Observe and Log agent state
        obs = env.observe(agent)
        env.log_agent_state(agent, obs, env.agent_states[agent])


         # Check if all agents are terminated
        if all(env.terminations.values()):
            print("All agents terminated. Ending the simulation.")
            break

        env.render()
        #time.sleep(1)

    env.close()

if __name__ == "__main__":
    test_subclass_features()
