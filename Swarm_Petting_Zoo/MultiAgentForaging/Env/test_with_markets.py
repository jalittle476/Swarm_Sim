
from foraging_world_with_markets import ForagingEnvironmentWithMarkets
from foraging_config import ForagingConfig
import time
from data_analysis import generate_agent_activity_heatmap, generate_exchange_heatmap_for_agent, overlay_exchanges_on_heatmap
import pandas as pd

def test_markets(step_limit=5):
    # Initialize the environment with the auction subclass
    env = ForagingEnvironmentWithMarkets(config=ForagingConfig())
    env.reset(seed=42)

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last(agent)

        # if termination or truncation:
        #     action = None
        # else:
            # Decide and execute the action
        action = env.decide_action(agent)

            # # For testing: Trigger an auction process with a random seller agent
            # if action is not None and agent == 'agent_0':  # Trigger auction for testing
            #     env.initiate_auction(agent)
            
        env.step(action)
        
        # Observe and log agent state
        obs = env.observe(agent)
        env.log_agent_state(agent, obs, env.agent_states[agent])

        # Check if all agents are terminated
        if all(env.terminations.values()):
            print("All agents terminated. Ending the simulation.")
            break
        
        
        env.render()
        #time.sleep(1)
    env.save_logs("simulation_logs.csv")
    env.generate_summary_table(filename="agent_summary.xlsx", file_format="excel")
    env.close()



if __name__ == "__main__":
    test_markets()
