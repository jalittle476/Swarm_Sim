
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
    #env.log_results("results.txt")
    env.generate_summary_table(filename="agent_summary.xlsx", file_format="excel")
    env.close()
    
    # Load data
    simulation_logs = pd.read_csv("simulation_logs.csv")
    
    # Separate rows by type
    agents = simulation_logs[simulation_logs['row_type'] == "agent"]
    exchanges = simulation_logs[simulation_logs['row_type'] == "exchange"]

    # Separate rows by type
    # Parse agent locations
    agents['x'], agents['y'] = zip(*agents['location']
        .str.strip("[]")
        .str.split()
        .map(lambda xy: (int(xy[0]), int(xy[1])) if len(xy) == 2 else (None, None))
    )
    # Ensure exchange coordinates are integers
    exchanges['x'] = exchanges['x'].astype(float).astype(int)
    exchanges['y'] = exchanges['y'].astype(float).astype(int)

    # Generate heatmaps
    generate_agent_activity_heatmap(agents, filename="agent_activity_heatmap.png")
    #generate_exchange_heatmap(exchanges, filename="exchange_density_heatmap.png")
    # Generate heatmap for all exchanges
    generate_exchange_heatmap_for_agent(exchange_data=exchanges, filename="global_exchange_heatmap.png")


    # # Overlay exchanges on individual agent heatmaps
    # for agent in agents['agent_id'].unique():
    #     filename = f"{agent}_overlay_heatmap.png"
    #     overlay_exchanges_on_heatmap(agents, exchanges, agent_id=agent, filename=filename)
    
    # Generate heatmaps for each agent's exchanges
    for agent in exchanges['agent_id'].dropna().unique():  # Filter out rows with missing 'agent_id'
        filename = f"{agent}_exchange_heatmap.png"
        generate_exchange_heatmap_for_agent(exchange_data=exchanges, agent_id=agent, filename=filename)




if __name__ == "__main__":
    test_markets()
