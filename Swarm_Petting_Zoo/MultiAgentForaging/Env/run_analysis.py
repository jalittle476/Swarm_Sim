import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from foraging_config import ForagingConfig 
# Import or define your heatmap functions
from data_analysis import (
    generate_agent_activity_heatmap,
    generate_exchange_heatmap_for_agent,
    overlay_exchanges_on_heatmap,
)

def run_analysis(log_filepath):
    """
    Run analysis on pre-existing simulation logs to generate various heatmaps.

    Parameters:
    - log_filepath (str): Path to the simulation log file (CSV).
    """
    # Load the data
    simulation_logs = pd.read_csv(log_filepath)
    print("Simulation logs loaded successfully.")

    # Separate rows by type
    agents = simulation_logs[simulation_logs['row_type'] == "agent"]
    exchanges = simulation_logs[simulation_logs['row_type'] == "exchange"]

    # Parse agent locations
    agents['x'], agents['y'] = zip(*agents['location']
        .str.strip("[]")
        .str.split()
        .map(lambda xy: (int(xy[0]), int(xy[1])) if len(xy) == 2 else (None, None))
    )
    
    # Ensure exchange coordinates are integers
    exchanges['x'] = exchanges['x'].astype(float).astype(int)
    exchanges['y'] = exchanges['y'].astype(float).astype(int)
    
    # Define the environment grid size
    grid_size = (ForagingConfig.size)

    # Generate agent activity heatmap
    print("Generating agent activity heatmap...")
    #generate_agent_activity_heatmap(agents, filename="agent_activity_heatmap.png", grid_size=grid_size)

    # Generate global exchange heatmap
    print("Generating global exchange heatmap...")
    generate_exchange_heatmap_for_agent(exchange_data=exchanges, filename="global_exchange_heatmap.png",grid_size=grid_size)

    # Generate heatmaps for each agent's exchanges
    print("Generating exchange heatmaps for each agent...")
    for agent in exchanges['agent_id'].dropna().unique():  # Filter out rows with missing 'agent_id'
        filename = f"{agent}_exchange_heatmap.png"
        generate_exchange_heatmap_for_agent(exchange_data=exchanges, agent_id=agent, filename=filename, grid_size=grid_size)

    print("Analysis complete!")

if __name__ == "__main__":
    # Path to the simulation log file
    log_file = "simulation_logs.csv"  # Update this to the actual path of your log file
    
    # Run the analysis
    run_analysis(log_file)
