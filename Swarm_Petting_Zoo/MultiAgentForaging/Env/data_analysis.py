# data_analysis.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def generate_agent_activity_heatmap(log_data, normalize=True, filename=None, agent_id=None, grid_size=None):
    """
    Generate and plot a heatmap of agent activity from simulation logs.

    Parameters:
    - log_data (pd.DataFrame): The simulation data containing 'x' and 'y' columns for agent locations.
    - normalize (bool): If True, normalize the heatmap using a logarithmic scale.
    - filename (str): If provided, saves the heatmap as an image file.
    - agent_id (str): If provided, filters the log data for the specified agent.
    - grid_size (int): The size of the grid. If None, it will be determined from the data.
    """
    # Filter for a specific agent if agent_id is provided
    if agent_id:
        log_data = log_data[log_data['agent_id'] == agent_id]

    # Determine grid size if not provided
    if grid_size is None:
        grid_size = max(log_data['x'].max(), log_data['y'].max()) + 1

    # Create a blank grid to store activity counts
    activity_heatmap = np.zeros((grid_size, grid_size))

    # Aggregate activity for the specified agent
    for _, row in log_data.iterrows():
        activity_heatmap[int(row['x']), int(row['y'])] += 1

    # Normalize activity for better contrast
    if normalize:
        activity_heatmap = np.log1p(activity_heatmap)  # log1p to handle zeros

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(activity_heatmap, cmap="viridis", origin="lower")
    plt.colorbar(label="Log of Visit Frequency" if normalize else "Visit Frequency")
    plt.title(f"Heatmap of Agent Activity ({'All Agents' if not agent_id else agent_id})")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.tight_layout()

    # Save the plot if a filename is provided
    if filename:
        plt.savefig(filename, dpi=300)
        print(f"Heatmap saved as {filename}")
    
    # Show the plot
    plt.show()

def generate_exchange_heatmap(log_data, grid_size=None, normalize=True, filename=None):
    """
    Generate and plot a heatmap of exchange density from simulation logs.

    Parameters:
    - log_data (pd.DataFrame): The simulation data containing 'x', 'y', and 'exchange_count' columns.
    - grid_size (int): The size of the grid. If None, it will be determined from the data.
    - normalize (bool): If True, normalize the heatmap using a logarithmic scale.
    - filename (str): If provided, saves the heatmap as an image file.
    """
    # Ensure only actual exchanges are included
    exchanges = log_data[log_data['exchange_count'] > 0]  # Filter rows with exchanges

    if 'state' in log_data.columns:  # Optional refinement
        exchanges = exchanges[exchanges['state'] == "Exchanging"]

    # Determine grid size if not provided
    if grid_size is None:
        grid_size = max(log_data['x'].max(), log_data['y'].max()) + 1

    # Create a blank grid to store exchange counts
    exchange_heatmap = np.zeros((grid_size, grid_size))

    # Aggregate exchanges into the heatmap
    for _, row in exchanges.iterrows():
        exchange_heatmap[int(row['x']), int(row['y'])] += row['exchange_count']

    # Normalize the heatmap
    if normalize:
        exchange_heatmap = np.log1p(exchange_heatmap)

    # Plot the exchange heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(exchange_heatmap, cmap="plasma", origin="lower")
    plt.colorbar(label="Log of Exchange Frequency" if normalize else "Exchange Frequency")
    plt.title("Heatmap of Exchange Density")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.tight_layout()

    # Save the plot if a filename is provided
    if filename:
        plt.savefig(filename, dpi=300)
        print(f"Exchange heatmap saved as {filename}")
    
    # Show the plot
    plt.show()


def overlay_exchanges_on_heatmap(agent_data, exchange_data, agent_id=None, filename=None):
    """
    Overlay exchange locations on an agent's activity heatmap.

    Parameters:
    - log_data (pd.DataFrame): The simulation data containing 'x', 'y', and 'exchange_count' columns.
    - agent_id (str): If provided, filters the log data for the specified agent.
    - filename (str): If provided, saves the overlay as an image file.
    """
    # Filter for the specified agent
    if agent_id:
        agent_data = agent_data[agent_data['agent_id'] == agent_id]

    # Create activity heatmap
    grid_size = max(agent_data['x'].max(), agent_data['y'].max()) + 1
    activity_heatmap = np.zeros((grid_size, grid_size))
    for _, row in agent_data.iterrows():
        activity_heatmap[int(row['y']), int(row['x'])] += 1  # Note the (y, x) order for heatmap

    # Normalize activity heatmap
    activity_heatmap = np.log1p(activity_heatmap)

    # Use exchange data directly
    exchange_points = exchange_data[['x', 'y']].dropna()


    # Plot the heatmap with exchange overlay
    plt.figure(figsize=(10, 8))
    plt.imshow(activity_heatmap, cmap="viridis", origin="lower", alpha=0.7)  # Use "lower" for Cartesian alignment
    plt.scatter(exchange_points['x'], exchange_points['y'], c="red", s=20, label="Exchanges")
    plt.colorbar(label="Log of Visit Frequency")
    plt.title(f"Activity Heatmap with Exchange Overlay ({'All Agents' if not agent_id else agent_id})")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.tight_layout()

    # Save the plot if a filename is provided
    if filename:
        plt.savefig(filename, dpi=300)
        print(f"Overlay heatmap saved as {filename}")

    # Show the plot
    plt.show()
    
def generate_exchange_heatmap_for_agent(exchange_data, agent_id=None, filename=None, grid_size=None):
    """
    Generate a heatmap of only the exchange locations for a specific agent or all agents.

    Parameters:
    - exchange_data (pd.DataFrame): DataFrame containing exchange logs with 'x', 'y', and 'agent_id'.
    - agent_id (str): If provided, filters exchange data for the specified agent.
    - filename (str): If provided, saves the heatmap as an image file.
    """
    # Filter for the specified agent
    if agent_id:
        exchange_data = exchange_data[exchange_data['agent_id'] == agent_id]


    # Unpack grid size
    heatmap_rows, heatmap_cols = (grid_size , grid_size)
    exchange_heatmap = np.zeros((heatmap_rows, heatmap_cols))
    # Create exchange heatmap
    for _, row in exchange_data.iterrows():
        exchange_heatmap[int(row['y']), int(row['x'])] += 1  # Note (y, x) indexing for heatmaps

    # Normalize the heatmap
    exchange_heatmap = np.log1p(exchange_heatmap)  # Log scale for better visualization

    # Plot the exchange heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(exchange_heatmap, cmap="viridis", origin="lower", alpha=0.9)  # Use "lower" for Cartesian alignment
    plt.colorbar(label="Log of Exchange Frequency")
    plt.title(f"Exchange Heatmap ({'All Agents' if not agent_id else agent_id})")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.tight_layout()

    # Save the plot if a filename is provided
    if filename:
        plt.savefig(filename, dpi=300)
        print(f"Exchange heatmap saved as {filename}")

    # Show the plot
    plt.show()



