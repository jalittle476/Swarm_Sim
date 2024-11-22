def generate_exchange_heatmap_for_agent(exchange_data, agent_id=None, filename=None, grid_size=(50, 50)):
    """
    Generate a heatmap of only the exchange locations for a specific agent or all agents.

    Parameters:
    - exchange_data (pd.DataFrame): DataFrame containing exchange logs with 'x', 'y', and 'agent_id'.
    - agent_id (str): If provided, filters exchange data for the specified agent.
    - filename (str): If provided, saves the heatmap as an image file.
    - grid_size (tuple): The size of the environment as (rows, cols).
    """
    # Filter for the specified agent
    if agent_id:
        exchange_data = exchange_data[exchange_data['agent_id'] == agent_id]

    # Create exchange heatmap with fixed grid size
    heatmap_rows, heatmap_cols = grid_size
    exchange_heatmap = np.zeros((heatmap_rows, heatmap_cols))
    for _, row in exchange_data.iterrows():
        exchange_heatmap[int(row['y']), int(row['x'])] += 1  # Note (y, x) indexing for heatmaps

    # Normalize the heatmap
    exchange_heatmap = np.log1p(exchange_heatmap)  # Log scale for better visualization

    # Plot the exchange heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(exchange_heatmap, cmap="viridis", origin="lower", extent=[0, heatmap_cols, 0, heatmap_rows], alpha=0.9)
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
