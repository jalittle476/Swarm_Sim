from dataclasses import dataclass

@dataclass
class ForagingConfig:
    
    #Environment Configuration
    seed: int = 255
    num_agents: int = 10
    size: int = 20
    num_resources: int = size**2 // 4
    fov: int = 2
    show_fov: bool = True
    show_gridlines: bool = False
    draw_numbers: bool = True
    record_sim: bool = False
    consider_dead_agents_as_obstacles: bool = False
    render_mode: str = "human"
    debug: bool = False
    full_battery_charge: int = 4 * size  # They could explore the perimeter of the space
    agent_to_visualize: str = "agent_0"
    distribution_type: str = "clustered"
    num_clusters: int = 5
    
    #Agent initial configuration
    
    initial_money: int = 100
    resource_reward: int = 50
    battery_usage_rate: int = 1
    battery_charge_cost: int = 10
    battery_charge_amount: int = 50 
    min_battery_level: int = size
    battery_recharge_threshold: float = 0.5
    
    
    #For Agent State Behavior
    std_dev_base_return: float = 0.05
    std_dev_foraging: float = 0.05
    
    #For Levy walk
    search_pattern: str = "levy_walk"
    beta: float = 1.5