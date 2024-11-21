from dataclasses import dataclass

@dataclass
class ForagingConfig:
    
    #Environment Configuration
    seed: int = 255
    num_agents: int = 50
    size: int = 50
    num_resources: int = 625
    fov: int = 5
    show_fov: bool = False
    show_gridlines: bool = False
    show_grid_addresses: bool = False
    draw_numbers: bool = False
    record_sim: bool = False
    consider_dead_agents_as_obstacles: bool = False
    render_mode: str = None
    debug: bool = False
    full_battery_charge: int = 4 * size  
    agent_to_visualize: str = "agent_0"
    distribution_type: str = "uniform"
    
    #Agent initial configuration
    
    initial_money: int = 0
    resource_reward: int = full_battery_charge
    battery_usage_rate: int = 1
    battery_charge_cost: int = full_battery_charge
    battery_charge_amount: int = full_battery_charge
    battery_recharge_threshold: float = 0.5
    
    
    #For Agent State Behavior
    std_dev_base_return: float = 0.05
    std_dev_foraging: float = 0.05
    std_dev_move: float = 0.05
    
    #For Levy walk
    search_pattern: str = "levy_walk"
    beta: float = 1.5