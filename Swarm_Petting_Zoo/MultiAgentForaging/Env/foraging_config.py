from dataclasses import dataclass

@dataclass
class ForagingConfig:
    
    #Environment Configuration
    seed: int = 255
    num_agents: int = 1
    size: int = 20
    num_resources: int = 200
    fov: int = 1
    show_fov: bool = True
    show_gridlines: bool = False
    draw_numbers: bool = False
    record_sim: bool = False
    consider_dead_agents_as_obstacles: bool = False
    render_mode: str = "human"
    debug: bool = False
    
    battery_usage_rate: int = 1
    battery_charge_cost: int = 10
    battery_charge_amount: int = 10
    min_battery_level: int = size
    
    std_dev_base_return: float = 0.8
    std_dev_foraging: float = 0.5