# simulation_runner.py
from foraging_world_with_transactions import ForagingEnvironmentWithTransactions
from foraging_config import ForagingConfig

def run_simulation(config, gui=None):
    env = ForagingEnvironmentWithTransactions(config=config)
    env.reset(seed=42)

    for agent in env.agent_iter():
        # Check if the simulation should be paused or stopped
        if gui and gui.simulation_paused:
            while gui.simulation_paused:
                if not gui.simulation_running:  # Allow exiting while paused
                    env.close()
                    return
                gui.process_events()  # Ensure the GUI remains responsive during pause
            continue

        if gui and not gui.simulation_running:
            break

        observation, reward, termination, truncation, info = env.last(agent)

        if termination or truncation:
            action = None
        else:
            action = env.decide_action(agent)
            
        env.step(action)
        obs = env.observe(agent)
        env.log_agent_state(agent, obs, env.agent_states[agent])

        if all(env.terminations.values()):
            print("All agents terminated. Ending the simulation.")
            break

        env.render()

    env.close()
