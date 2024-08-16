from foraging_world_with_auctions_v2 import ForagingEnvironmentWithAuction
import time

def test_subclass_features(step_limit=10):
    # Initialize the environment with the auction subclass
    env = ForagingEnvironmentWithAuction(num_agents=2, size=20, num_resources=5, fov=2, render_mode="human")
    env.reset(seed=42)

    step_count = 0

    while step_count < step_limit:
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last(observe=False)

            # Print the step count header before any actions are taken
            print(f"\n=== Step {step_count + 1} ===")  # Enhanced step count header

            if termination or truncation:
                env.step(None)
                continue
            
            #Observe and Log agent state
            obs = env.observe(agent)
            #env.log_agent_state(agent, obs)
            
            # Decide and execute the action
            action = env.decide_action(agent)
            env.step(action)
            

            # Check agent state and handle battery depletion or other state changes
            if env.check_agent_state(agent, obs):
                env.step(None)
                continue


            env.render()
            time.sleep(1)

            step_count += 1
            if step_count >= step_limit or all(env.terminations.values()):
                break

        # If the outer loop conditions are met, ensure the loop terminates
        if step_count >= step_limit or all(env.terminations.values()):
            break

    env.close()

if __name__ == "__main__":
    test_subclass_features()
