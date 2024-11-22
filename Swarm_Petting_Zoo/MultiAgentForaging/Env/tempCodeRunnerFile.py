    # Initialize the environment with the auction subclass
    env = ForagingEnvironmentWithMarkets(config=ForagingConfig())
    env.reset(seed=42)
    env_size = env.size