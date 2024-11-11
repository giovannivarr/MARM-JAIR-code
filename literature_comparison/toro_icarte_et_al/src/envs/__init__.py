from gym.envs.registration import register

# ----------------------------------------- OFFICE
register(
    id='Office-single-Toro-Icarte',
    entry_point='envs.grids.grid_environment:OfficeRMToroIcarteEnv',
    max_episode_steps=1000
)

register(
    id='Office-single-Strategy',
    entry_point='envs.grids.grid_environment:OfficeRMStrategyEnv',
    max_episode_steps=1000
)
