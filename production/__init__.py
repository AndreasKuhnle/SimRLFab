from gym.envs.registration import register

register(
    id='production-v0',
    entry_point='production.envs:ProductionEnv',
)