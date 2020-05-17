from logger import *
from production.envs.production_env import *
from production.envs import ProductionEnv
from tensorforce.environments import Environment
from tensorforce.execution import Runner

# tf.set_random_seed(10)

timesteps = 10 ** 2  # Set time steps per episode
episodes = 10 ** 2  # Set number of episodes

environment_production = Environment.create(environment='production.envs.ProductionEnv',
                                            max_episode_timesteps=timesteps)

# Tensorforce runner
runner = Runner(agent='config/ppo1.json', environment=environment_production)
environment_production.agents = runner.agent

# Run training
runner.run(num_episodes=episodes)

environment_production.environment.statistics.update({'time_end': environment_production.environment.env.now})
export_statistics_logging(statistics=environment_production.environment.statistics,
                          parameters=environment_production.environment.parameters,
                          resources=environment_production.environment.resources)
