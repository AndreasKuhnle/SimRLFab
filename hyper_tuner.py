import argparse
import os
import pickle

import ConfigSpace as cs
from hpbandster.core.nameserver import NameServer, nic_name_to_host
from hpbandster.core.result import json_result_logger, logged_results_to_HBS_result
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB
import numpy as np

from tensorforce.environments import Environment
from tensorforce.execution import Runner

from production.envs.production_env import *
from production.envs import ProductionEnv

#######################################
#### RUN COMMAND by:  & python hyper_tuner.py 'production.envs.ProductionEnv'
#######################################

TIMESTEPS = 10**2
NUM_EPISODES = 10**2
ITERATIONS = 10**1

class TensorforceWorker(Worker):

    def __init__(self, *args, environment=None, **kwargs):
        # def __init__(self, run_id, nameserver=None, nameserver_port=None, logger=None, host=None, id=None, timeout=None):
        super().__init__(*args, **kwargs)
        assert environment != None
        self.environment = environment

    def compute(self, config_id, config, budget, working_directory):
        if self.environment.max_episode_timesteps() == None:
            min_capacity = 1000 + config['batch_size']
        else:
            min_capacity = self.environment.max_episode_timesteps() + config['batch_size']
        max_capacity = 100000
        capacity = min(max_capacity, max(min_capacity, config['memory'] * config['batch_size']))
        frequency = max(4, int(config['frequency'] * config['batch_size']))

        if config['baseline'] == 'no':
            baseline_policy = None
            baseline_objective = None
            baseline_optimizer = None
            estimate_horizon = False
            estimate_terminal = False
            estimate_advantage = False
        else:
            estimate_horizon = 'late'
            estimate_advantage = (config['estimate_advantage'] == 'yes')
            if config['baseline'] == 'same-policy':
                baseline_policy = None
                baseline_objective = None
                baseline_optimizer = None
            elif config['baseline'] == 'auto':
                # other modes, shared network/policy etc !!!
                baseline_policy = dict(network=dict(type='auto', internal_rnn=False))
                baseline_objective = dict(
                    type='value', value='state', huber_loss=0.0, early_reduce=False
                )
                baseline_optimizer = dict(
                    type='adam', learning_rate=config['baseline_learning_rate']
                )
            else:
                assert False

        if config['l2_regularization'] < 3e-5:  # yes/no better
            l2_regularization = 0.0
        else:
            l2_regularization = config['l2_regularization']

        if config['entropy_regularization'] < 3e-5:  # yes/no better
            entropy_regularization = 0.0
        else:
            entropy_regularization = config['entropy_regularization']

        # Set agent configuration according to configspace
        print("### Set agent configuration according to configspace")
        agent = dict(
            agent='tensorforce',
            policy=dict(network=dict(type='auto', internal_rnn=False)),
            memory=dict(type='replay', capacity=capacity),  # replay, recent
            update=dict(unit='timesteps', batch_size=config['batch_size'], frequency=frequency),  # timesteps, episode
            optimizer=dict(type='adam', learning_rate=config['learning_rate']),
            objective=dict(
                type='policy_gradient', ratio_based=True, clipping_value=0.1, 
                early_reduce=False
            ),
            reward_estimation=dict(
                horizon=config['horizon'], discount=config['discount'],
                estimate_horizon=estimate_horizon, estimate_actions=False,
                estimate_terminal=False, estimate_advantage=estimate_advantage
            ),
            baseline_policy=baseline_policy, baseline_objective=baseline_objective,
            baseline_optimizer=baseline_optimizer,
            preprocessing=None,
            l2_regularization=l2_regularization, entropy_regularization=entropy_regularization
        )

        # Set state representation according to configspace
        print("### Set state representation according to configspace")

        # Example state configurations to evaluate
        config_state = None
        if config['state'] == 0:
            config_state = []
        elif config['state'] == 1:
            config_state = ['bin_buffer_fill']
        elif config['state'] == 2:
            config_state = ['bin_buffer_fill', 'distance_to_action']
        elif config['state'] == 3:
            config_state = ['bin_buffer_fill', 'distance_to_action', 'bin_machine_failure']
        elif config['state'] == 4:
            config_state = ['bin_buffer_fill', 'distance_to_action', 'bin_machine_failure', 'order_waiting_time']


        self.environment.environment.parameters.update({'TRANSP_AGENT_STATE': config_state})
        self.environment.environment.parameters.update({'TRANSP_AGENT_REWARD': config['reward']})
        #self.environment.environment.parameters.update({'TRANSP_AGENT_REWARD_INVALID_ACTION': config['reward_invalid']})
        #self.environment.environment.parameters.update({'TRANSP_AGENT_REWARD_OBJECTIVE_WEIGHTS': config['reward_weighted']})
        self.environment.environment.parameters.update({'TRANSP_AGENT_MAX_INVALID_ACTIONS': config['max_invalid_actions']})
        self.environment.environment.parameters.update({'TRANSP_AGENT_WAITING_TIME_ACTION': config['waiting_if_invalid_actions']})

        # num_episodes = list()
        final_reward = list()
        max_reward = list()
        rewards = list()

        for n in range(round(budget)):
            runner = Runner(agent=agent, environment=self.environment)
            #runner = Runner(agent='config/ppo2.json', environment=self.environment)

            # performance_threshold = runner.environment.max_episode_timesteps() - agent['reward_estimation']['horizon']

            # def callback(r, p):
            #     return True

            runner.run(num_episodes=NUM_EPISODES, use_tqdm=False)
            runner.close()

            # num_episodes.append(len(runner.episode_rewards))
            final_reward.append(float(np.mean(runner.episode_rewards[-20:], axis=0)))
            average_rewards = [
                float(np.mean(runner.episode_rewards[n: n + 20], axis=0))
                for n in range(len(runner.episode_rewards) - 20)
            ]
            max_reward.append(float(np.amax(average_rewards, axis=0)))
            rewards.append(list(runner.episode_rewards))

        # mean_num_episodes = float(np.mean(num_episodes, axis=0))
        mean_final_reward = float(np.mean(final_reward, axis=0))
        mean_max_reward = float(np.mean(max_reward, axis=0))
        # loss = mean_num_episodes - mean_final_reward - mean_max_reward
        loss = -mean_final_reward - mean_max_reward

        return dict(loss=loss, info=dict(rewards=rewards))

    @staticmethod
    def get_configspace():
        """
        It builds the configuration space with the needed hyperparameters.
        It is easily possible to implement different types of hyperparameters.
        Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
        :return: ConfigurationsSpace-Object
        """
        configspace = cs.ConfigurationSpace()

        memory = cs.hyperparameters.UniformIntegerHyperparameter(name='memory', lower=2, upper=50)
        configspace.add_hyperparameter(hyperparameter=memory)

        batch_size = cs.hyperparameters.UniformIntegerHyperparameter(
            name='batch_size', lower=32, upper=8192, log=True
        )
        configspace.add_hyperparameter(hyperparameter=batch_size)

        frequency = cs.hyperparameters.UniformFloatHyperparameter(
            name='frequency', lower=3e-2, upper=1.0, log=True
        )
        configspace.add_hyperparameter(hyperparameter=frequency)

        learning_rate = cs.hyperparameters.UniformFloatHyperparameter(
            name='learning_rate', lower=1e-5, upper=3e-2, log=True
        )
        configspace.add_hyperparameter(hyperparameter=learning_rate)

        horizon = cs.hyperparameters.UniformIntegerHyperparameter(
            name='horizon', lower=1, upper=50
        )
        configspace.add_hyperparameter(hyperparameter=horizon)

        discount = cs.hyperparameters.UniformFloatHyperparameter(
            name='discount', lower=0.8, upper=1.0, log=True
        )
        configspace.add_hyperparameter(hyperparameter=discount)

        baseline = cs.hyperparameters.CategoricalHyperparameter(
            name='baseline', choices=('no', 'auto', 'same-policy')
        )
        configspace.add_hyperparameter(hyperparameter=baseline)

        baseline_learning_rate = cs.hyperparameters.UniformFloatHyperparameter(
            name='baseline_learning_rate', lower=1e-5, upper=3e-2, log=True
        )
        configspace.add_hyperparameter(hyperparameter=baseline_learning_rate)

        estimate_advantage = cs.hyperparameters.CategoricalHyperparameter(
            name='estimate_advantage', choices=('no', 'yes')
        )
        configspace.add_hyperparameter(hyperparameter=estimate_advantage)

        l2_regularization = cs.hyperparameters.UniformFloatHyperparameter(
            name='l2_regularization', lower=1e-5, upper=1.0, log=True
        )
        configspace.add_hyperparameter(hyperparameter=l2_regularization)

        entropy_regularization = cs.hyperparameters.UniformFloatHyperparameter(
            name='entropy_regularization', lower=1e-5, upper=1.0, log=True
        )
        configspace.add_hyperparameter(hyperparameter=entropy_regularization)

        state = cs.hyperparameters.CategoricalHyperparameter(
            name='state', choices=(0, 1, 2, 3, 4)
        )
        configspace.add_hyperparameter(hyperparameter=state)

        reward = cs.hyperparameters.CategoricalHyperparameter(
            name='reward', choices=('valid_action', 'utilization', 'waiting_time_normalized')
        )
        configspace.add_hyperparameter(hyperparameter=reward)

        """
        reward_invalid = cs.hyperparameters.UniformFloatHyperparameter(
            name='reward_invalid', lower=-1.0, upper=0.0
        )
        configspace.add_hyperparameter(hyperparameter=reward_invalid)
        """
        """
        reward_weighted = cs.hyperparameters.CategoricalHyperparameter(
            name='reward_weighted', choices=({'utilization': 1.0, 'waiting_time': 1.0})  # {'inventory_balance': 1.0, 'transport_time': 1.0}
        ) # Alternatives: 'utilization': 1.0, 'waiting_time': 1.0, 'transport_time': 1.0, 'inventory_balance': 1.0
        configspace.add_hyperparameter(hyperparameter=reward_weighted)

        configspace.add_condition(
            condition=cs.EqualsCondition(
                child=reward_weighted, parent=reward, value='weighted_objectives'
            )
        )
        """
        max_invalid_actions = cs.hyperparameters.UniformIntegerHyperparameter(
            name='max_invalid_actions', lower=1, upper=10
        )
        configspace.add_hyperparameter(hyperparameter=max_invalid_actions)

        waiting_if_invalid_actions = cs.hyperparameters.UniformIntegerHyperparameter(
            name='waiting_if_invalid_actions', lower=1, upper=10
        )
        configspace.add_hyperparameter(hyperparameter=waiting_if_invalid_actions)

        configspace.add_condition(
            condition=cs.NotEqualsCondition(
                child=baseline_learning_rate, parent=baseline, value='no'
            )
        )

        configspace.add_condition(
            condition=cs.NotEqualsCondition(
                child=estimate_advantage, parent=baseline, value='no'
            )
        )

        return configspace


def main():
    parser = argparse.ArgumentParser(description='Tensorforce hyperparameter tuner')
    parser.add_argument(
        'environment', help='Environment (name, configuration JSON file, or library module)'
    )
    parser.add_argument(
        '-l', '--level', type=str, default=None,
        help='Level or game id, like `CartPole-v1`, if supported'
    )
    parser.add_argument(
        '-m', '--max-repeats', type=int, default=1, help='Maximum number of repetitions'
    )
    parser.add_argument(
        '-n', '--num-iterations', type=int, default=ITERATIONS, help='Number of BOHB iterations'
    )
    parser.add_argument(
        '-d', '--directory', type=str, default='tuner', help='Output directory'
    )
    parser.add_argument(
        '-r', '--restore', type=str, default=None, help='Restore from given directory'
    )
    parser.add_argument('--id', type=str, default='worker', help='Unique worker id')
    args = parser.parse_args()

    print(args.environment)

    if args.level == None:
        environment = Environment.create(environment=args.environment, max_episode_timesteps=TIMESTEPS)  # , max_episode_timesteps=timesteps
    else:
        environment = Environment.create(environment=args.environment, level=args.level)

    if False:
        host = nic_name_to_host(nic_name=None)
        port = 123
    else:
        host = 'localhost'
        port = None

    server = NameServer(run_id=args.id, working_directory=args.directory, host=host, port=port)
    nameserver, nameserver_port = server.start()

    worker = TensorforceWorker(
        environment=environment, run_id=args.id, nameserver=nameserver,
        nameserver_port=nameserver_port, host=host
    )
    # TensorforceWorker(run_id, nameserver=None, nameserver_port=None, logger=None, host=None, id=None, timeout=None)
    # logger: logging.logger instance, logger used for debugging output
    # id: anything with a __str__method, if multiple workers are started in the same process, you MUST provide a unique id for each one of them using the `id` argument.
    # timeout: int or float, specifies the timeout a worker will wait for a new after finishing a computation before shutting down. Towards the end of a long run with multiple workers, this helps to shutdown idling workers. We recommend a timeout that is roughly half the time it would take for the second largest budget to finish. The default (None) means that the worker will wait indefinitely and never shutdown on its own.

    worker.run(background=True)

    # config = cs.sample_configuration().get_dictionary()
    # print(config)
    # res = worker.compute(config=config, budget=1, working_directory='.')
    # print(res)

    if args.restore == None:
        previous_result = None
    else:
        previous_result = logged_results_to_HBS_result(directory=args.restore)

    result_logger = json_result_logger(directory=args.directory, overwrite=True)  # ???

    optimizer = BOHB(
        configspace=worker.get_configspace(), min_budget=0.5, max_budget=float(args.max_repeats),
        run_id=args.id, working_directory=args.directory,
        nameserver=nameserver, nameserver_port=nameserver_port, host=host,
        result_logger=result_logger, previous_result=previous_result
    )
    # BOHB(configspace=None, eta=3, min_budget=0.01, max_budget=1, min_points_in_model=None, top_n_percent=15, num_samples=64, random_fraction=1 / 3, bandwidth_factor=3, min_bandwidth=1e-3, **kwargs)
    # Master(run_id, config_generator, working_directory='.', ping_interval=60, nameserver='127.0.0.1', nameserver_port=None, host=None, shutdown_workers=True, job_queue_sizes=(-1,0), dynamic_queue_size=True, logger=None, result_logger=None, previous_result = None)
    # logger: logging.logger like object, the logger to output some (more or less meaningful) information

    results = optimizer.run(n_iterations=args.num_iterations)
    # optimizer.run(n_iterations=1, min_n_workers=1, iteration_kwargs={})
    # min_n_workers: int, minimum number of workers before starting the run

    optimizer.shutdown(shutdown_workers=True)
    server.shutdown()
    environment.close()

    with open(os.path.join(args.directory, 'results.pkl'), 'wb') as filehandle:
        pickle.dump(results, filehandle)

    print('Best found configuration:', results.get_id2config_mapping()[results.get_incumbent_id()]['config'])
    print('Runs:', results.get_runs_by_id(config_id=results.get_incumbent_id()))
    print('A total of {} unique configurations where sampled.'.format(len(results.get_id2config_mapping())))
    print('A total of {} runs where executed.'.format(len(results.get_all_runs())))
    print('Total budget corresponds to {:.1f} full function evaluations.'.format(
        sum([r.budget for r in results.get_all_runs()]) / args.max_repeats)
    )


if __name__ == '__main__':
    main()