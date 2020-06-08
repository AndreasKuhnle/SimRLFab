import sys
import progressbar
from tensorforce.environments import Environment
from logger import *
import numpy as np
from production.envs.initialize_env import *
from production.envs.resources import *
from production.envs.time_calc import Time_calc
from datetime import datetime

class ProductionEnv(Environment):
    """
    Python-Packages: numpy, progressbar2, simpy, pandas

    MINUTES as basic time Unit
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.count_episode = 0
        self.last_export_time = 0.0
        self.last_export_real_time = datetime.now()

        # Simpy Environment + Tensorforce-Agent
        self.env = simpy.Environment()
        self.counter = 0
        self.agents = dict() 

        self.parameters = define_production_parameters(env=self.env, episode=self.count_episode)
        self.time_calc = Time_calc(parameters=self.parameters, episode=self.count_episode)
        self.statistics, self.stat_episode = define_production_statistics(parameters=self.parameters)
        self.resources = define_production_resources(env=self.env, statistics=self.statistics, parameters=self.parameters, agents=self.agents, time_calc=self.time_calc)

        self.statistics['sim_start_time'] = datetime.now()

    def execute(self, actions):
        reward = None
        terminal = False
        states = None
        self.counter += 1

        # print(self.counter, "Agent-Action: ", int(actions))

        if (self.counter % self.parameters['EXPORT_FREQUENCY'] == 0 or self.counter % self.max_episode_timesteps() == 0) and not self.parameters['EXPORT_NO_LOGS']:
            self.export_statistics(self.counter, self.count_episode)

        if self.counter == self.max_episode_timesteps():
            print("Last episode action ", datetime.now())
            terminal = True

        # If multiple transport agents then for loop required
        for agent in Transport.agents_waiting_for_action:
            agent = Transport.agents_waiting_for_action.pop(0)

            if self.parameters['TRANSP_AGENT_ACTION_MAPPING'] == 'direct':
                agent.next_action = [int(actions)]
            elif self.parameters['TRANSP_AGENT_ACTION_MAPPING'] == 'resource':
                agent.next_action = [int(actions[0]), int(actions[1])]
            agent.state_before = None

            self.parameters['continue_criteria'].succeed()
            self.parameters['continue_criteria'] = self.env.event()

            self.env.run(until=self.parameters['step_criteria'])  # Waiting until action is processed in simulation environment
            # Simulation is now in state after action processing

            reward, terminal = agent.calculate_reward(actions)

            if terminal:
                print("Last episode action ", datetime.now())
                self.export_statistics(self.counter, self.count_episode)

            agent = Transport.agents_waiting_for_action[0]
            states = agent.calculate_state()  # Calculate state for next action determination

            if self.parameters['TRANSP_AGENT_ACTION_MAPPING'] == 'direct':
                self.statistics['stat_agent_reward'][-1][3] = [int(actions)]
            elif self.parameters['TRANSP_AGENT_ACTION_MAPPING'] == 'resource':
                self.statistics['stat_agent_reward'][-1][3] = [int(actions[0]), int(actions[1])]
            self.statistics['stat_agent_reward'][-1][4] = round(reward, 5)
            self.statistics['stat_agent_reward'][-1][5] = agent.next_action_valid
            self.statistics['stat_agent_reward'].append([self.count_episode, self.counter, round(self.env.now, 5), None, None, None, states])

            return states, terminal, reward

    def reset(self):
        print("####### Reset Environment #######")

        self.count_episode += 1
        self.counter = 0    

        if self.count_episode == self.parameters['CHANGE_SCENARIO_AFTER_EPISODES']:
            self.change_production_parameters()

        print("Sim start time: ", self.statistics['sim_start_time'])

        # Setup and start simulation
        if self.env.now == 0.0:
            print('Run machine shop simpy environment')
            self.env.run(until=self.parameters['step_criteria'])

        states = self.resources['transps'][0].calculate_state()

        return states

    def close(self):
        print("####### Close Environment #######")
        if not self.parameters['EXPORT_NO_LOGS']:
            self.statistics.update({'time_end' : self.env.now})
            export_statistics_logging(statistics=self.statistics, parameters=self.parameters, resources=self.resources)
        super().close()

    def render(self, mode='human', close=False):
        print("####### Render Environment #######")
        pass

    def states(self):
        state_type = 'bool'
        number = 0
        # Avaliable Action are always part of state vector
        if self.parameters['TRANSP_AGENT_ACTION_MAPPING'] == 'direct':
            number += len(self.resources['transps'][0].mapping)
        elif self.parameters['TRANSP_AGENT_ACTION_MAPPING'] == 'resource':
            number += (len(self.resources['transps'][0].mapping) - 1) ** 2 + 1
        # State value alternatives sorted according to the type
        if 'bin_buffer_fill' in self.parameters['TRANSP_AGENT_STATE']:
            number += self.parameters['NUM_MACHINES'] + self.parameters['NUM_SOURCES']
        if 'bin_location' in self.parameters['TRANSP_AGENT_STATE']:
            number += self.parameters['NUM_MACHINES'] + self.parameters['NUM_SOURCES'] + self.parameters['NUM_SINKS']
        if 'bin_machine_failure' in self.parameters['TRANSP_AGENT_STATE']:
            number += self.parameters['NUM_MACHINES']
        if 'int_buffer_fill' in self.parameters['TRANSP_AGENT_STATE']:
            state_type = 'int'
            number += self.parameters['NUM_MACHINES'] + self.parameters['NUM_SOURCES']
        if 'rel_buffer_fill' in self.parameters['TRANSP_AGENT_STATE']:
            state_type = 'float'
            number += self.parameters['NUM_MACHINES'] + self.parameters['NUM_SOURCES']
        if 'rel_buffer_fill_in_out' in self.parameters['TRANSP_AGENT_STATE']:
            state_type = 'float'
            number += self.parameters['NUM_MACHINES'] * 2 + self.parameters['NUM_SOURCES']
        if 'order_waiting_time' in self.parameters['TRANSP_AGENT_STATE']:
            state_type = 'float'
            number += self.parameters['NUM_MACHINES'] + self.parameters['NUM_SOURCES'] 
        if 'order_waiting_time_normalized' in self.parameters['TRANSP_AGENT_STATE']:
            state_type = 'float'
            number += self.parameters['NUM_MACHINES'] + self.parameters['NUM_SOURCES']
        if 'distance_to_action' in self.parameters['TRANSP_AGENT_STATE']:
            state_type = 'float'
            number += self.parameters['NUM_MACHINES'] + self.parameters['NUM_SOURCES'] 
        if 'remaining_process_time' in self.parameters['TRANSP_AGENT_STATE']:
            state_type = 'float'
            number += self.parameters['NUM_MACHINES']
        if 'total_process_time' in self.parameters['TRANSP_AGENT_STATE']:
            state_type = 'float'
            number += self.parameters['NUM_MACHINES']

        print("State space size: ", number)
        return dict(type=state_type, shape=(number))

    def actions(self):
        if self.parameters['TRANSP_AGENT_ACTION_MAPPING'] == 'direct':
            number = len(self.resources['transps'][0].mapping)
        elif self.parameters['TRANSP_AGENT_ACTION_MAPPING'] == 'resource':
            shape = (2,)
            number = len(self.resources['transps'][0].mapping)
            return dict(type='int', num_values=number, shape=shape)
        print("Action space size: ", number)
        return dict(type='int', num_values=number)

    def change_production_parameters(self):
        print("CHANGE_OF_PRODUCTION_PARAMETERS")
        for mach in self.resources['machines']:
            mach.capacity = mach.capacity * 2
        self.parameters['TRANSP_TIME'] = [[x / 3.0 for x in y] for y in self.parameters['TRANSP_TIME']]
        self.parameters['TRANSP_SPEED'] = self.parameters['TRANSP_SPEED'] * 3.0

    def export_statistics(self, counter, episode_counter):
        # Episodic KPI logger & printout
        episode_length = self.env.now - self.last_export_time
        episode_length_real_time = datetime.now() - self.last_export_real_time
        valid_actions = np.array(self.statistics['stat_agent_reward'])[:-1, 5]
        sum_reward = np.array(self.statistics['stat_agent_reward'])[:-1, 4]
        export_data = [str(episode_counter), str(counter), str(round(self.env.now, 5)), str(round(episode_length, 5)), str(round(episode_length_real_time.total_seconds(), 5)), str(sum(valid_actions)), str(round(sum(sum_reward), 5))]

        # Export all episode statistics
        # Procedure consideres epsiode overlaps in case operations reach over the episode length
        self.stat_episode_diff = dict()
        for stat in self.statistics['episode_statistics']:
            self.stat_episode_diff[stat] = self.statistics[stat] - self.stat_episode[stat]
        for mach in range(self.parameters['NUM_MACHINES']):
            list_of_stats = ['stat_machines_working', 'stat_machines_changeover', 'stat_machines_broken', 'stat_machines_idle']
            for stat in list_of_stats:
                if stat == 'stat_machines_working' and self.resources['machines'][mach].buffer_processing != None:
                    if self.resources['machines'][mach].broken:
                        if self.resources['machines'][mach].last_broken_start > self.last_export_time + self.parameters['EPSILON']:  # Begin broken in mid of episode
                            self.stat_episode_diff[stat][mach] = self.statistics[stat][mach] - self.stat_episode[stat][mach] - self.resources['machines'][mach].last_process_time + (self.resources['machines'][mach].last_broken_start - self.resources['machines'][mach].last_process_start)
                            self.resources['machines'][mach].last_process_time -= (self.resources['machines'][mach].last_broken_start - self.resources['machines'][mach].last_process_start)
                            self.resources['machines'][mach].last_process_start = self.resources['machines'][mach].last_broken_start + self.resources['machines'][mach].last_broken_time
                        elif self.resources['machines'][mach].last_broken_time > episode_length:  # Entire episode broken
                            self.stat_episode_diff[stat][mach] = 0.0
                        else:
                            raise Exception("Unexcpected case!")
                        # Update broken statistic
                        self.stat_episode_diff['stat_machines_broken'][mach] = self.statistics['stat_machines_broken'][mach] - self.stat_episode['stat_machines_broken'][mach] - self.resources['machines'][mach].last_broken_time + (self.env.now - self.resources['machines'][mach].last_broken_start)
                        self.resources['machines'][mach].last_broken_time -= (self.env.now - self.resources['machines'][mach].last_broken_start)
                        self.resources['machines'][mach].last_broken_start = self.env.now
                        self.stat_episode['stat_machines_broken'][mach] = self.stat_episode['stat_machines_broken'][mach] + self.stat_episode_diff['stat_machines_broken'][mach]
                    else:
                        # Process interrupted by breakdown
                        if self.resources['machines'][mach].last_broken_start > self.resources['machines'][mach].last_process_start and self.resources['machines'][mach].last_broken_start + self.resources['machines'][mach].last_broken_time < self.env.now:
                            self.stat_episode_diff[stat][mach] = self.statistics[stat][mach] - self.stat_episode[stat][mach] - self.resources['machines'][mach].last_process_time + (self.env.now - self.resources['machines'][mach].last_process_start) - self.resources['machines'][mach].last_broken_time
                            self.resources['machines'][mach].last_process_time -= (self.env.now - self.resources['machines'][mach].last_process_start) - self.resources['machines'][mach].last_broken_time
                            self.resources['machines'][mach].last_process_start = self.env.now
                        else:
                            self.stat_episode_diff[stat][mach] = self.statistics[stat][mach] - self.stat_episode[stat][mach] - self.resources['machines'][mach].last_process_time + (self.env.now - self.resources['machines'][mach].last_process_start)
                            self.resources['machines'][mach].last_process_time -= (self.env.now - self.resources['machines'][mach].last_process_start)
                            self.resources['machines'][mach].last_process_start = self.env.now
                    self.stat_episode[stat][mach] = self.stat_episode[stat][mach] + self.stat_episode_diff[stat][mach]
                elif stat == 'stat_machines_broken' and self.resources['machines'][mach].broken and self.resources['machines'][mach].buffer_processing == None:
                    self.stat_episode_diff[stat][mach] = self.statistics[stat][mach] - self.stat_episode[stat][mach] - self.resources['machines'][mach].last_broken_time + (self.env.now - self.resources['machines'][mach].last_broken_start)
                    self.resources['machines'][mach].last_broken_time -= (self.env.now - self.resources['machines'][mach].last_broken_start)
                    self.resources['machines'][mach].last_broken_start = self.env.now
                    self.stat_episode[stat][mach] = self.stat_episode[stat][mach] + self.stat_episode_diff[stat][mach]
                elif stat == 'stat_machines_idle' and self.resources['machines'][mach].idle.triggered:
                    if self.resources['machines'][mach].broken:
                        self.stat_episode[stat][mach] = self.statistics[stat][mach]
                    else:
                        if self.stat_episode[stat][mach] <= self.statistics[stat][mach]:
                            self.stat_episode_diff[stat][mach] = self.statistics[stat][mach] - self.stat_episode[stat][mach] + (self.env.now - self.resources['machines'][mach].time_start_idle_stat)  # max(0.0, self.statistics[stat][mach] - self.stat_episode[stat][mach])
                        else:
                            self.stat_episode_diff[stat][mach] = episode_length
                        self.stat_episode[stat][mach] = max(self.stat_episode[stat][mach] + self.stat_episode_diff[stat][mach], self.statistics[stat][mach])
                else:
                    if stat == 'stat_machines_broken' and self.resources['machines'][mach].buffer_processing != None and self.resources['machines'][mach].broken:
                        continue
                    self.stat_episode[stat][mach] = self.statistics[stat][mach]
        for transp in range(self.parameters['NUM_TRANSP_AGENTS']):
            list_of_stats = ['stat_transp_working', 'stat_transp_walking', 'stat_transp_handling', 'stat_transp_idle']
            for stat in list_of_stats:
                if (stat == 'stat_transp_working' or stat == 'stat_transp_walking') and self.resources['transps'][transp].current_order != None:
                    if self.env.now > self.stat_episode[stat][transp].last_handling_time + self.stat_episode[stat][transp].last_handling_start:  # Handling over
                        self.stat_episode_diff[stat][transp] = self.statistics[stat][transp] - self.stat_episode[stat][transp] - self.resources['transps'][transp].last_transport_time + (self.env.now - self.resources['transps'][transp].last_transport_start)
                        self.stat_episode[stat][transp] = self.stat_episode[stat][transp] + self.stat_episode_diff[stat][transp]
                elif stat == 'stat_transp_handling' and self.resources['transps'][transp].current_order != None:
                    if self.env.now < self.stat_episode[stat][transp].last_handling_time + self.stat_episode[stat][transp].last_handling_start:
                        self.stat_episode_diff[stat][transp] = self.statistics[stat][transp] - self.stat_episode[stat][transp] - self.resources['transps'][transp].last_handling_time + (self.env.now - self.resources['transps'][transp].last_handling_start)
                        self.stat_episode[stat][transp] = self.stat_episode[stat][transp] + self.stat_episode_diff[stat][transp]
                elif stat == 'stat_transp_idle' and self.resources['transps'][transp].idle.triggered:
                    if self.stat_episode[stat][transp] <= self.statistics[stat][transp]:
                        self.stat_episode_diff[stat][transp] = self.statistics[stat][transp] - self.stat_episode[stat][transp] + (self.env.now - self.resources['transps'][transp].time_start_idle)
                    else:
                        self.stat_episode_diff[stat][transp] = episode_length
                    self.stat_episode[stat][transp] = self.stat_episode[stat][transp] + self.stat_episode_diff[stat][transp]
                else:
                    self.stat_episode[stat][transp] = self.statistics[stat][transp]

        self.stat_episode['stat_machines_processed_orders'] = self.statistics['stat_machines_processed_orders'].copy()

        # Compute KPI values and add it to export_data list
        for stat in self.statistics['episode_statistics']:
            if stat == 'stat_machines_processed_orders':
                export_data.append(str(round(np.sum(self.stat_episode_diff[stat]), 5)))
            else:
                export_data.append(str(round(np.mean(self.stat_episode_diff[stat] / episode_length), 5)))

        export_data.append(str(round(sum([np.mean(self.stat_episode_diff[stat] / episode_length) for stat in ['stat_machines_working', 'stat_machines_changeover', 'stat_machines_broken', 'stat_machines_idle']]), 5)))
        export_data.append(str(self.statistics['stat_transp_selected_idle'][0]))
        export_data.append(str(self.statistics['stat_transp_forced_idle'][0]))
        export_data.append(str(self.statistics['stat_transp_threshold_waiting_reached'][0]))
        indices = [k for k, v in self.statistics['stat_order_eop'].items() if v > self.last_export_time]
        # Number of finished orders
        export_data.append(str(len(indices)))
        if len(indices) > 0:
            # Avg order waiting time
            waiting_time = np.mean([self.statistics['stat_order_waiting'][id] for id in indices])
            export_data.append(str(round(waiting_time, 5)))
            # Alpha factor
            cycle_time = np.mean([self.statistics['stat_order_leadtime'][id] for id in indices])
            process_time = np.mean([self.statistics['stat_order_processing'][id] for id in indices])
            dynFF = cycle_time / process_time
            util = np.sum(self.stat_episode_diff['stat_machines_working']) / (np.sum(self.stat_episode_diff['stat_machines_working']) + np.sum(self.stat_episode_diff['stat_machines_idle']))
            alpha = max(0.0, (dynFF - 1) * (1 - util) / util)
            export_data.append(str(round(alpha, 5)))
        else:
            export_data.append(str(np.nan))
            export_data.append(str(np.nan))
        # Weighted avg inventory
        self.statistics['stat_inv_episode'] = np.array(self.statistics['stat_inv_episode'])
        export_data.append(str(round(np.average(self.statistics['stat_inv_episode'][:,1], axis=0, weights=self.statistics['stat_inv_episode'][:,0]), 5)))

        # Console printout
        string = ""
        for text in export_data:
            string = string + str(text) + ","
        string = string[:-1]
        titel = self.statistics['episode_log_header']
        os.system('cls' if os.name == 'nt' else "printf '\033c'")
        for index in range(len(export_data)):
            sys.stdout.write("\n " + titel[index] + ": \t" + export_data[index])
        sys.stdout.write("\n")
        sys.stdout.flush()

        # Export in log files
        self.statistics['episode_log'].write("%s\n" % (string))
        self.statistics['episode_log'].flush()
        os.fsync(self.statistics['episode_log'].fileno())
        
        pd.DataFrame(self.statistics['stat_agent_reward'][:-1]).to_csv(self.parameters['PATH_TIME'] + "_agent_reward_log.txt", header=None, index=None, sep=',', mode='a')

        # Reset statistics for episode
        self.last_export_time = self.env.now
        self.last_export_real_time = datetime.now()
        self.statistics['stat_inv_episode'] = [self.statistics['stat_inv_episode'][-1]]
        self.statistics['stat_agent_reward'] = [self.statistics['stat_agent_reward'][-1]]
        self.statistics['stat_transp_selected_idle'] = np.array([0] * self.parameters['NUM_TRANSP_AGENTS'])
        self.statistics['stat_transp_forced_idle'] = np.array([0] * self.parameters['NUM_TRANSP_AGENTS'])
        self.statistics['stat_transp_threshold_waiting_reached'] = np.array([0] * self.parameters['NUM_TRANSP_AGENTS'])
