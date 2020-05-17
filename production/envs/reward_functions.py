from production.envs.time_calc import *
from production.envs.heuristics import *
from production.envs.resources import *
import simpy
import numpy as np

def get_reward_valid_action(transport_resource, invalid_reward):
    result_reward = invalid_reward
    if transport_resource.next_action_destination == -1 or transport_resource.next_action_origin == -1:  # Waiting or empty action selected
        result_reward = transport_resource.parameters['TRANSP_AGENT_REWARD_WAITING_ACTION']
    elif transport_resource.next_action_valid:
        result_reward = 1.0
    return result_reward

def get_reward_utilization(transport_resource, invalid_reward):
    result_reward = invalid_reward
    if transport_resource.next_action_destination == -1 or transport_resource.next_action_origin == -1:  # Waiting or empty action selected
        result_reward = transport_resource.parameters['TRANSP_AGENT_REWARD_WAITING_ACTION']
    elif transport_resource.next_action_valid:
        util = 0.0
        for mach in transport_resource.resources['machines']:
            util += mach.get_utilization_step()
        util = util / transport_resource.parameters['NUM_MACHINES']
        transport_resource.last_reward_calc = util
        result_reward = np.exp(util / 1.5) - 1.0
        if transport_resource.next_action_destination.type == 'machine':
            result_reward = transport_resource.parameters['TRANSP_AGENT_REWARD_SUBSET_WEIGHTS'][0] * result_reward
        else:
            result_reward = transport_resource.parameters['TRANSP_AGENT_REWARD_SUBSET_WEIGHTS'][1] * result_reward
    return result_reward

def get_reward_waiting_time_normalized(transport_resource, invalid_reward):
    result_reward = invalid_reward
    if transport_resource.next_action_destination == -1 or transport_resource.next_action_origin == -1:  # Waiting or empty action selected
        result_reward = transport_resource.parameters['TRANSP_AGENT_REWARD_WAITING_ACTION']
    elif transport_resource.next_action_order != None:
        if transport_resource.next_action_origin.type == 'machine':
            result_reward = transport_resource.next_action_origin.get_normalized_wt_all_machines()
            if result_reward == -1:
                result_reward = transport_resource.next_action_origin.machine_wt_normalizer.get_z_score_normalization(transport_resource.next_action_order.get_total_waiting_time())
        else:
            result_reward = transport_resource.next_action_origin.get_normalized_wt_all_sources()
            if result_reward == -1:
                result_reward = transport_resource.next_action_origin.source_wt_normalizer.get_z_score_normalization(transport_resource.next_action_order.get_total_waiting_time())
        result_reward = min(max(np.exp(-0.1 * result_reward) - 0.5, 0.0), 1.0)
        if transport_resource.next_action_destination.type == 'machine':
            result_reward = transport_resource.parameters['TRANSP_AGENT_REWARD_SUBSET_WEIGHTS'][0] * result_reward
        else:
            result_reward = transport_resource.parameters['TRANSP_AGENT_REWARD_SUBSET_WEIGHTS'][1] * result_reward
    return result_reward

def get_reward_const_weighted(transport_resource, invalid_reward):
    result_reward = invalid_reward
    if transport_resource.next_action_destination == -1 or transport_resource.next_action_origin == -1:  # Waiting or empty action selected
        result_reward = transport_resource.parameters['TRANSP_AGENT_REWARD_WAITING_ACTION']
    if transport_resource.next_action_valid:
        if transport_resource.next_action_destination.type == 'machine':
            result_reward = transport_resource.parameters['TRANSP_AGENT_REWARD_SUBSET_WEIGHTS'][0]
        else:
            result_reward = transport_resource.parameters['TRANSP_AGENT_REWARD_SUBSET_WEIGHTS'][1]
    return result_reward

def get_reward_transport_time(transport_resource, invalid_reward):
    result_reward = invalid_reward
    if transport_resource.next_action_destination == -1 or transport_resource.next_action_origin == -1:  # Waiting or empty action selected
        result_reward = transport_resource.parameters['TRANSP_AGENT_REWARD_WAITING_ACTION']
    elif transport_resource.next_action_valid:
        result_reward = 1.0
    return result_reward

def get_reward_throughput(transport_resource, invalid_reward):
    result_reward = invalid_reward
    if transport_resource.next_action_destination == -1 or transport_resource.next_action_origin == -1:  # Waiting or empty action selected
        result_reward = transport_resource.parameters['TRANSP_AGENT_REWARD_WAITING_ACTION']
    elif transport_resource.next_action_valid:
        if transport_resource.next_action_destination.type == 'sink':
            result_reward = 1.0
        else:
            result_reward = 0.0
    return result_reward

def get_reward_weighted_objectives(transport_resource, invalid_reward):
    result_reward = invalid_reward
    dict_rew_func = {'utilization': get_reward_utilization, 'waiting_time': get_reward_waiting_time_normalized}
    if transport_resource.next_action_destination == -1:  # Waiting action selected
        result_reward = transport_resource.parameters['TRANSP_AGENT_REWARD_WAITING_ACTION']
    elif transport_resource.next_action_valid:
        for key, value in transport_resource.parameters['TRANSP_AGENT_REWARD_OBJECTIVE_WEIGHTS'].items():
            result_reward += value * dict_rew_func[key](transport_resource, invalid_reward)
    return result_reward

def get_reward_conwip(transport_resource, invalid_reward):
    result_reward = invalid_reward
    if transport_resource.next_action_valid:
        if transport_resource.next_action_destination != -1 or transport_resource.next_action_origin == -1:  # Waiting or empty action selected
            inv_diff = abs(
                transport_resource.parameters['TRANSP_AGENT_CONWIP_INV'] - transport_resource.statistics['stat_inv_episode'][-1][1])
            result_reward = np.exp(-1.0 * (inv_diff - 1.0)) - 1.0
    return result_reward

def get_reward_sparse_valid_action(transport_resource):
    result_reward = transport_resource.parameters['TRANSP_AGENT_REWARD_INVALID_ACTION']
    if transport_resource.next_action_destination == -1 or transport_resource.next_action_origin == -1:  # Waiting or empty action selected
        result_reward = transport_resource.parameters['TRANSP_AGENT_REWARD_WAITING_ACTION']
    elif transport_resource.next_action_valid:
        result_reward = 1.0
    return result_reward

def get_reward_sparse_utilization(transport_resource):
    result_reward = 0.0
    util = 0.0
    for mach in transport_resource.resources['machines']:
        util += mach.get_utilization_step()
    util = util / transport_resource.parameters['NUM_MACHINES']
    transport_resource.last_reward_calc = util
    result_reward = np.exp(util / 1.5) - 1.0
    return result_reward

def get_reward_sparse_waiting_time(transport_resource):
    result_reward = 0.0
    indices = [k for k, v in transport_resource.statistics['stat_order_eop'].items() if v > transport_resource.last_reward_calc_time]
    if len(indices) > 0:
        result_reward = np.mean([transport_resource.statistics['stat_order_waiting'][id] for id in indices])
        result_reward = min(max(np.exp(-0.1 * result_reward) - 0.5, 0.0), 1.0)
    return result_reward