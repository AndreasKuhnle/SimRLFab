from production.envs.time_calc import *
from production.envs.heuristics import *
from production.envs.resources import *
from production.envs.reward_functions import *
import simpy
import numpy as np

class Transport(Resource):
    all_transp_orders = []  # Overall list of available transport orders
    agents_waiting_for_action = []

    def __init__(self, env, id, resp_area, agent_type, statistics, parameters, resources, agents, time_calc, location, label):
        Resource.__init__(self, statistics, parameters, resources, agents, time_calc, location)
        print("Transportation %s created" % id)
        self.env = env
        self.id = id
        self.label = label
        self.resp_area = resp_area
        self.type = "transp"
        self.idle = env.event()
        self.current_location = self.time_calc.randomStreams["transp_agent"][self.id].choice(
            self.resources["sources"])
        self.transp_log = [["action", "sim_time", "from_at", "to_at", "duration"]]
        self.current_order = None
        self.time_start_idle = 0.0
        self.last_transport_time = 0.0
        self.last_transport_start = 0.0
        self.last_handling_time = 0.0
        self.last_handling_start = 0.0
        self.env.process(self.transporting())  # Processed started on creation of resource
        self.agent_type = agent_type
        self.mapping = None
        if self.agent_type == "FIFO":
            self.agent = Decision_Heuristic_Transp_FIFO(env=self.env, statistics=statistics, parameters=parameters,
                                                        resources=resources, agents=agents, agents_resource=self)
        elif self.agent_type == "NJF":
            self.agent = Decision_Heuristic_Transp_NJF(env=self.env, statistics=statistics, parameters=parameters,
                                                        resources=resources, agents=agents, agents_resource=self)
        elif self.agent_type == "EMPTY":
            self.agent = Decision_Heuristic_Transp_EMPTY(env=self.env, statistics=statistics, parameters=parameters,
                                                        resources=resources, agents=agents, agents_resource=self)
        if self.parameters['TRANSP_AGENT_ACTION_MAPPING'] == 'direct':
            self.mapping = []
            for mach in [x for x in self.resources['machines'] if x.id in [0,1,2,3,4]]:
                self.mapping.append([self.resources['sources'][0], mach])
            for mach in [x for x in self.resources['machines'] if x.id in [1,2,3,4]]:
                self.mapping.append([self.resources['sources'][1], mach])
            for mach in [x for x in self.resources['machines'] if x.id in [5,6,7]]:
                self.mapping.append([self.resources['sources'][2], mach])
            for mach in [x for x in self.resources['machines'] if x.id in [0,1]]:
                self.mapping.append([mach, self.resources['sinks'][0]])
            for mach in [x for x in self.resources['machines'] if x.id in [2,3,4]]:
                self.mapping.append([mach, self.resources['sinks'][1]])
            for mach in [x for x in self.resources['machines'] if x.id in [5,6,7]]:
                self.mapping.append([mach, self.resources['sinks'][2]])
            print("Action mapping: ", [[x[0].id, x[1].id] for x in self.mapping])
            if self.parameters['TRANSP_AGENT_EMPTY_ACTION']:
                self.mapping.extend([[-1, x] for x in self.resources['all_resources']])
                print("Empty action: [-1, x]")
            if self.parameters['TRANSP_AGENT_WAITING_ACTION']:
                self.mapping.append([-1, -1])
                print("Waiting action: [-1, -1]")
        elif self.parameters['TRANSP_AGENT_ACTION_MAPPING'] == 'resource':
            self.mapping = []
            for res in self.resources['all_resources']:
                self.mapping.append(res)
            print("Action mapping: ", [x.id for x in self.mapping])
            if self.parameters['TRANSP_AGENT_WAITING_ACTION']:
                self.mapping.append(-1)
                print("Waiting action: -1")
        self.counter = 0
        self.sum_reward = 0.0
        self.state_before = self.calculate_state()
        self.next_action = None
        self.latest_reward = 0.0
        self.invalid_counter = 0
        self.next_action_valid = True
        self.next_action_order = None
        self.next_action_origin = None
        self.next_action_destination = None
        self.last_action_id = None
        self.last_reward_calc = 0.0
        self.last_reward_calc_time = 0.0
        self.counter_action_subsets = [0, 0, 0]  # valid, entry, exit
        self.next_destination_problem_order = None

    def in_resp_area(self, order):
        return self.resp_area[order.current_location.id][order.get_next_step().id]

    @classmethod
    def put(cls, order, trans_agents):
        if order not in Transport.all_transp_orders:
            Transport.all_transp_orders.append(order)
            for transp_agent in trans_agents:
                if transp_agent.in_resp_area(order):
                    if transp_agent.idle.triggered:
                        idle_time = transp_agent.env.now - transp_agent.time_start_idle
                        transp_agent.transp_log.append(
                            ["idle", round(transp_agent.time_start_idle, 5), transp_agent.current_location.id,
                             transp_agent.current_location.id, round(idle_time, 5)])
                        order.statistics['stat_transp_idle'][transp_agent.id] += idle_time
                        transp_agent.time_start_idle = 0.0
                        transp_agent.idle = order.env.event()  # Reset event
                        order.env.process(transp_agent.transporting())  # Restart idle process
                        break

    def get_inventory(self):
        inv = 0
        if self.current_order != None:
            inv = 1
        return inv

    def get_order_destination(self, order, origin, destination):
        result_order, result_origin, result_destination = None, None, None
        if destination.type == "machine" and order.get_next_step().type == "machine":
            if order.current_location == origin and \
                    order.get_next_step().machine_group == destination.machine_group and destination.is_free():
                result_order = order
                result_order.reserved = True
                result_origin = order.current_location
                result_destination = destination
                # Adjust new destination machine and sink resource in process steps list of the order
                if order.get_next_step() != result_destination:
                    order.prod_steps[order.actual_step] = result_destination
                    if order.prod_steps[-2].id < 2:
                        order.prod_steps[-1] = self.resources['sinks'][0]
                    elif order.prod_steps[-2].id < 5:
                        order.prod_steps[-1] = self.resources['sinks'][1]
                    elif order.prod_steps[-2].id < 8:
                        order.prod_steps[-1] = self.resources['sinks'][2]
        else:  # Destination is sink -> always free
            if order.current_location == origin and \
                    order.get_next_step() == destination:
                result_order = order
                result_order.reserved = True
                result_origin = order.current_location
                result_destination = destination
        return result_order, result_origin, result_destination

    def get_next_action(self):
        self.counter += 1

        # Transport when order waiting time threshold reached
        for order in [x for x in Transport.all_transp_orders if x.get_total_waiting_time() > self.parameters['WAITING_TIME_THRESHOLD'] and not x.reserved]:
            if order.get_next_step().type == 'machine':
                for destination in [x for x in self.resources['machines'] if x.machine_group == order.get_next_step().machine_group]:
                    result_order, result_origin, result_destination = self.get_order_destination(order=order, origin=order.current_location, destination=destination)
            else:
                result_order, result_origin, result_destination = self.get_order_destination(order=order, origin=order.current_location, destination=order.get_next_step())
            if result_order != None and result_destination != None:
                result_order = self.next_action_order = Transport.all_transp_orders.pop(Transport.all_transp_orders.index(order))
                result_valid = self.next_action_valid = True
                self.next_action_destination = result_destination
                self.next_action_order = result_order
                self.next_action_origin = result_origin
                print(self.counter, " Order waiting time threshold reached for Order_ID: ", result_order.id)
                self.statistics['stat_transp_threshold_waiting_reached'][self.id] += 1
                return result_order, result_destination

        # Succeed step-event of environment -> continue execute() procedure in "production_env"
        self.parameters['step_criteria'].succeed()
        self.parameters['step_criteria'] = self.env.event()

        # Wait until action is calculated in "production_env"
        Transport.agents_waiting_for_action.append(self)
        yield self.parameters['continue_criteria']
        self.last_action_id = self.next_action[0]

        # If agent type is a heuristic, then use heuristic decision agents
        if self.agent_type != "TRPO":
            result_order, result_destination = self.agent.act(Transport.all_transp_orders)
            result_origin = self.next_action_origin = result_order.current_location
            result_destination = self.next_action_destination = result_destination
            result_valid = self.next_action_valid = True
            if result_destination.id >= self.parameters['NUM_MACHINES']:
                self.next_action[0] = result_destination.id - self.parameters['NUM_SOURCES']
            else:
                self.next_action[0] = result_destination.id
            return result_order, result_destination

        result_order = None
        result_origin = None
        result_destination = None
        self.latest_reward = 0.0

        # Translate RL-agent action output into dispatching action
        # Action-Mapping Coding: [<ORDER-ORIGIN>, <DESTINATION>]
        if self.parameters['TRANSP_AGENT_ACTION_MAPPING'] == 'direct':
            action_origin = self.mapping[self.next_action[0]][0]
            action_destination = self.mapping[self.next_action[0]][1]
        elif self.parameters['TRANSP_AGENT_ACTION_MAPPING'] == 'resource':
            action_origin = self.mapping[self.next_action[0]]
            action_destination = self.mapping[self.next_action[1]]
        if self.parameters['PRINT_CONSOLE']: print("Action ID: ", self.next_action[0], "\t Origin ID: ", action_origin.id,
                                                   "\t Destination ID: ", action_destination.id)
        if action_origin == -1 and action_destination == -1:  # Waiting action
            result_order = result_origin = result_destination = -1
            result_valid = True
        elif action_origin == -1 and action_destination != -1:  # Empty action
            result_order = result_origin = -1
            result_destination = action_destination
            result_valid = True
        else:  # Dispatching action
            result_valid = False
            for order in [x for x in Transport.all_transp_orders if not x.reserved]:
                result_order, result_origin, result_destination = self.get_order_destination(order=order,
                                                                                             origin=action_origin,
                                                                                             destination=action_destination)
                if result_order != None and result_destination != None:
                    result_order = Transport.all_transp_orders.pop(Transport.all_transp_orders.index(order))
                    result_valid = True
                    break

        self.next_action = None

        if result_order == None:  # Only for invalid actions
            # print("invalid action")
            self.next_action_destination = None
            self.next_action_order = None
            self.next_action_origin = None
            self.next_action_valid = False
        else:
            self.next_action_destination = result_destination
            self.next_action_order = result_order
            self.next_action_origin = result_origin
            self.next_action_valid = result_valid

        return result_order, result_destination

    def calculate_state(self):
        result_state = []
        state_type = 'bool'

        # Valid action information always part of state space
        if self.parameters['TRANSP_AGENT_ACTION_MAPPING'] == 'direct':
            state = [False] * len(self.mapping)
            for loc in range(len(self.mapping)):
                orig = self.mapping[loc][0]
                dest = self.mapping[loc][1]
                if orig == -1:
                    state[loc] = True
                    continue
                for order in Transport.all_transp_orders:
                    if dest.type == "machine" and order.get_next_step().type == "machine":
                        if not order.reserved and order.current_location == orig and \
                                order.get_next_step().machine_group == dest.machine_group and \
                                dest.is_free():
                            state[loc] = True
                            break
                    else:
                        if not order.reserved and order.current_location == orig and \
                                order.get_next_step() == dest:
                            state[loc] = True
                            break
        elif self.parameters['TRANSP_AGENT_ACTION_MAPPING'] == 'resource':
            state = [False] * ((len(self.mapping) - 1) ** 2 + 1)
            loc = -1
            for orig in self.mapping:
                for dest in self.mapping:
                    if orig == -1 and dest == -1:
                        state[loc] = True
                        loc += 1
                    elif orig == -1 or dest == -1:
                        continue
                    for order in Transport.all_transp_orders:
                        if dest.type == "machine" and order.get_next_step().type == "machine":
                            if not order.reserved and order.current_location == orig and \
                                    order.get_next_step().machine_group == dest.machine_group and \
                                    dest.is_free():
                                state[loc] = True
                                break
                        else:
                            if not order.reserved and order.current_location == orig and \
                                    order.get_next_step() == dest:
                                state[loc] = True
                                break
        result_state.extend(state)

        if 'bin_buffer_fill' in self.parameters['TRANSP_AGENT_STATE']:
            state_type = 'bool'
            state = [False] * (self.parameters['NUM_MACHINES'] + self.parameters['NUM_SOURCES'])
            for order in Transport.all_transp_orders:
                state[order.current_location.id] = True
            result_state.extend(state)

        if 'bin_location' in self.parameters['TRANSP_AGENT_STATE']:
            state_type = 'bool'
            state = [False] * (self.parameters['NUM_MACHINES'] + self.parameters['NUM_SOURCES'] + self.parameters['NUM_SINKS'])
            state[self.current_location.id] = True
            result_state.extend(state)

        if 'bin_machine_failure' in self.parameters['TRANSP_AGENT_STATE']:
            state_type = 'bool'
            state = [False] * self.parameters['NUM_MACHINES']
            for mach in self.resources['machines']:
                state[mach.id] = mach.broken
            result_state.extend(state)

        if 'int_buffer_fill' in self.parameters['TRANSP_AGENT_STATE']:
            state_type = 'int'
            state = [0] * (self.parameters['NUM_MACHINES'] + self.parameters['NUM_SOURCES'])
            for order in Transport.all_transp_orders:
                state[order.current_location.id] += 1
            result_state.extend(state)

        if 'rel_buffer_fill' in self.parameters['TRANSP_AGENT_STATE']:
            state_type = 'float'
            state = [0.0] * (self.parameters['NUM_MACHINES'] + self.parameters['NUM_SOURCES'])
            for order in Transport.all_transp_orders:
                state[order.current_location.id] += 1.0
            for loc in range(len(state)):
                state[loc] = state[loc] / self.resources['all_resources'][loc].capacity
            result_state.extend(state)

        if 'rel_buffer_fill_in_out' in self.parameters['TRANSP_AGENT_STATE']:
            state_type = 'float'
            state = [0.0] * (self.parameters['NUM_MACHINES'] * 2 + self.parameters['NUM_SOURCES'])
            for res in self.resources['machines']:
                state[res.id * 2] = 1.0 - len(res.buffer_in) / res.capacity
                state[res.id * 2 + 1] = 1.0 - len(res.buffer_out) / res.capacity
            for res in self.resources['sources']:
                state[self.parameters['NUM_MACHINES'] + res.id] = 1.0 - len(res.buffer_out) / res.capacity
            result_state.extend(state)

        if 'order_waiting_time' in self.parameters['TRANSP_AGENT_STATE']:
            state_type = 'float'
            state = [0.0] * (self.parameters['NUM_MACHINES'] + self.parameters['NUM_SOURCES'])
            for order in Transport.all_transp_orders:
                state[order.current_location.id] += order.get_total_waiting_time()
            result_state.extend(state)

        if 'order_waiting_time_normalized' in self.parameters['TRANSP_AGENT_STATE']:
            state_type = 'float'
            state = [-10.0] * (self.parameters['NUM_MACHINES'] + self.parameters['NUM_SOURCES'])  # Default value -10.0
            for mach in self.resources['machines']:
                state[mach.id] = mach.get_normalized_wt_all_machines()
            for source in self.resources['sources']:
                state[source.id] = source.get_normalized_wt_all_sources()
            result_state.extend(state)

        if 'distance_to_action' in self.parameters['TRANSP_AGENT_STATE']:
            state_type = 'float'
            state = [-1.0] * (self.parameters['NUM_MACHINES'] + self.parameters['NUM_SOURCES'])  # Default value -1.0
            for order in Transport.all_transp_orders:
                state[order.current_location.id] = self.parameters['TRANSP_TIME'][self.current_location.id][
                                                       order.current_location.id] / self.parameters['MAX_TRANSP_TIME']
            result_state.extend(state)

        if 'remaining_process_time' in self.parameters['TRANSP_AGENT_STATE']:
            state_type = 'float'
            state = [0.0] * self.parameters['NUM_MACHINES']
            for mach in self.resources['machines']:
                if mach.buffer_processing != None:
                    state[mach.id] = mach.last_process_time - (self.env.now - mach.last_process_start)
            for loc in range(len(state)):
                if self.statistics['stat_machines_processed_orders'][loc] > 0.0:
                    state[loc] = state[loc] / (self.statistics['stat_machines_working'][loc] / self.statistics['stat_machines_processed_orders'][loc])
            result_state.extend(state)

        if 'total_process_time' in self.parameters['TRANSP_AGENT_STATE']:
            state_type = 'float'
            state = [0.0] * self.parameters['NUM_MACHINES']
            for mach in self.resources['machines']:
                if mach.buffer_processing != None:
                    state[mach.id] = mach.last_process_time - (self.env.now - mach.last_process_start)
                for order in mach.buffer_in:
                    state[mach.id] += min(self.parameters['MAX_PROCESS_TIME'][mach.id], max(self.parameters['MIN_PROCESS_TIME'][mach.id], self.time_calc.randomStreams["process_time"][mach.id].exponential(scale=self.parameters['AVERAGE_PROCESS_TIME'][mach.id])))
            for loc in range(len(state)):
                if self.statistics['stat_machines_processed_orders'][loc] > 0.0:
                    state[loc] = state[loc] / (self.resources['machines'][loc].capacity * (self.statistics['stat_machines_working'][loc] / self.statistics['stat_machines_processed_orders'][loc]))
            result_state.extend(state)

        if state_type == 'int':
            result_state = [int(x) for x in result_state]
        elif state_type == 'float':
            result_state = [float(x) for x in result_state]
        return result_state

    def calculate_reward(self, action):
        result_reward = self.parameters['TRANSP_AGENT_REWARD_INVALID_ACTION']
        result_terminal = False
        if self.invalid_counter < self.parameters['TRANSP_AGENT_MAX_INVALID_ACTIONS']:  # If true, then invalid action selected
            if self.parameters['TRANSP_AGENT_REWARD'] == "valid_action":
                result_reward = get_reward_valid_action(self, result_reward)
            elif self.parameters['TRANSP_AGENT_REWARD'] == "utilization":
                result_reward = get_reward_utilization(self, result_reward)
            elif self.parameters['TRANSP_AGENT_REWARD'] == "waiting_time_normalized":
                result_reward = get_reward_waiting_time_normalized(self, result_reward)
            elif self.parameters['TRANSP_AGENT_REWARD'] == "const_weighted":
                result_reward = get_reward_const_weighted(self, result_reward)
            elif self.parameters['TRANSP_AGENT_REWARD'] == "transport_time":
                result_reward = get_reward_transport_time(self, result_reward)
            elif self.parameters['TRANSP_AGENT_REWARD'] == "throughput":
                result_reward = get_reward_throughput(self, result_reward)
            elif self.parameters['TRANSP_AGENT_REWARD'] == "weighted_objectives":
                result_reward = get_reward_weighted_objectives(self, result_reward)
            elif self.parameters['TRANSP_AGENT_REWARD'] == "conwip":
                result_reward = get_reward_conwip()
        else:
            self.invalid_counter = 0
            result_reward = 0.0
            #result_terminal = True

        if self.next_action_valid:
            self.invalid_counter = 0
            self.counter_action_subsets[0] += 1
            if self.next_action_destination != -1 and self.next_action_origin != -1 and self.next_action_destination.type == 'machine':
                self.counter_action_subsets[1] += 1
            elif self.next_action_destination != -1 and self.next_action_origin != -1 and self.next_action_destination.type == 'sink':
                self.counter_action_subsets[2] += 1
        # If explicit episode limits are set in configuration
        if self.parameters['TRANSP_AGENT_REWARD_EPISODE_LIMIT'] > 0:
            result_reward = 0.0
            if (self.parameters['TRANSP_AGENT_REWARD_EPISODE_LIMIT_TYPE'] == 'valid' and self.counter_action_subsets[0] == self.parameters['TRANSP_AGENT_REWARD_EPISODE_LIMIT']) or \
                (self.parameters['TRANSP_AGENT_REWARD_EPISODE_LIMIT_TYPE'] == 'entry' and self.counter_action_subsets[1] == self.parameters['TRANSP_AGENT_REWARD_EPISODE_LIMIT']) or \
                (self.parameters['TRANSP_AGENT_REWARD_EPISODE_LIMIT_TYPE'] == 'exit' and self.counter_action_subsets[2] == self.parameters['TRANSP_AGENT_REWARD_EPISODE_LIMIT']) or \
                (self.parameters['TRANSP_AGENT_REWARD_EPISODE_LIMIT_TYPE'] == 'time' and self.env.now - self.last_reward_calc_time > self.parameters['TRANSP_AGENT_REWARD_EPISODE_LIMIT']):
                result_terminal = True
                self.last_reward_calc_time = self.env.now
                self.invalid_counter = 0
                self.counter_action_subsets = [0, 0, 0]
            if result_terminal:
                if self.parameters['TRANSP_AGENT_REWARD_SPARSE'] == "utilization":
                    result_reward = get_reward_sparse_utilization(self)
                elif self.parameters['TRANSP_AGENT_REWARD_SPARSE'] == "waiting_time":
                    result_reward = get_reward_sparse_waiting_time(self)
                elif self.parameters['TRANSP_AGENT_REWARD_SPARSE'] == "valid_action":
                    result_reward = get_reward_sparse_valid_action(self)
        else:
            self.last_reward_calc_time = self.env.now
        self.latest_reward = result_reward
        return result_reward, result_terminal

    def transport_available(self):
        if len(Transport.all_transp_orders) == 0:
            return False
        counter_not_free_source = 0
        for order in Transport.all_transp_orders:
            if self.in_resp_area(order) and not order.reserved:
                if order.get_next_step().is_free_machine_group():
                    return True
                else:
                    if order.current_location.type == "source":
                        counter_not_free_source += 1
        if counter_not_free_source == len(Transport.all_transp_orders):
            return False

        return False

    def transporting(self):
        while True:
            order, destination = None, None
            if not self.transport_available():
                if self.parameters['PRINT_CONSOLE']: print("Transportation is now idle")
                self.time_start_idle = self.env.now
                self.idle.succeed()
                break

            if self.parameters['PRINT_CONSOLE']:
                print("############# State report at %s #############" % round(self.env.now, 5))
                print("Machine state")
                print(
                    "Legende:  MachineID :  Inbound-Buffer Orders  -  Idle? True/False // Inside Machine Order  -  Outbound-Buffer Orders  -  Broken? True/False")
                for machine in self.resources['machines']:
                    print("Machine", machine.id, ": ", len(machine.buffer_in), [x.id for x in machine.buffer_in], " - ",
                          machine.idle.triggered,
                          [machine.buffer_processing.id if machine.buffer_processing != None else " "], " - ",
                          len(machine.buffer_out), [x.id for x in machine.buffer_out], " - ", machine.broken)

            order, destination = yield self.env.process(self.get_next_action())

            if self.parameters['PRINT_CONSOLE']:
                print("Order state")
                for ord in Transport.all_transp_orders:
                    print("Order", ord.id, ":  ", ord.current_location.id, " -> ", ord.get_next_step().id,
                          "   || Production Steps: ",
                          [i.id for i in ord.prod_steps], " - Current Step:", ord.actual_step, " ",
                          "***" if order != None and order != -1 and order.id == ord.id else "")  # Order that is selected is highlighted by ***

            if order == -1 and destination == -1:  # If waiting action
                if self.parameters['PRINT_CONSOLE']: print("Waiting action selected. Idle for ",
                                                           self.parameters['TRANSP_AGENT_WAITING_TIME_ACTION'],
                                                           " time units.")
                self.transp_log.append(
                    ["waiting_action", round(self.env.now, 5), self.current_location.id, self.current_location.id,
                     self.parameters['TRANSP_AGENT_WAITING_TIME_ACTION']])
                self.statistics['stat_transp_idle'][self.id] += self.parameters['TRANSP_AGENT_WAITING_TIME_ACTION']
                yield self.env.timeout(self.parameters['TRANSP_AGENT_WAITING_TIME_ACTION'])

            elif order == -1 and destination != -1 and destination != None:  # If empty action
                if self.parameters['PRINT_CONSOLE']: print("Empty move action selected. Move to ", destination.id)
                self.transp_log.append(["empty_action", round(self.env.now, 5), self.current_location.id, destination.id, 0.0])
                move_time = self.parameters['TRANSP_TIME'][self.current_location.id][destination.id]
                self.transp_log.append(["move_empty", round(self.env.now, 5), self.current_location.id, destination.id,
                                        round(move_time, 5)])
                yield self.env.timeout(move_time)
                self.current_location = destination

            if order == None:  # If invalid action
                self.invalid_counter += 1
                self.next_action_valid = False
                if self.invalid_counter >= self.parameters['TRANSP_AGENT_MAX_INVALID_ACTIONS']:
                    self.transp_log.append(
                        ["invalid_action_limit_forced_waiting", round(self.env.now, 5), self.current_location.id,
                         self.current_location.id, self.parameters['TRANSP_AGENT_WAITING_TIME_ACTION']])
                    self.statistics['stat_transp_idle'][self.id] += self.parameters['TRANSP_AGENT_WAITING_TIME_ACTION']
                    self.statistics['stat_transp_forced_idle'][self.id] += 1
                    if self.parameters['PRINT_CONSOLE']: print(self.counter, "invalid_action_limit_forced_waiting")
                    yield self.env.timeout(self.parameters['TRANSP_AGENT_WAITING_TIME_ACTION'])
                else:
                    yield self.env.timeout(self.parameters['EPSILON'])

            if order != None and order != -1:
                if self.parameters['PRINT_CONSOLE']: print(
                    "Transport starting from LocationID %s: OrderID %s from LocationID %s to LocationID %s" % (
                    self.current_location.id, order.id, order.current_location.id, destination.id))

                # Move to current location of the selected order
                transp_time = self.time_calc.transp_time(start=self.current_location, end=order.current_location,
                                                         transp=self, statistics=self.statistics,
                                                         parameters=self.parameters)
                self.transp_log.append(
                    ["move_to_empty", round(self.env.now, 5), self.current_location.id, order.current_location.id,
                     round(transp_time, 5)])
                yield self.env.timeout(transp_time)
                self.current_location = order.current_location

                # Handling order pick-up
                time_start_handling = self.env.now
                self.current_order = order.current_location.get_buffer_out(order)
                order.order_log.append(["picked_up", order.id, round(self.env.now, 5), self.id])
                handling_time = 0.0
                if order.current_location.type == "source":
                    handling_time = self.time_calc.handling_time(MachineOrSource="source", LoadOrUnload="unload",
                                                                 transp=self, statistics=self.statistics,
                                                                 parameters=self.parameters)
                elif order.current_location.type == "machine":
                    handling_time = self.time_calc.handling_time(MachineOrSource="machine", LoadOrUnload="unload",
                                                                 transp=self, statistics=self.statistics,
                                                                 parameters=self.parameters)
                self.transp_log.append(
                    ["pick_up", round(self.env.now, 5), self.current_location.id, self.current_location.id,
                     round(handling_time, 5)])
                self.last_handling_time = handling_time
                self.last_handling_start = self.env.now
                yield self.env.timeout(handling_time)

                # Transportation
                transp_time = self.time_calc.transp_time(start=order.current_location, end=destination, transp=self,
                                                         statistics=self.statistics, parameters=self.parameters)
                self.transp_log.append(["transport", round(self.env.now, 5), order.current_location.id, destination.id,
                                        round(transp_time, 5)])
                self.last_transport_time = transp_time
                self.last_transport_start = self.env.now
                yield self.env.timeout(transp_time)
                self.current_location = destination
                order.order_log.append(["arrived", order.id, round(self.env.now, 5), self.id])

                # Handling order put down
                handling_time = 0.0
                if order.current_location.type == "sink":
                    handling_time = self.time_calc.handling_time(MachineOrSource="source", LoadOrUnload="load",
                                                                 transp=self, statistics=self.statistics,
                                                                 parameters=self.parameters)
                elif order.current_location.type == "machine":
                    handling_time = self.time_calc.handling_time(MachineOrSource="machine", LoadOrUnload="load",
                                                                 transp=self, statistics=self.statistics,
                                                                 parameters=self.parameters)
                yield self.env.timeout(handling_time)
                order.time_handling += self.env.now - time_start_handling
                self.transp_log.append(
                    ["put_down", round(self.env.now, 5), self.current_location.id, self.current_location.id,
                     round(handling_time, 5)])
                self.statistics['stat_order_handling'][order.id] += order.time_handling
                order.order_log.append(["put_down", order.id, round(self.env.now, 5), self.id])

                destination.put_buffer_in(order)
                order.current_location = destination
                self.current_order = None

                order.transported.succeed()
                order.transported = self.env.event()
                order.reserved = False
