from production.envs.time_calc import *

"""
Heuristic Decision Agents
"""

class Decision_Heuristic(object):
    def __init__(self, env, statistics, parameters, resources, agents, agents_resource):
        self.statistics = statistics
        self.parameters = parameters
        self.resources = resources
        self.agents = agents
        self.env = env
        self.agents_resource = agents_resource
        agents.update({'Decision_Heuristic_Transp' : []})
        agents.update({'Decision_Heuristic_Machine' : []})

    def act(self, states):
        raise NotImplementedError

    def get_next_machine_min_buffer_fill(self, order, statistics, parameters, resources):
        """Return next machine for processing with smallest buffer fill if multiple machines are in same machine group. Used for Transp-Heuristics."""
        result_machine = None
        min_buffer_fill = round(float('inf'), 0)
        if order.get_next_step().type == "sink":
            return order.get_next_step(), min_buffer_fill
        for mach in [x for x in resources['machines'] if order.get_next_step().machine_group == x.machine_group]:
            if len(mach.buffer_in) < min_buffer_fill:
                result_machine = mach
                min_buffer_fill = len(mach.buffer_in)
        return result_machine, min_buffer_fill

class Decision_Heuristic_Transp_NJF(Decision_Heuristic):
    """Selects the next transportation order for a transportation agent based on the distance to the order pickup location"""
    def __init__(self, env, statistics, parameters, resources, agents, agents_resource):
        super(self.__class__, self).__init__(env=env, statistics=statistics, parameters=parameters, resources=resources, agents=agents, agents_resource=agents_resource)
        agents['Decision_Heuristic_Transp'].append(self)
        print("NJF_Transp_Decision created")

    def act(self, states):
        if states == None:
            return None, None
        result_order = None
        result_dest = None
        min_distance = float('inf')
        for order in states:
            if order.get_next_step().is_free_machine_group() and not order.reserved:
                distance = self.parameters['TRANSP_TIME'][self.agents_resource.current_location.id][order.current_location.id]
                if distance < min_distance:
                    min_distance = distance
                    result_order = order
                    result_dest, _ = self.get_next_machine_min_buffer_fill(order=order, statistics=self.statistics, parameters=self.parameters, resources=self.resources)
        result_order = states.pop(states.index(result_order))
        result_order.reserved = True
        return result_order, result_dest

class Decision_Heuristic_Transp_EMPTY(Decision_Heuristic):
    """Selects the next transportation order for a transportation agent based on the total order waiting time"""
    def __init__(self, env, statistics, parameters, resources, agents, agents_resource):
        super(self.__class__, self).__init__(env=env, statistics=statistics, parameters=parameters, resources=resources, agents=agents, agents_resource=agents_resource)
        agents['Decision_Heuristic_Transp'].append(self)
        print("EMPTY_Transp_Decision created")

    def act(self, states):
        if states == None:
            return None, None
        result_order = None
        result_dest = None
        min_fill_level = round(float('inf'), 0)
        for order in states:
            if order.get_next_step().is_free_machine_group() and not order.reserved:
                dest, fill_level = self.get_next_machine_min_buffer_fill(order=order, statistics=self.statistics, parameters=self.parameters, resources=self.resources)
                if fill_level < min_fill_level:
                    min_fill_level = fill_level
                    result_order = order
                    result_dest = dest
        result_order = states.pop(states.index(result_order))
        result_order.reserved = True
        return result_order, result_dest

class Decision_Heuristic_Transp_FIFO(Decision_Heuristic):
    """Selects the next transportation order for a transportation agent based on the total order waiting time"""
    def __init__(self, env, statistics, parameters, resources, agents, agents_resource):
        super(self.__class__, self).__init__(env=env, statistics=statistics, parameters=parameters, resources=resources, agents=agents, agents_resource=agents_resource)
        agents['Decision_Heuristic_Transp'].append(self)
        print("FIFO_Transp_Decision created")

    def act(self, states):
        if states == None:
            return None, None
        for order in sorted(states, key=lambda x: x.id, reverse=False):  # FIFO sort based on ID
            if order.get_next_step().is_free_machine_group() and not order.reserved:
                order = states.pop(states.index(order))
                order.reserved = True
                mach, _ = self.get_next_machine_min_buffer_fill(order=order, statistics=self.statistics, parameters=self.parameters, resources=self.resources)
                return order, mach
        return None, None

class Decision_Heuristic_Machine_FIFO(Decision_Heuristic):
    """Selects the next processing order for a machine agent based on the total order waiting time"""
    def __init__(self, env, statistics, parameters, resources, agents, agents_resource):
        super(self.__class__, self).__init__(env=env, statistics=statistics, parameters=parameters, resources=resources, agents=agents, agents_resource=agents_resource)
        agents['Decision_Heuristic_Machine'].append(self)
        print("FIFO_Machine_Decision created")

    def act(self, states):
        if states == None:
            return None
        for order in states:
            return [order]