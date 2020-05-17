from production.envs.time_calc import *
from production.envs.heuristics import *
from production.envs.resources import *
from production.envs.transport import *
from production.envs.order import *
import simpy

class Source(Resource):
    counter_order = 0

    def __init__(self, env, id, capacity, resp_area, statistics, parameters, resources, agents, time_calc, location, label):
        Resource.__init__(self, statistics=statistics, parameters=parameters, resources=resources, agents=agents, time_calc=time_calc, location=location)
        print("Source %s created" % id)
        self.env = env
        self.id = id
        self.label = label
        self.type = "source"
        self.idle = env.event()
        self.resp_area = resp_area
        self.capacity = capacity
        self.buffer_out = []
        self.env.process(self.order_creating())  # Process started at creation
        self.source_wt_normalizer = None

    def put_buffer_out(self, order):
        self.statistics['stat_inv_buffer_out_mean'][1][self.id] = (self.statistics['stat_inv_buffer_out_mean'][1][self.id] * self.statistics['stat_inv_buffer_out_mean'][0][self.id] +
                (self.env.now - self.statistics['stat_inv_buffer_out_mean'][0][self.id]) * len(self.buffer_out)) / self.env.now
        self.statistics['stat_inv_buffer_out_mean'][0][self.id] = self.env.now
        self.statistics['stat_inv_buffer_out'][self.id] = len(self.buffer_out) + 1.0
        self.buffer_out.append(order)

    def get_buffer_out(self, order):
        self.statistics['stat_inv_buffer_out_mean'][1][self.id] = (self.statistics['stat_inv_buffer_out_mean'][1][self.id] * self.statistics['stat_inv_buffer_out_mean'][0][self.id] +
                (self.env.now - self.statistics['stat_inv_buffer_out_mean'][0][self.id]) * len(self.buffer_out)) / self.env.now
        self.statistics['stat_inv_buffer_out_mean'][0][self.id] = self.env.now
        self.statistics['stat_inv_buffer_out'][self.id] = len(self.buffer_out) - 1.0
        self.source_wt_normalizer(order.get_total_waiting_time())
        result_order = self.buffer_out.pop(self.buffer_out.index(order))
        if self.idle.triggered:
            self.idle = self.env.event()
            self.env.process(self.order_creating())
        return result_order

    def is_free(self):
        return False

    def is_free_machine_group(self):
        return False

    def get_max_waiting_time(self):
        max_wt = None
        max_wt = max([order.get_total_waiting_time() for order in self.buffer_out])
        return max_wt

    def get_normalized_wt_all_sources(self):
        return self.source_wt_normalizer.get_z_score_normalization(self.get_max_waiting_time())

    def get_inventory(self):
        return len(self.buffer_out)

    def order_creating(self):
        while True:
            if self.parameters['SOURCE_ORDER_GENERATION_TYPE'] == "ALWAYS_FILL_UP":
                if len(self.buffer_out) >= self.capacity:
                    self.idle.succeed()
                    break
                yield self.env.timeout(self.parameters['EPSILON'])

                if Source.counter_order >= self.parameters['NUM_ORDERS'] - 1:
                    break

            elif self.parameters['SOURCE_ORDER_GENERATION_TYPE'] == "MEAN_ARRIVAL_TIME":
                yield self.env.timeout(self.time_calc.time_to_order_generation(self, self.statistics, self.parameters))

            prod_steps, variant = self.time_calc.create_intermediate_production_steps_and_variant(statistics=self.statistics, parameters=self.parameters, resources=self.resources, at_resource=self)
            order = Order(env=self.env, id=Source.counter_order,prod_steps=prod_steps, variant=variant, statistics=self.statistics, parameters=self.parameters, resources=self.resources, agents=self.agents, time_calc=self.time_calc)
            Source.counter_order += 1

            order.set_sop()

            order.current_location = self
            self.put_buffer_out(order)
            order.order_log.append(["created", order.id, round(self.env.now, 5), self.id])

            self.env.process(order.order_processing())
