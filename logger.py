import numpy as np
import pandas as pd
import csv
import os
from datetime import datetime

class Console_export(object):
    def __init__(self, path):
        self.path = path + "_sim_summary.txt"
    def printLog(self, *args, **kwargs):
        print(*args, **kwargs)
        with open(self.path,'a') as file:
            print(*args, **kwargs, file=file)

def export_statistics_logging(statistics, parameters, resources):
    if parameters['EXPORT_NO_LOGS']: return None
    statistics['sim_end_time']  = datetime.now()
    path = parameters['PATH_TIME']
    ce = Console_export(path)
    ce.printLog("Start logger ", datetime.now())

    """ 
    Statistics & Logging 
    """
    # Cut-off last processes at end of simulation
    for mach in range(parameters['NUM_MACHINES']):
        list_of_stats = ['stat_machines_working', 'stat_machines_changeover', 'stat_machines_broken',
                         'stat_machines_idle']
        for stat in list_of_stats:
            if stat == 'stat_machines_working':
                if resources['machines'][mach].last_process_start > statistics['time_end']:
                    resources['machines'][mach].last_process_start -= resources['machines'][mach].last_broken_time
                if resources['machines'][mach].last_process_start + resources['machines'][mach].last_process_time > statistics['time_end']:
                    statistics[stat][mach] -= resources['machines'][mach].last_process_start + resources['machines'][mach].last_process_time - statistics['time_end']
            if stat == 'stat_machines_broken':
                if resources['machines'][mach].last_broken_start + resources['machines'][mach].last_broken_time > statistics['time_end']:
                    statistics[stat][mach] -= resources['machines'][mach].last_broken_start + resources['machines'][mach].last_broken_time - statistics['time_end']

    statistics['stat_machines_working'] = np.true_divide(statistics['stat_machines_working'], statistics['time_end'])
    statistics['stat_machines_changeover'] = np.true_divide(statistics['stat_machines_changeover'], statistics['time_end'])
    statistics['stat_machines_broken'] = np.true_divide(statistics['stat_machines_broken'], statistics['time_end'])
    statistics['stat_machines_idle'] = np.true_divide(statistics['stat_machines_idle'], statistics['time_end'])

    statistics['stat_transp_working'] = np.true_divide(statistics['stat_transp_working'], statistics['time_end'])
    statistics['stat_transp_walking'] = np.true_divide(statistics['stat_transp_walking'], statistics['time_end'])
    statistics['stat_transp_handling'] = np.true_divide(statistics['stat_transp_handling'], statistics['time_end'])
    statistics['stat_transp_idle'] = np.true_divide(statistics['stat_transp_idle'], statistics['time_end'])
    
    ce.printLog("##########################")
    ce.printLog("Simulation")
    ce.printLog("##########################")
    ce.printLog("Start time: ", statistics['sim_start_time'])
    ce.printLog("End time: ", statistics['sim_end_time'])
    duration = statistics['sim_end_time'] - statistics['sim_start_time']
    ce.printLog("Duration [min]: ", duration.total_seconds() / 60.0)
    
    ce.printLog("##########################")
    ce.printLog("Orders")
    ce.printLog("##########################")
    ce.printLog('Finished orders: ', len(statistics['orders_done']))
    ce.printLog('Prefilled orders: ', statistics['stat_prefilled_orders'])
    cycle_time = 0.0
    for order in statistics['orders_done']:
        cycle_time += order.eop - order.sop
    ce.printLog('Average order cycle time: ', cycle_time / len(statistics['orders_done']))

    ce.printLog("##########################")
    ce.printLog("Maschines")
    ce.printLog("##########################")
    ce.printLog("Working - Changeover - Broken - Idle || Total")
    for i in range(parameters['NUM_MACHINES']):
        ce.printLog("{0:.3f}".format(statistics['stat_machines_working'][i]), "{0:.3f}".format(statistics['stat_machines_changeover'][i]), "{0:.3f}".format(statistics['stat_machines_broken'][i]), "{0:.3f}".format(statistics['stat_machines_idle'][i]), " || ",
            "{0:.3f}".format(statistics['stat_machines_working'][i]+statistics['stat_machines_changeover'][i]+statistics['stat_machines_broken'][i]+statistics['stat_machines_idle'][i]))
    ce.printLog("--------------------------")
    ce.printLog("{0:.3f}".format(np.mean(statistics['stat_machines_working'])), "{0:.3f}".format(np.mean(statistics['stat_machines_changeover'])), "{0:.3f}".format(np.mean(statistics['stat_machines_broken'])), "{0:.3f}".format(np.mean(statistics['stat_machines_idle'])), " || ",
            "{0:.3f}".format(np.mean(statistics['stat_machines_working']) + np.mean(statistics['stat_machines_changeover']) + np.mean(statistics['stat_machines_broken']) + np.mean(statistics['stat_machines_idle'])))
    ce.printLog("##########################")
    ce.printLog("Transport")
    ce.printLog("##########################")
    ce.printLog("Working - Walking - Handling - Idle || Total")
    for i in range(parameters['NUM_TRANSP_AGENTS']):
        ce.printLog("{0:.3f}".format(statistics['stat_transp_working'][i]), "{0:.3f}".format(statistics['stat_transp_walking'][i]), "{0:.3f}".format(statistics['stat_transp_handling'][i]), "{0:.3f}".format(statistics['stat_transp_idle'][i]), " || ",
            "{0:.3f}".format(statistics['stat_transp_walking'][i]+statistics['stat_transp_handling'][i]+statistics['stat_transp_idle'][i]))
    ce.printLog("--------------------------")
    ce.printLog("{0:.3f}".format(np.mean(statistics['stat_transp_working'])), "{0:.3f}".format(np.mean(statistics['stat_transp_walking'])), "{0:.3f}".format(np.mean(statistics['stat_transp_handling'])), "{0:.3f}".format(np.mean(statistics['stat_transp_idle'])), " || ",
            "{0:.3f}".format(np.mean(statistics['stat_transp_walking']) + np.mean(statistics['stat_transp_handling']) + np.mean(statistics['stat_transp_idle'])))
    ce.printLog("##########################")

    # Close report file
    statistics['agent_reward_log'].close()
    statistics['episode_log'].close()

    # Calculate statistics of last quarter
    pd_episode_log = pd.read_csv(parameters['PATH_TIME'] + "_episode_log.txt", sep=",", header=0, index_col=0)
    last_quarter = int(len(pd_episode_log.index) / 4)
    dt_weights_time = pd_episode_log['dt'].tail(last_quarter).tolist()
    dt_weights_orders = pd_episode_log['finished_orders'].tail(last_quarter).tolist()
    lq_stats = dict()
    for kpi in pd_episode_log.columns:
        if kpi in ['dt', 'dt_real_time', 'valid_actions', 'total_reward', 'machines_total', 'selected_idle', 'forced_idle', 'threshold_waiting', 'finished_orders', 'processed_orders']:
            lq_stats.update({kpi: np.average(pd_episode_log[kpi].tail(last_quarter).tolist())})
        elif kpi in ['machines_working', 'machines_changeover', 'machines_broken', 'machines_idle', 'machines_processed_orders', 'transp_working', 'transp_walking', 'transp_handling', 'transp_idle', 'alpha', 'inventory']:
            lq_stats.update({kpi: np.average(pd_episode_log[kpi].tail(last_quarter).tolist(), weights=dt_weights_time)})
        elif kpi in ['order_waiting_time']:
            lq_stats.update({kpi: np.average(pd_episode_log[kpi].tail(last_quarter).tolist(), weights=dt_weights_orders)})
        else:
            lq_stats.update({kpi: 0.0})
    pd.DataFrame.from_dict(lq_stats, orient="index").to_csv(parameters['PATH_TIME'] + "_kpi_log.txt", sep=",", header=0)

    ce.printLog("Export order log ", datetime.now())

    export_df = []
    for x in statistics['orders_done']:
        export_df.append(x.order_log)
    pd.DataFrame(export_df).to_csv(str(path) + '_order_log.txt', header=None, index=None, sep=',', mode='a')

    ce.printLog("Export transport log ", datetime.now())

    export_df = pd.DataFrame(columns = None)
    for x in resources['transps']:
        temp_df = pd.DataFrame(x.transp_log)
        new_header = temp_df.iloc[0] 
        temp_df = temp_df[1:]
        temp_df.columns = new_header 
        temp_df = temp_df.add_prefix("transp_" + str(x.id) + "_")  
        export_df = pd.concat([export_df, temp_df], axis=1)
    export_df.to_csv(str(path) + '_transport_log.txt', index=None, sep=',', mode='a')

    ce.printLog("Export machine log ", datetime.now())

    export_df = pd.DataFrame(columns = None)
    for x in resources['machines']:
        temp_df = pd.DataFrame(x.machine_log)
        new_header = temp_df.iloc[0] 
        temp_df = temp_df[1:]
        temp_df.columns = new_header 
        temp_df = temp_df.add_prefix("machine_" + str(x.id) + "_") 
        export_df = pd.concat([export_df, temp_df], axis=1)
    export_df.to_csv(str(path) + '_machine_log.txt', index=None, sep=',', mode='a')
                
    ce.printLog("End logger ", datetime.now())