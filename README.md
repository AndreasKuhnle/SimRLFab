# SimPyRLFab
Simulation and reinforcement learning framework for production planning and control of complex job shop manufacturing systems.

## Introduction

Complex job shop manufacturing systems are motivated by the manufacturing characteristics of the semiconductor wafer fabrication. A job shop consists of several machines (processing resources) that process jobs (products, orders) based on a defined list or process steps. After every process, the job is dispatched and transported to the next processing machine. Machines are usually grouped in sub-areas by the type processing type, i.e. similar processing capabilities are next to each other. 

In operations management, two tasks are considered to improve opperational efficiency, i.e. increase capacity utilization, raise system throughput, and reduce order cycle times. First, job shop scheduling is an optimization problem which assigns a list of jobs to machines at particular times. It is considered as NP-hard due to the large number of constraints and even feasible solutions can be hard to compute in reasonable time. Second, order dispatching optimizes the order flow and dynamically determines the next processing resource. Depending on the degree of stochastic processes either scheduling or dispatching is enforced. In manufacturing environments with a high degree of unforseen and stochastic processes, efficient dispatching approaches are required to operate the manufacturing system on a robust and high performance. 

According to Mönch (2013) there are several characteristics that cause the complexity characteristics:
- Unequal job release dates
- Sequence-dependent setup times
- Prescribed job due dates
- Different process types (e.g., single processing, batch processing)
- Frequent machine breakdowns and other disturbances
- Re-entrant flows of jobs

> Mönch, L., Fowler, J. W., & Mason, S. J. (2013). Production Planning and Control for Semiconductor Wafer Fabrication Facilities.

This framework provides an integrated simulation and reinforcement learning model to investigate the potential of data-driven reinforcement learning in production planning and control of complex job shop systems. The simulation model allows parametrization of a broad range of job shop-like manufacturing systems. Furthermore, performance statistics and logging of performance indicators are provided. Reinforcement learning is implemented to control the order dispatching and several dispatchin heuristics provide benchmarks that are used in practice. 

## Features

The simulation model covers the following features (`initialize_env.py`):
- Number of resources:
    - Machines: processing resources
    - Sources: resources where new jobs are created and placed into the system
    - Sinks: resources where finished jobs are placed
    - Transport resources: dispatching and transporting resources
- Layout of fix resources (machines, sources, and sinks) based on a distance matrix
- Sources:
    - Buffer capacity (only outbound buffer)
    - Job release / generation process
    - Restrict jobs that are released at a specific source
- Machines:
    - Buffer capacity (inbound and outbound buffers)
    - Process time and distribution for stochastic process times
    - Machine group definition (machines in the same group are able to perform the same process and are interchangeable)
    - Breakdown process based on mean time between failure (MTBL) and mean time of line (MTOL) definition
    - Changeover times for different product variants
- Sinks:
    - Buffer capacity (only inbound buffer)
- Transport resources:
    - Handling times
    - Transport speed
- Others:
    - Distribution of job variants
    - Handling times to load and unload resources
    - Export frequency of log-files
    - Turn on / off console printout for detailed report of simulation processes and debugging
    - Seed for random number streams

<p align="center"> 
<img src="/docu/layout.png" width="500">
</p>

The reinforcement learning is based on the **Tensorforce** library and allows the combination of a variety of popular deep reinforcement learning models. Further details are found in the **Tensorforce** documentation. Problem-specific configurations for the order dispatching task are the following (`initialize_env.py`):
- State representation, i.e. which information elements are part of the state vector
- Reward function (incl. consideration of multiple objective functions and weighted reward functions according to action subset type)
- Action representation, i.e. which actions are allowed (e.g., "idling" action) and type of mapping of discrete action number to dispatching decisions
- Episode definition and limit
- RL-specific parameters such as learning rate, discount rate, neural network configuration etc. are defined in the Tensorforce agent configuration file

In practice, heuristics are applied to control order dispatching in complex job shop manufacturing systems. The following heuristics are provided as benchmark:
- **FIFO**: First In First Out selects the next job accoding to the sequence of appearance in the system to prevent long cycle times. 
- **NJF**: Nearest Job First dispatches the job which is closest to the machine to optimize the route of the dispatching / transport resource.
- **EMPTY**: It dispatches a job to the machine with the smalles inbound buffer to supply machines that run out of jobs.
The destination machine for the next process of a job, if alternative machines are available based on the machine group definition, is determined for all heuristics according to the smallest number of jobs in the inbound buffer.

By default, the sequencing and processing of orders at machines is based on a FIFO-procedure.

The default configuration provided in this package is based on a semiconductor setting presented in:
> Kuhnle, A., Röhrig, N., & Lanza, G. (2019). "Autonomous order dispatching in the semiconductor industry using reinforcement learning", Procedia CIRP, p. 391-396

> Kuhnle, A., Schäfer, L., Stricker, N., & Lanza, G. (2019). "Design, Implementation and Evaluation of Reinforcement Learning for an Adaptive Order Dispatching in Job Shop Manufacturing Systems". Procedia CIRP, p. 234-239.

## Running guide

Set up and run a simulation and training experiment:
1. Define production parameters and agent configuration (see above, `initialize_env.py`)
2. Set timesteps per episode and number of episodes (`run.py`)
3. Select RL-agent configuration file (`run.py`)
4. Run
5. Analyse performance in log-files

## Installation

Required packages (Python 3.6): 
```bash
pip install -r requirements.txt
```

## Extensions (not yet implemented)

- Job due dates
- Batch processing
- Alternative maintenance strategies (predictive, etc.)
- Alternative strategies for order sequencing and processing at machines
- Mutliple RL-agents for several production control tasks
- etc.

## Acknowledgments

We extend our sincere thanks to the German Federal Ministry of Education and Research (BMBF) for supporting this research (reference nr.: 02P14B161).
