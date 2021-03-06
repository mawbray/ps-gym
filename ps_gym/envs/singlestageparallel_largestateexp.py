import sys
#sys.path.append(r'C:\Users\g41606mm\Dropbox\Projects\Python\RL\Scheduling\OR-gym\env\model_builder\production_scheduling\envs')
import gym
import itertools
import numpy as np
from scipy.stats import *
from ps_gym.utils import assign_env_config
from collections import deque
import math
import random
import matplotlib.pyplot as plt
import numpy.random as rnd





class SingleStageParallelLStateExp(gym.Env):
    '''
    The single stage parallel chemical batch production scheduling problem is organised as follows, according to
    https://doi.org/10.1021/ie9605490 (Cerda, Henning and Grossmann, 1997). The problem is transcribed from 
    the original continuous-time model to discrete-time model via the framework provided by https://doi.org/10.1016/j.compchemeng.2012.06.025
    (Subramanian, Maravelias and Rawlings, 2012). Obviously the more recent general state space formulation provided by 
    https://doi.org/10.3390/pr5040069 (Gupta and Maravelias, 2019) could well be used/adapted but it was not considered 
    necessary for this preliminary case study. The problem definition follows:

    1. Each customer order or job only involves a single product O_i,i∈I where I is the set of orders/products  
        -> total control set (defined in agent construction and restricted appropriately in future).

	2. A due date t_(E )(discrete time indices from current step) for each customer order has been specified. 
        The manufacture of a product for inventory without a particular due date is also made possible by assigning to it the last day of the planning horizon as a fictitious due date.
	
    3. Each order can be manufactured in a subset (L_i∈L) of the available equipment items (dynamically restricting control set)
	
    4. Batches of the same order are successively processed in the same unit: 
        i.e. the production is organised by campaigns – this is automatically assigned by the environment. 
	
    5. Each order comprises an integer number of full-size batches of the same size. 
        This is the usual policy in specialty chemical plants in order to operate equipment at maximum capacity.
	
    6. The processing time for order O_i at unit/line l (TP_il) depends on both the nature of the order O_i and the type of unit/line l. 
        The maximum batch size (B_il) for the product involved in the order O_i varies with the unit l -> encoded in the environment rules and determines total periods order is in manufacture/equipment is out of use
	
    7. To begin another task in a processing line or unit, the current one should be completed (i.e. sequencing) 
        -> dynamic restriction of control set available based on underlying system state.
	
    8. Before starting a new campaign, a changeover period for cleaning and equipment setup is usually needed. 
        Changeover time for a pair of orders O_m and O_i (TCL_mi) depends on both the nature of such orders and job sequence (O_m,O_i). – directed by environment rules
	
    9. Because of flavour and/or colour incompatibilities, some job sequences are forbidden at any equipment item – directed by environment rules 
        -> restricts control space dynamically
	
    10. Some of the units/lines are not available from the beginning of the time horizon of interest, i.e., finite unit release times. 
        Also, the production of some orders cannot be started at the beginning of the horizon, i.e.,  finite job release times
	
    11. Due to limited storage capacity, new shippings of raw materials may be received during the scheduling period. 
        Limitations on the availability of raw materials, utilities, or manpower may prevent one from using some otherwise feasible production schedules
	
    12. Resource constraints related to raw material availability, limited manpower, or maintenance periods will indeed be considered in {enterCS name}


    '''
    
    def __init__(self, *args, **kwargs):

        '''
        periods = [positive integer] number of periods in simulation. --> length of horizon / gcf of all processing times 
        I0 = [non-negative integer; dimension |Products|] initial inventories for each product.
        te = [non-negative integer; dimension |Products|] delivery date for each order/product.
        r = [non-negative float; dimension |Products|]  cost for tardiness of product.
   
        dist = [integer] value between 1 and 6. Specifies distribution for customer demand.
            1:poisson,
	    2:binom,
	    3:randint,
	    4:geom,
	    5:betabinom,
	    6:bernoulli,
	    7-9: User defined	    
        All distributions are specified by their arguments. see scipy for more information.
        seed_int = [integer] seed for random state.
        '''
        # set default (arbitrary) values when creating environment (if no args or kwargs are given)                                                                   
        self.studytype      = 'ex1'                                                                         # defining case study size 
        self.due_dates      = [10, 22, 25, 20, 28, 30, 17, 23, 30, 21, 30, 28, 15, 29, 40]                  # defining due date of each of the 10 orders/products to be made
        self.products       = [i for i in range(15)]                                                        # defining product number    
        self.due_date_dict  = {key:value for key, value in zip(self.products, self.due_dates)}              # creating look up dictionary of product due dates.
        
        self.n_units            = 4                                                                         # defining number of units available
        units                   = [i for i in range(self.n_units)]                                          # listing unit index
        
        # get feasible unit operations 
        ops                     = self.set_feasible_unit_ops(self.studytype)                                # retreving set of possible operations for each unit
        self.feasible_unit_ops  = {key: value for key,value in zip(units, ops)}                             # generating dictionary of feasible operations for each unit

        # get max batch size
        batch                   = self.set_max_batch_sizes()                                                # retreving max batch sizes for each order in each unit
        self.batch_size_dict    = {key: value for key, value in zip(units, batch)}                          # generating dictionary of max_batch_sizes of products in each unit
        
        # get processing times 
        p_t                     = self.set_process_times()                                                  # retreving process times for each order in each unit
        self.procs_time_dict    = {key: value for key, value in zip(units, p_t)}                            # generating dictionary of processing of products in each unit

        # get feasible successors
        feasible_set_successors = self.set_possible_successors(self.studytype)
        self.feasible_succs     = {key: value for key, value in zip(self.products, feasible_set_successors)}

        # get cleaning times
        c_t                     = self.set_cleaning_times()                                                 # collating all cleaning times for dictionary generation 
        self.clean_time_dict    = {key: value for key, value in zip(self.products, c_t)}                    # generating dictionary of cleaning_time of products in each unit
  
        # definition of horizon and discretisation 
        self.dt                 = 0.5                                                                       # gcf of all processing times and cleaning times
        self.periods            = math.ceil(100/self.dt)   -1                                 # number of periods required i.e. time horizon given                        
       
        # work up definitions 
        self.I_0                    = [0 for _ in range(20)]                                                                              
        self.finite_release_times   = False                                                                 # include finite release times or not?
        self.idle_option            = True                                                                  # provide control option to idle the unit
        self.allocation             = 'makespan'                                                            # reward definition
        self.shipping               = False                                                                 # if shipping from product from state inventory
        self.round2nearest          = True                                                                  # if rounding control to nearest allowables
        self.maintenance            = False
        self.rolling_horizon        = True

        # constraints definitions
        self.g_in_R     = False
        self.pf         = 20
        self.g_p_fn     = 2

        #  defining types of stohasticity 
        #  dictionary with options for distributions
        self.due_date_stochasticity     = False
        self.maintenance_stochasticity  = False
        self.processing_stochasticity   = False
        self.user_D1, self.user_D2, self.user_D3 = np.zeros(self.periods), np.zeros(self.periods), np.zeros(self.periods)
        self.distributions       = {1:poisson,
                                    2:binom,
                                    3:randint,
                                    4:geom,
                                    5:betabinom,
                                    6:bernoulli,
                                    7:self.user_D1, 8: self.user_D2, 9: self.user_D3}
        
        # defining parameters of default distributions
        # maintenance 
        self.breakdown_temp     = [1/20, 1/70, 1/80, 1/55]                          # inverse of mean number of days between unit breakdowns
        self.maintenance_t      = [1, 1, 1, 1]                                      # maintenance time for unit breakdown in days, note this could be defined probabilistically, but defined deterministically here
        self.dist_ubd           = {i:{'p':1-np.exp(-self.breakdown_temp[i]*self.dt)} for i in units}
        self.dist_ubdk          = 6
        # processing time 
        self.dist_p     = {i: {} for i in units}                                    # empty dictionary for unit order specific variability
        self.min_var_p  = -1                                                        # lower bound of uniform processing time variation (periods from the mean)
        self.max_var_p  = 2                                                        # upper bound of uniform processing time variation (periods from the mean)
        self.dist_ptk   = 3
        for i in units:
            for k in ops[i]:
                    self.dist_p[i][k] = [self.min_var_p, self.max_var_p]
        
        # implement control masking?
        self.dynamic_restrict_control = True

        # Defining elements of reward strucutre
        self.time_cost  = -1
        self.tardy_cost = -1

        # notice period for due dates (in discrete time indices)
        self.notice_period = 2 
                          
        # add environment configuration dictionary and keyword arguments
        assign_env_config(self, kwargs)

        # selection of case study
        if self.studytype == 'ex1': self.N = 15
        if self.studytype == 'ex2': self.N = 9
        if self.studytype == 'ex3': self.N = 10
        assert self.studytype in ['ex1','ex2','ex3'], 'studytype must be either "ex1", "ex2" or "ex3"'

        if self.finite_release_times:
            self.RTO = self.RTO_fn(self.studytype)
            self.RTE = self.RTE_fn()    

        # constraints 
        self.A, self.B 	= np.diag([1]*self.N), np.ones(self.N).reshape(-1,1)
        self.ng 	    = self.B.shape[0]
        self.g 		    = lambda x: self.A@x - self.B 			                                                # definition of linear constraints - x must be shape (nx,1) 
        self.g_terminal = [False]*self.ng

        # get order sizes
        orders                  = self.set_order_size(self.studytype)
        self.order_size_dict    = {key: value for key, value in zip(self.products, orders)}

        # demand and definitions of distribtuions objects.
        self.dist_ddk = 1
        self.dist_ddd = {i:{'mu':int(k)} for i,k in zip(self.products, self.due_dates)}            # constrained to be poisson 

        self.dist_pt        = [self.dist_ptk, self.dist_p]                                                          # selecting discrete distribution for processing times and setting params - see for user defined options (https://doi.org/10.1016/S0098-1354(01)00735-9)
        self.dist_ubd       = [self.dist_ubdk, self.dist_ubd ]                                               # selecting distribution for unit breakdown (typically bernouilli https://doi.org/10.1016/j.compchemeng.2019.106670) and setting params
        self.dist_dd        = [self.dist_ddk, self.dist_ddd ]              # selecting distribution for due date (in DOI:10.1021/ie0007724 uniform is selected but, typically it seems to be poisson)  and setting params - here demand will 

        # allocating aggregation function for constraints
        if self.g_p_fn == 1: self.agg_fn, self.p_norm = np.linalg.norm, 1
        if self.g_p_fn == 2: self.agg_fn, self.p_norm = np.linalg.norm, 2
        if self.g_p_fn == 3: self.agg_fn, self.p_norm = lambda x: sum(x.pow(2)), False
        if self.g_p_fn == 4: self.agg_fn, self.p_norm = lambda x: max(x) + 1/10 * np.log(sum(np.exp(10*(x - max(x))))) , False            # aggregate function as used in https://arxiv.org/pdf/2012.11790.pdf
        if self.g_p_fn not in [1,2,3,4]: raise ValueError('g_p_fn must be in [1,2,3, 4]')
        
        
        self.I0  = [self.I_0[i] for i in range(self.N)]                                            # initial inventory of products    
        self.seed_int   = 0
        # input parameters
        try:
            self.init_inv = np.array(list(self.I0))
        except:
            self.init_inv = np.array([self.I0])
        self.num_periods = self.periods
        
        # check inputs
        assert np.all(self.init_inv) >=0, "The initial inventory cannot be negative"
        try:
            assert self.num_periods > 0, "The number of periods must be positive. Num Periods = {}".format(self.num_periods)
        except TypeError:
            print('\n{}\n'.format(self.num_periods))
        assert np.all(self.time_cost <= 0), "The rewards for length of time horizon cannot be positive."

        # action space (orders available to schedule)
        m       = self.N
        u       = self.n_units
        self.nx = m+2*u+1
        self.action_space = gym.spaces.Box(
            low=np.zeros(self.n_units), high=self.N, dtype=np.int16)
        # observation space (on-hand Inventory position at each echelon, which is any integer value)
        self.observation_space = gym.spaces.Box(
            low=-np.ones(self.nx)*np.array(self.set_order_size(self.studytype)).max()*self.num_periods*10,
            high=np.ones(self.nx)*np.array(self.set_order_size(self.studytype)).max()*self.num_periods, dtype=np.int32)

        # intialize
        self.reset()

    def seed(self,seed=None):
        '''
        Set random number generation seed
        '''
        # seed random state
        if seed != None:
            np.random.seed(seed=int(seed))

    def set_feasible_predecessors(self):
        # feasible predecessors
        feasible_pretask_o1     = [2, 5, 8, 9, 12]         
        feasible_pretask_o2     = [2, 6, 10, 12]
        feasible_pretask_o3     = [1, 5, 8, 9, 11, 12]          
        feasible_pretask_o4     = [4, 5, 9, 13, 14]          
        feasible_pretask_o5     = [3, 6, 9, 11, 13]          
        feasible_pretask_o6     = [0, 4, 8, 12, 13, 14]          
        feasible_pretask_o7     = [2, 4, 7, 11, 12, 14]
        feasible_pretask_o8     = [4, 6, 11, 14]
        feasible_pretask_o9     = [0, 2, 5, 9, 12]
        feasible_pretask_o10    = [0, 2, 3, 8, 12, 13, 14]
        feasible_pretask_o11    = [1, 2, 7, 11, 12]
        feasible_pretask_o12    = [2, 4, 6, 7, 10, 12, 14]
        feasible_pretask_o13    = [0, 1, 5, 8, 9, 10]
        feasible_pretask_o14    = [3, 4, 5, 9, 14]
        feasible_pretask_o15    = [3, 5, 7, 9, 11, 13]

        feasible_set = [feasible_pretask_o1, feasible_pretask_o2, feasible_pretask_o3, feasible_pretask_o4, \
                        feasible_pretask_o5, feasible_pretask_o6, feasible_pretask_o7, feasible_pretask_o8, \
                        feasible_pretask_o9, feasible_pretask_o10, feasible_pretask_o11, feasible_pretask_o12,\
                        feasible_pretask_o13, feasible_pretask_o14, feasible_pretask_o15]
        
        return feasible_set

    def set_feasible_unit_ops(self, CS):

        if CS == 'ex1':
            e1 = [0, 2, 5, 8, 9, 12]
            e2 = [3, 4, 5, 9, 13, 14]
            e3 = [1, 2, 6, 10, 11, 12]
            e4 = [4, 6, 7, 11, 14]
        if CS == 'ex2':
            e1 = [0, 2, 5, 8]
            e2 = [3, 4, 5]
            e3 = [1, 2, 6]
            e4 = [4, 6, 7]
        if CS == 'ex3':
            e1 = [0, 2, 5, 8, 9]
            e2 = [3, 4, 5, 9]
            e3 = [1, 2, 6]
            e4 = [4, 6, 7]

        return [e1, e2, e3, e4]



    def set_possible_successors(self, CS):

        if CS == 'ex1':
            o1  = [5, 8, 9, 12]
            o2  = [2, 10, 12]
            o3  = [0, 1, 6, 8, 9, 10, 11]
            o4  = [4, 9, 13, 14]
            o5  = [3, 5, 6, 7, 11, 13]
            o6  = [0, 2, 3, 8, 12, 13, 14]
            o7  = [1, 4, 7, 11]
            o8  = [6, 10, 11, 14]
            o9  = [0, 2, 5, 9, 12]
            o10 = [0, 2, 3, 4, 8, 12, 13, 14]
            o11 = [1, 11, 12]
            o12 = [2, 4, 6, 7, 10, 14]
            o13 = [0, 1, 2, 5, 6, 8, 9, 10, 11]
            o14 = [3, 4, 5, 9, 14]
            o15 = [3, 5, 6, 7, 9, 11, 13]

        if CS == 'ex2':
            o1  = [5, 8]
            o2  = [2]
            o3  = [0, 1, 6, 8]
            o4  = [4]
            o5  = [3, 5, 6, 7]
            o6  = [0, 2, 3, 8]
            o7  = [1, 4, 7]
            o8  = [6]
            o9  = [0, 2, 5]
            o10 = [0, 2, 3, 4, 8]

        if CS == 'ex3':
            o1  = [5, 8, 9]
            o2  = [2]
            o3  = [0, 1, 6, 8, 9]
            o4  = [4, 9]
            o5  = [3, 5, 6, 7]
            o6  = [0, 2, 3, 8]
            o7  = [1, 4, 7]
            o8  = [6]
            o9  = [0, 2, 5, 9]
            o10 = [0, 2, 3, 4, 8]

        return [o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15]

    def set_process_times(self):

        process_time_e1         = [2.0, np.inf, 1.0, np.inf, np.inf, 2.5, np.inf, np.inf, 1.5, 2.5, np.inf, np.inf, 3.0, np.inf, np.inf]                 # listing processing time of each product in unit 1
        process_time_e2         = [np.inf, np.inf, np.inf, 1.5, 1.5, 2.0, np.inf, np.inf, np.inf, 2.0, np.inf, np.inf, np.inf, 2.5, 1.0]                    # listing processing time of each product in unit 2
        process_time_e3         = [np.inf, 1.0, 1.0, np.inf, np.inf, np.inf, 1.0, np.inf, np.inf, np.inf, 1.5, 2.0, 1.0, np.inf, np.inf]                    # listing processing time of each product in unit 3
        process_time_e4         = [np.inf, np.inf, np.inf, np.inf, 1.0, np.inf, 1.5, 2.0, np.inf, np.inf, np.inf, 1.5, np.inf, np.inf, 2.0]                 # listing processing time of each product in unit 4
        p_t                     = [process_time_e1, process_time_e2, process_time_e3, process_time_e4]                                                      # collating all processing times for dictionary generation 

        return p_t

    def set_max_batch_sizes(self):

        max_batch_e1            = [100, 0, 140, 0, 0, 280, 0, 0, 200, 250, 0, 0, 120, 0, 0 ]              # listing max_batch_size of each product in unit 1
        max_batch_e2            = [0, 0, 0, 120, 90, 210, 0, 0, 0, 270, 0, 0, 0, 115, 130]             # listing max_batch_size of each product in unit 2
        max_batch_e3            = [0, 210, 170, 0, 0, 0, 390, 0, 0, 0, 190, 140, 155, 0, 0]          # listing max_batch_size of each product in unit 3
        max_batch_e4            = [0, 0, 0, 0, 130, 0, 290, 120, 0, 0, 0, 150, 0, 0, 145]          # listing max_batch_size of each product in unit 4
        batch                   = [max_batch_e1, max_batch_e2, max_batch_e3, max_batch_e4]                  # collating all max_batch_size for dictionary generation 

        return batch

    def set_cleaning_times(self):
        # these are explicitly linked to the possible set of successors

        c_time_o1   = [np.inf, np.inf, np.inf, np.inf, np.inf, 0.5, np.inf, np.inf, 1.0, 0.5, np.inf, np.inf, 1.5, np.inf, np.inf]                       # listing cleaning time between end of production of order 1 and next successor (index in list)
        c_time_o2   = [np.inf, np.inf, 1.0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1.0, np.inf, 1.5, np.inf, np.inf]                          # listing cleaning time between end of production of order 2 and next successor (index in list)
        c_time_o3   = [1.0, 0.5, np.inf, np.inf, np.inf, np.inf, 0.5, np.inf, 1.5, 0.5, 1.0, 2.0, np.inf, np.inf, np.inf]                                      # listing cleaning time between end of production of order 3 and next successor (index in list)
        c_time_o4   = [np.inf, np.inf, np.inf, np.inf, 0.5, np.inf, np.inf, np.inf, np.inf, 0.5, np.inf, np.inf, np.inf, 2.0, 1.0]                          # listing cleaning time between end of production of order 4 and next successor (index in list)
        c_time_o5   = [np.inf, np.inf, np.inf, 0.5, np.inf, 0.5, 1.0, 0.5, np.inf, np.inf, np.inf, 0.5, np.inf, 0.5, np.inf]                                   # listing cleaning time between end of production of order 5 and next successor (index in list)
        c_time_o6   = [1.5, np.inf, 0.5, 0.5, np.inf, np.inf, np.inf, np.inf, 1.0, np.inf, np.inf, np.inf, 0.5, 1.0, 1.5]                                # listing cleaning time between end of production of order 6 and next successor (index in list)
        c_time_o7   = [np.inf, 2.0, np.inf, np.inf, 1.0, np.inf, np.inf, 0.5, np.inf, np.inf, np.inf, 1.0, np.inf, np.inf, np.inf]                             # listing cleaning time between end of production of order 7 and next successor (index in list)
        c_time_o8   = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1.5, np.inf, np.inf, np.inf, 0.5, 0.5, np.inf,  np.inf, 1.5]                 # listing cleaning time between end of production of order 8 and next successor (index in list)
        c_time_o9   = [2.0, np.inf, 1.0, np.inf, np.inf, 0.5, np.inf, np.inf, np.inf, 1.5, np.inf, np.inf, 3.0, np.inf, np.inf]                                # listing cleaning time between end of production of order 8 and next successor (index in list)
        c_time_o10  = [1.0, np.inf, 0.5, 0.5, 1.0, np.inf, np.inf, np.inf, 2.5, np.inf, np.inf, np.inf,  0.5, 2.0, 1.0]                                  # listing cleaning time between end of production of order 8 and next successor (index in list)
        c_time_o11  = [np.inf, 1.0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 0.5, 2.5, np.inf, np.inf]                      # listing cleaning time between end of production of order 8 and next successor (index in list)
        c_time_o12  = [np.inf, np.inf, 1.0, np.inf, 1.5, np.inf, 2.0, 1.0, np.inf, np.inf, 0.5, np.inf, np.inf, np.inf, 1.0]                   # listing cleaning time between end of production of order 8 and next successor (index in list)
        c_time_o13  = [1.5, 0.5, 2.0, np.inf, np.inf, 2.0, 2.5, np.inf, 0.5, 0.5, 1.0, 1.5, np.inf, np.inf, np.inf]                      # listing cleaning time between end of production of order 8 and next successor (index in list)
        c_time_o14  = [np.inf, np.inf, np.inf, 0.5, 0.5, 0.5, np.inf, np.inf, np.inf, 0.5, np.inf, np.inf, np.inf, np.inf, 0.5]                # listing cleaning time between end of production of order 8 and next successor (index in list)
        c_time_o15  = [np.inf, np.inf, np.inf, 0.5, np.inf, 0.5, 0.5, 0.5, np.inf, 1.5, np.inf, 0.5, np.inf, 0.5, np.inf]                   # listing cleaning time between end of production of order 8 and next successor (index in list)



        return [c_time_o1, c_time_o2, c_time_o3, c_time_o4, c_time_o5, c_time_o6, c_time_o7, c_time_o8, c_time_o9, c_time_o10, c_time_o11, c_time_o12, c_time_o13, c_time_o14, c_time_o15]

    def set_order_size(self, CS):

        if CS == 'ex1': return [700, 850, 900, 900, 500, 1350, 950, 850, 450, 650, 300, 450, 200, 700, 300]
        if CS == 'ex2': return [700, 850, 900, 900, 500, 1350, 950, 850, 450, False]
        if CS == 'ex3': return [550, 850, 700, 900, 500, 1050, 950, 850, 450, 650]

    def RTO_fn(self, CS):

        if CS == 'ex1': return [0, 5, 0, 6, 0, 2, 3, 0, 2, 6, 0, 1.5, 0, 0, 5.5]
        if CS == 'ex2': return [0, 5, 0, 6, 0, 2, 3, 0, 2, False]
        if CS == 'ex3': return [0, 5, 0, 6, 0, 2, 3, 0, 2, 6]

    def RTE_fn(self):

        return [0, 3, 2, 3]

        
    def _RESET(self):
        '''
        Create and initialize all variables and containers.
        Nomenclature:
            I = On hand inventory at the start of each period at each stage (except last one).
            T = Pipeline inventory/i.e. that which is in production currently
            R = Replenishment order placed at each period at each stage (except last one).
            D = Customer demand at each period (at the retailer)
            S = Sales performed at each period at each stage.
            B = Backlog at each period at each stage.
            LS = Lost sales at each period at each stage.
            P = Total profit at each stage.
        '''
        periods = self.num_periods
        m       = self.N
        u       = self.n_units
        I0      = self.init_inv
        order_list      = [value for value in self.order_size_dict.values()]
        due_dates_nom   = [value for value in self.due_date_dict.values()]
        if self.due_date_stochasticity: due_dates = [self.distributions[self.dist_dd[0]].rvs(**self.dist_dd[1][i]) for i in range(self.N)]
        if not self.due_date_stochasticity: due_dates = [value for value in self.due_date_dict.values()]
        self.due_date_ep = {order: due_dates[order] for order in range(self.N)}

        # simulation result lists
        self.I=np.zeros([periods+1, m])                     # inventory of each product at the beginning of each period
        self.T=np.zeros([periods+1, m])                     # pipeline inventory of each product at the beginning of each period
        self.D=np.zeros([periods+1, m])                     # outstanding demands size for each product
        self.F=np.zeros([periods+1, m])                     # demand for product fulfilled when?
        self.C=np.zeros([periods+1, m])                     # no of discrete time indices until order fulfilled.
        self.R=np.zeros([periods+1, m])                     # tardy or not? How many periods late are we?
        self.r_hist = np.zeros([periods+1])
        self.action_log     = np.zeros((periods+1, u))      # what operation is unit n performing (0-self.N-1 production, self.N idle)
        self.time_to_compl  = np.zeros((periods+1, u))      # how long until a given operation in a given unit is complete (in periods) - 0 means available now  
        self.time_processed = np.zeros((periods+1, u))      # time an operation has been processed for.
        self.state_hist     = np.zeros((periods+1,self.nx))
        self.outstanding_os = [i for i in range(m)]         # container to track outstanding operations
        self.most_recent_op = {}
        self.batch_periods  = {unit: [] for unit in range(u)}
        self.batch_sizes    = {unit: [] for unit in range(u)}
        self.time_2_comp    = {unit: 0 for unit in range(u)}
        self.op_in_unit     = {key: {} for key in range(u)}
        self.op_processing  = np.zeros((periods+1, u))      # container to track at what period an order was being processed in a given unit
        self.campaign_size  = {unit: 0 for unit in range(u)}
        self.comp_ops       = {}
        for unit in range(u):
            self.most_recent_op[unit] = [np.inf, 0]
        # initialization
        self.period         = 0                             # initialize time
        self.I[0,:]         = np.array(I0)                  # initial inventory
        self.D[0,:]         = np.array(order_list[:m])      # initial demands
        self.C[0,:]         = np.array(due_dates_nom[:m])       # time until due
        

        # set state
        self._update_state()
        self._dynamic_restrict_control({})
        
        return self.state, {'control_set': self.control_set}

    def _update_state(self):

        m = self.N 
        u = self.n_units
        t = self.period
        nx = self.nx
        bkdt = 1/np.array(self.breakdown_temp)
        state = np.zeros(nx)

        # update time left in unit 
        for key in self.time_2_comp.keys():
            self.time_2_comp[key] -= 1


        ## risk index append inventory
        if t == 0:
            state[:m] = (self.I[t].flatten() - self.D[0,:].flatten()) * self.C[t,:].flatten()
        else:
            state[:m] = (self.I[t].flatten() - - self.D[0,:].flatten()) * self.C[t,:].flatten()
        # append unit processing which order
        #print(state.shape)
        if t == 0:
            state[:u] = np.array([-1 for _ in range(u)]).flatten() 
        elif t > 0:
            state[:u] = self.action_log[t-1].flatten()
        #append time left in processing 
        if t == 0: 
            state[u:2*u] = np.array(list(self.time_2_comp.values())).flatten()
        if t > 0:
            state[u:2*u] = np.array(list(self.time_2_comp.values())).flatten()

        # append time index
        state[-1] = t

        self.state = state.copy()
        self.state_hist[t,:] = state.copy()

        return
    
    def _update_control_schedule(self,control):
        # function to calculate number of batches to run, given selection of new control. 
        n           = self.period
        n_units     = self.n_units
        

        if n > 0:
            control_log = self.action_log[n-1].copy()
        if n == 0:
            control_log = [np.inf for i in range(n_units)]
        order_size      = self.order_size_dict
        max_batch_size  = self.batch_size_dict
        control         = control.astype(int).reshape(-1)
        
     
        # checking if control selection is a new operation
        boolean_check   = [True for _ in range(n_units)]
        for i in range(n_units):
         
            if control_log[i] == control[i]:
                boolean_check[i] = False

            assert control[i] in self.control_set[i], f'control selected (order {control[i]+1}) not compatible with schedule in unit {i+1}'

        # if new operation, checking order size, previous operation and processing time (which is cleaning time + processing time).
        c_length    = [False for _ in range(n_units)]                   # campaign length memory allocation 
        process_p   = [False for _ in range(n_units)]                   # operational time for new control decision 
        self.new_ops = {}
        for i in range(n_units):

            # determine campaign length
            if boolean_check[i] == True:
                if control[i] < self.N:
                    oi_size  = order_size[control[i]] - self.I[n, control[i]]                                                   # finding batch size to make, based on known demand and insted we can say that there is due date uncertainty. 
                    if oi_size <= 0.:
                        control[i] = self.N
                        self.action_log[n,i] = control[i]
                        self.time_to_compl[n,i]= 1                      # because of initialisation next index should automatically be 0.
                        break
                    bs_size  = max_batch_size[i][control[i]]                                                                    # finding max_batch_size of order production in selected unit
                    assert (oi_size != False) & (bs_size != False), f" illigitimate control decision (unit {i+1}, order {control[i+1]}) - Does order exist? {oi_size} / can unit process order? {bs_size} "
                    c_length[i] = math.ceil(oi_size/bs_size)                                                                    # number of campaigns required to fullfil order
                    #print('campagins check', f'order {control[i]+1} unit {i+1}', oi_size, order_size[control[i]], self.I[n, control[i]],  bs_size, c_length[i], oi_size/bs_size )
                    if n> 0:
                        _, process_p[i], p_t = self._operational_time_left(i, control[i], c_length[i], self.most_recent_op[i])   # finding number of periods to schedule for based on previous control and unit and order selection
                        
                    else:
                        _, process_p[i], p_t = self._operational_time_left(i, control[i], c_length[i], [control_log[i], 0])   # finding number of periods to schedule for based on

                    self.new_ops[i] = control[i]
                    self.op_in_unit[i][n] = control[i]
                    self.batch_sizes[i] = {control[i]: [max_batch_size[i][control[i]] for j in self.batch_periods[i][control[i]][1:]] }            # allocating batch size 

                    # make scheduling decision for remaining periods and pencil in countdown for future time periods.
                    x = int(self.batch_periods[i][control[i]][-1] - n)
                    # make scheduling decision for remaining periods and pencil in countdown for future time periods.
                  
                    for j in range(process_p[i]):
                        
                        if n+j <= self.num_periods:
                            self.action_log[n+j, i]     = control[i]                # scheduling 
                            self.time_to_compl[n+j,i]   = process_p[i] - j          # estimating operational period (might be good practice, might not be.. )
                            if (process_p[i] - j) <= p_t: 
                                 self.op_processing[n+j, i] = 1 * (control[i] +1) 
                            else:
                                self.op_processing[n+j, i] = -1 * (control[i] +1)
                                   

                    # reset time processeed
                    self.time_processed[n, i]  =  0

                elif control[i] == self.N:
                    
                    self.action_log[n,i] = control[i]
                    self.time_to_compl[n,i] = 1                      # because of initialisation next index should automatically be 0.
                    self.time_2_comp[i] = 1
                   

            elif boolean_check[i] == False:

                if control[i] == self.N:
                    self.action_log[n,i] = control[i]
                    self.time_to_compl[n,i]= 1                      # because of initialisation next index should automatically be 0.
                    self.time_2_comp[i] = 1
                else: 
                    pass 
                    
            # updating periods operation has been in processing (really a lazy way to do this) across all units
            if n < self.num_periods: self.time_processed[n+1, i] += 1 + self.time_processed[n, i]                       # index n+1 previously = 0

        return     

    def _operational_time_left(self, unit_index, order_index, campaigns, prev_op_index_t):

        process_time    = self.procs_time_dict
        clean_times     = self.clean_time_dict
        n               = self.period
        prev_op_index, t_n = prev_op_index_t
        dist_pt         = self.dist_pt

        # finding cleaning and release times
        # if no change of order
        if order_index == prev_op_index:                                  # this generally shouldn't happen based on previous general logic
            c_t = 0
            rte = 0
            rto = 0
        # if chnage of order and not first operation scheduled
        elif (order_index != prev_op_index) & (prev_op_index != np.inf):
            if prev_op_index < self.N:
                c_t = max(0, clean_times[int(prev_op_index)][int(order_index)] - (n - t_n)*self.dt)
            if prev_op_index == self.N: 
                c_t = 0
            # if another order has been previously scheduled and completed then we do not need to consider release times 
            # however unit may well have been idled in which case we need to consider how far the time horizon has progressed
            if self.finite_release_times:
                rto = max(0, self.RTO[int(order_index)] - n*self.dt) 
                rte = max(0, self.RTE[int(unit_index)] - n*self.dt) 
            if not self.finite_release_times:
                rto, rte = 0, 0
        # if first order scheduled in the unit then consider full release times
        elif (prev_op_index == np.inf):
            if self.finite_release_times:
                rto = max(0, self.RTO[int(order_index)] - n*self.dt) 
                rte = max(0, self.RTE[int(unit_index)] - n*self.dt) 
            if not self.finite_release_times:
                rto, rte = 0, 0
            c_t = 0       # assume no cleaning time required at the start of horzion i.e. no cleaning for first operation scheduled
        
        # finding processing time 
        p_tb = process_time[unit_index][order_index]
        # calculating finishing time of each batch within the campaign 
        ptplus = [0]
        if self.processing_stochasticity: 
            for i in range(campaigns):
                ptplus.append(self.distributions[dist_pt[0]].rvs(*dist_pt[1][unit_index][order_index]) *self.dt) # adding stochasticity 
        if not self.processing_stochasticity: 
            for i in range(campaigns):
                ptplus.append(0)
        # start times of the campaigns (also the end times of the n-1 campaign)
        prev_t_index = n + math.ceil(max(max(rto,c_t), rte)/self.dt)                # adding cleaning time and release times     
        self.batch_periods[unit_index]= {order_index: [n]}
        for i in range(campaigns):
            prev_t_index += math.ceil((max(self.dt, p_tb+ptplus[i])/self.dt))
            self.batch_periods[unit_index][order_index].append(prev_t_index)   

        # calculating entire time for processing of campaign
        p_t             = (self.batch_periods[unit_index][order_index][-1] - self.batch_periods[unit_index][order_index][0])*self.dt   
        lead_periods    = math.ceil(p_t/self.dt)
        p_T             = campaigns * p_tb + math.ceil(max(max(rto,c_t), rte)) 
        self.time_2_comp[unit_index] = p_T/self.dt
        
        
        
        return [p_t, lead_periods, lead_periods - math.ceil(max(max(rto,c_t), rte)/self.dt)]


    def _check_terminal_operations(self):

        n               = self.period
        n_units         = self.n_units
        order_size      = self.order_size_dict
        max_batch_size  = self.batch_size_dict
        batch_periods   = self.batch_periods

        if (n > 0) :
            
            control_log             = self.action_log[n-1].copy()               # previous control at time t-1
            time_to_complete_now    = self.time_to_compl[n,:].squeeze()         # time to complete from current time step t
            completed_ops   = {}

            for i in range(n_units):
                # determine if unit operation is complete 
                if int(time_to_complete_now[i]) == 0:
                    completed_ops[i] = control_log[i]
                    self.time_2_comp[i] =1
                    #if control_log[i] ==4 : print(f'yes, control {control_log[i]} did finish in unit {i}')
                    
                    if control_log[i] < self.N:
                        self.most_recent_op[i] = [control_log[i], n]
                        try:
                            if control_log[i] in self.outstanding_os:
                                self.outstanding_os.remove(int(control_log[i]))
                        except: 
                            print(f'WARNING: could not remove completed op {control_log[i]} from outstanding list {self.outstanding_os}')
                else: 
                    pass
        else:
            raise ValueError("Cannot check whether operation is terminal, time index must at least be greater than 0")


        return completed_ops 


    def _update_transition_(self, I, T, D, C, R, order_completion):

        # order completion is a dictionary of {unit:order}=={key:value}
        order_size      = self.order_size_dict
        max_batch_size  = self.batch_size_dict
        n               = self.period 
        new_ops         = self.new_ops
        due_dates       = self.due_date_ep
        batch_periods   = self.batch_periods
        batch_sizes     = self.batch_sizes


        # transferring pipeline inventory to on-hand inventory
        for unit, order_b in batch_periods.items():
            if type(order_b) is dict: 
                for order, date_comp in order_b.items(): 
                    count = 0
                    for date in date_comp[1:]:
                        #print(n, date_comp[1:], I[int(order)],batch_sizes[unit][int(order)][count]   )
                        if (date == n) & (order < self.N):
                            #print(unit, order, date,n ,(date == n), date_comp)
                            I[int(order)] += batch_sizes[unit][int(order)][count] 
                            T[int(order)] -= batch_sizes[unit][int(order)][count] 
                        count += 1


        # transferring new operations to pipeline inventory 
        for unit, order_b in batch_periods.items():
            if type(order_b) is dict: 
                for order, date_comp in order_b.items(): 
                    count = 0
                    n_d   = len(date_comp)
                    for j in range(n_d -1):
                        if (n>= date_comp[j]) & (n< date_comp[j+1]) & (order < self.N):
                            T[int(order)] = batch_sizes[unit][int(order)][j]                            

        # shipping deliveries that are ready, otherwise incurring penalty and updating tardy order tracker
        for key, value in due_dates.items():
            # updating time until orders are due
            if (key < self.N):
                    if key in self.outstanding_os: C[int(key)] -= self.dt 
                    if key not in self.outstanding_os: C[int(key)] = max(0, C[int(key)] - self.dt)
                    if (n* self.dt == value - self.notice_period*self.dt) or (C[int(key)] == 0): 
                        C[int(key)] = value - n*self.dt 
            if (math.floor(C[int(key)]/self.dt) <= int(0)) & (key <self.N):           # if order due 
                if ((I[int(key)] >= self.order_size_dict[int(key)]) & (D[int(key)] >= self.order_size_dict[int(key)])):   # check on-hand inventory can fulfill demand size
                    #print(f'True - order {key +1} shipped at time index {n}', (I[int(key)] >= self.order_size_dict[int(key)]), (D[int(key)] >= self.order_size_dict[int(key)]))
                    if self.shipping: I[int(key)] -= self.order_size_dict[int(key)]                                     # if we are shipping, remove mass from inventory
                    D[int(key)] -= self.order_size_dict[int(key)]                                                       # check off the demand
                    R[int(key)] = 0
                elif key not in self.outstanding_os:                                                                    # check if demand has been fulfilled                                    
                    R[int(key)] = 0
                else:
                    R[int(key)] = self.tardy_cost                                                                       # otherwise assume we are late and denote demand lateness 

            
                 


        return I, T, D, C, R

    def _round2nearest(self, control):

        control_set = self.control_set
        control_set = list(control_set.values())
        n_units     = self.n_units

  
        cs = []
        for i in range(n_units):

            try: 
                t = control.squeeze()[i]
                x = min(control_set[i], key=lambda x:abs(x-t))
      
            except:
                x = control_set[i]

            cs.append(x)


        return np.array(cs).squeeze().reshape(-1)


    def _dynamic_restrict_control(self, comp_ops):
        # return dictionary of feasible controls where each key is a unit and value is set of feasible controls

        successors  = self.feasible_succs.copy()            # dictionary of feasible successors to a given order
        unit_comp   = self.feasible_unit_ops.copy()         # dictionary of unit-order compatibility
        controls    = self.action_log.copy()                # interested in what the control was at the previous iteration
        n           = self.period                           # what is the current period?
        completedops= comp_ops.copy()                       # dictionary of completed operations
        n_units     = self.n_units                          # number of units available 
        control_log = self.action_log[n-1].copy()           # controls at last time period 
        controlset  = {}                                    # holder
        os          = self.outstanding_os.copy()
        recent_op   = self.most_recent_op


        if n == 0: 
            for i in range(n_units):
                j             = unit_comp[i].copy()                     # if first control interaction, all options are open 


                if self.idle_option: j.append(self.N)                   # add option to idle machine
                controlset[i] = j                                       # append controlset to QL
                    
            
        
        elif n > 0:                                                     # if not first control interaction lets have a look

            for i in range(n_units):                
             
                if i in completedops.keys():                                                        # if unit has just completed an operation
                    last_op         = completedops[i]                                               # retrieve operation just completed
                    l_op            = recent_op[i][0]
                    if last_op < self.N:
                        feasible_succ   = successors[last_op]                                       # return those feasible successor ops of prev. completed operation
                    elif (last_op == self.N) & (l_op != np.inf) :
                        feasible_succ   = successors[l_op]
                    else: 
                        feasible_succ  = unit_comp[i] 
                    
                    
                    feasible_ops    = unit_comp[i]                                                  # return those feasible operations in this unit 
                    intersection    = set(feasible_succ).intersection(set(feasible_ops))            # check intersection of those two lists
                    j               = list(intersection.intersection(set(os)))                      # check intersections of feasibly possible tasks and those tasks which are yet to be completed
                    if self.idle_option: j.append(self.N)                                           # add option to idle machine

                    controlset[i]   = j                                                             # define controls 


                elif i not in completedops.keys():
                  
                    controlset[i]   = [control_log[i]]                                              # if unit has not completed the operation, then the only possible choice is to continue with that operation
                    

        self.control_set = controlset

        
        return controlset

    def _schedule_maintenance(self, control_set, comp_ops, control):
        """ 
        implement maintenace period by clearing current operation and forcing unit to play
        maintenance
        """

        dist_ubd    = self.dist_ubd                         # distribution describing probability of unit breakdown
        m_t         = self.maintenance_t                    # required maintenance period for each unit
        n_units     = self.n_units                          # number of units available 
        controlset  = control_set
        deterbkd    = self.breakdown_temp
        n           = self.period
        campaign_prog = self.batch_periods
        # assess unit breakdown
        
        for i in range(n_units):

            if (i not in comp_ops.keys()) & (self.action_log[n,i] < self.N) :                    # if unit has not completed previous operation 

                if self.maintenance_stochasticity:
                    if self.distributions[dist_ubd[0]].rvs(**dist_ubd[1][i]) == 1:
                        # declaring maintenance period 
                        controlset[i] = [self.N+1]                                                  
                        mt            = math.ceil(m_t[i]/self.dt)
                        indi          = np.where(self.time_to_compl[n:,i] > 0)[0] + n
                        # wiping markers for previously scheduled operations
                        if indi.shape[0] >0:
                            self.time_to_compl[indi,i] = np.zeros(len(indi))#.reshape(-1,1)
                            self.time_2_comp[i]        = mt +1
                            self.op_processing[indi,i] = np.zeros(len(indi))#.reshape(-1,1)
                        # wiping pipeline inventory and markers for that unit
                        self.batch_periods[i]   = []
                        self.batch_sizes[i]     = []
                        self.T[n,int(self.action_log[n,i])] = 0
                        # allocating maintenance
                        for j in range(mt):
                            if n+j <= self.num_periods:
                                    self.action_log[n+j, i]     = int(self.N+1)                     # scheduling 
                                    self.time_to_compl[n+j,i]   = mt - j                            # estimating operational period (might be good practice, might not be.. )
                                    if (mt - j) <= math.ceil(m_t[i]/self.dt): 
                                            self.op_processing[n+j, i] = 1 * (self.N+2) 
                                    else:
                                        self.op_processing[n+j, i] = -1 * (self.N+2)
                               
                if not self.maintenance_stochasticity:
                    if int(1/deterbkd[i]) == int(n):
                        # declaring maintenance period 
                        controlset[i] = [self.N+1]                                                  
                        mt            = math.ceil(m_t[i]/self.dt)
                        indi          = np.where(self.time_to_compl[n:,i] > 0)[0] + n
                        # wiping markers for previously scheduled operations
                        if indi.shape[0] >0:
                            self.time_to_compl[indi,i] = np.zeros(len(indi))#.reshape(-1,1)
                            self.op_processing[indi,i] = np.zeros(len(indi))#.reshape(-1,1)
                            self.time_2_comp[i]        = mt +1
                        # wiping pipeline inventory and markers for that unit
                        self.batch_periods[i]   = []
                        self.batch_sizes[i]     = []
                        self.T[n,int(self.action_log[n,i])] = 0
                        # allocating maintenance
                        for j in range(mt):
                            if n+j <= self.num_periods:
                                    self.action_log[n+j, i]     = int(self.N+1)                     # scheduling 
                                    self.time_to_compl[n+j,i]   = mt - j                            # estimating operational period (might be good practice, might not be.. )
                                    if (mt - j) <= math.ceil(m_t[i]/self.dt): 
                                            self.op_processing[n+j, i] = 1 * (self.N+2) 
                                    else:
                                        self.op_processing[n+j, i] = -1 * (self.N+2)

                self.control_set = controlset

        return controlset



    def _check_constraint_set(self, t, x, u):
        """ 
	    Checking state transition does not violate constraints (point wise)
	    """
        op_in_unit  = self.op_in_unit
        n_units     = self.n_units 
        nx, ng      = self.nx, self.ng
        cf, nk      = self.g, self.num_periods
        term_chk    = self.g_terminal
        
        # constraints are defined in terms of binary variables - finding count of each element in 
        controls  = list(u)
        counts    = [0 for _ in range(self.N)]
        orders    = [i for i in range(self.N)]
        '''
        if self.control_penalty:
            for i in orders:
                counts[i] = controls.count(i)

        if not self.control_penalty:
            for unit in range(n_units):
                d  = list(op_in_unit[unit].values())
                for i in orders:
                    counts[i] += d.count(i)
        '''

        for i in orders:
            counts[i] = controls.count(i)

        x = np.array(counts)  
        con_value = cf(x.reshape(-1,1))

        if self.g_p_fn != 4:
            for i in range(ng):
                if t < nk:
                    if not term_chk[i]:				# if constraint is not terminal and t<nk  then enforce it 
                        con_value[i] = max(0, con_value[i])	        
                else:						        # if t == nk then enforce all constraints including those that are terminal
                    con_value[i] = max(0, con_value[i])

        agg_fn, p_norm = self.agg_fn, self.p_norm

        if p_norm:
            g_  = agg_fn(con_value.reshape(-1,1), ord = p_norm)
        if not p_norm:
            g_  = agg_fn(con_value.reshape(-1,1))
                   
                
        return g_.squeeze(), max(con_value.reshape(-1,1))

    def _reward_allocation(self, x, u):

        n = self.period 
        R = self.R

        # all allocation policies firsly minimise lateness and then additionally 

        if self.allocation == 'makespan': reward = np.sum(R[n,:].squeeze()).squeeze() + self.time_cost


        self.r_hist[n-1] = reward

        return reward
            
    def _STEP(self, control):
        '''
        Take a step in time in the multiperiod single stage parallel production scheduling problem.
        action = [integer; dimension |units|] order to process in each unit as scheduled by the control function
        '''
     
        # get inventory at hand and pipeline inventory at beginning of the period
        n = self.period
        I = self.I[n,:].copy()  # inventory at start of period n
        T = self.T[n,:].copy()  # pipeline inventory at start of period n
        D = self.D[n,:].copy()  # unfulfilled demand at start of period n 
        m = self.N              # number of products
        C = self.C[n,:].copy()  # due dates in periods remaining
        R = self.R[n,:].copy()

        # round control input determininstically to nearest allowable control
        if self.round2nearest: control = self._round2nearest(control) 
        #print('control', control)
        
        # update schedule based on current control decision 
        self._update_control_schedule(control)

        # ---- step period on and update state ---- # 
        self.period += 1
        n           = self.period

        # determine those operations with have finished
        comp_ops = self._check_terminal_operations()
        self.comp_ops = comp_ops
        
        # update inventory (on-hand and pipeline), ship and check if order is late
        I, T, D, C, R = self._update_transition_(I, T, D, C, R, comp_ops)

        # update self containers
        self.I[n,:] = I.copy()
        self.T[n,:] = T.copy()
        self.D[n,:] = D.copy()
        self.C[n,:] = C.copy()
        self.R[n,:] = R.copy()

        # need to implement dynamic restriction of states
        if self.dynamic_restrict_control: controlset = self._dynamic_restrict_control(comp_ops)
        if not self.dynamic_restrict_control: controlset = {} 

        if self.maintenance: controlset  = self._schedule_maintenance(controlset, comp_ops, control)

        # update state
        self._update_state()
        
        # check constraint
        g, max_g   = self._check_constraint_set(n,self.state,control.astype(int))
        
        # determine if simulation should terminate

        if (self.period >= self.num_periods-1) or ((self.outstanding_os == []) and (list(controlset.values()).count(list(controlset.values())[0]) == len(list(controlset.values())))) :
            done = True         
        else:
            done = False
      
        reward = self._reward_allocation(self.state, control)
        
        # asseting constraint on scheduled orders i.e. an order can only be scheduled once during the horizon 
        if self.g_in_R:
            reward -= self.pf * g

        return self.state, reward, done, {'control_set': controlset, 'processing': self.op_processing, 'g_vio': g, 'max_g': max_g}
    
    
    def sample_action(self):
        '''
        Generate an action by sampling from the action_space
        '''
        control = [None for _ in range(self.n_units)]

     

        for i in range(self.n_units):
            
            try: 
                control[i] = random.choice(self.control_set[i])
            except: 
                control[i] = self.control_set[i]


        return np.array(control).squeeze()

    def step(self, action):
        return self._STEP(action)

    def reset(self):
        return self._RESET()

    def runs_of_ones_list(self,unit):
        bits = self.op_processing[:,unit]
        return [sum(g) for b, g in itertools.groupby(bits)]
    
    def generate_schedule(self, path):
        n       = self.period
        units   = [i for i in range(self.n_units)] 
        op_pro  = self.op_processing
        control = self.action_log
        n_t, uni = control.shape
        control[n:,:] = np.array([-5]*(n_t-n)*uni).reshape(n_t-n, uni)
        schedule = {unit: [] for unit in units}
        
        for unit in units:

            x_  = control[0,unit]
            j   = 0
            gby = self.runs_of_ones_list(unit)
            print('time index pre' ,gby)
            gby = [item for item in gby if item > 0]
            print('time index post', gby)
            for i in range(n+1):
                x = control[i, unit]
                
                if (x != x_) & (x_!=self.N):
                    
                    try: schedule[unit].append([x_+1, i-(gby[j])/(x_+1), i])
                    except: print('please check schedule for ', unit, 'there may be error in schedule figure generation:', 
                                  'plotted schedule', schedule, 'controls taken', control[unit], 'op_processing', self.op_processing, 'time index', i)
                    j +=1
                if j >= len(gby):
                    break
                x_ = x.copy()

        schedule = {unit: np.array(schedule[unit]) for unit in units}
            
 
       
       
        from matplotlib import rc
        rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

       
        plt.figure(figsize=(12,6))
        colork = ['mediumblue' for i in range(self.N+2)] 
        gap = 20/5000000000
        idx = 1
        lbls = []
        ticks = []
        tmax = -1
        ds_max = 0
        for unit in units:
            idx -= 0.15
            #idx -= 1
            ticks.append(idx)
            lbls.append(f"Unit {unit+1}")
            for i in range(self.N):
                plt.plot([0,n],[idx,idx],lw=25,alpha=.01,color='white')
            #x = 0
            u_control = schedule[unit]

            
            for op in range(u_control.shape[0]):
                #x +=1
                ds = u_control[op]             
                if ds[2] > ds_max: ds_max = ds[2]
                plt.plot([ds[1]+gap,ds[2]-gap], [idx,idx],color=colork[int(ds[0]-1)], lw=25, solid_capstyle='butt')
                cond = ds[0] == self.N+2
                if not cond: txt = f"T{int(ds[0])}"
                if cond: txt = 'M'
                plt.text(ds[1] +(ds[2]-ds[1])/2, idx, txt, color='white', weight='bold', ha='center', va='center')
        plt.xlim(0,ds_max)
        plt.gca().set_yticks(ticks)
        plt.gca().set_yticklabels(lbls, fontsize = 14);
        plt.xlabel('Discrete Time Index', fontsize = 14, labelpad = 8)
        plt.savefig(path+'.svg')
        plt.close()
        
           

        R = self.R


        return

