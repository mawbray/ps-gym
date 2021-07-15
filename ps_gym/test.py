#!usr/bin/env python

'''
Tests to ensure environments load and basic functionality
is satisfied.
'''

#import or_gym
#from or_gym.version import ENV_LIST
import numpy as np
import gym
import matplotlib.pyplot as plt
from envs.singlestageparallel import *
#from env.experiment_definitions import *



def set_process_times1():

    process_time_e1         = [0.5, 0.5, False, False, 1.0, False, False, False, False, False]             # listing processing time of each product in unit 1
    process_time_e2         = [False, False, 0.5, 1.0, False, False, 0.5, False, False, False]            # listing processing time of each product in unit 2
    process_time_e3         = [False, False, False, False, False, 1.0, 0.5, 0.5, False, False]         # listing processing time of each product in unit 3
    process_time_e4         = [False, False, False, 1.0, False, 0.5, False, 1.0, False, False]        # listing processing time of each product in unit 4
    p_t                     = [process_time_e1, process_time_e2, process_time_e3, process_time_e4]      # collating all processing times for dictionary generation 

    return p_t


class TestEnv:
    def __init__(self):
        env = 'SingleStageParallel-v0'  # ENV_LIST
        print('initialised')
        self.test_episode(env)
        

    def _build_env(self, env_name):

        units       = [i for i in range(4)]
                 # get processing times 
        p_t             = set_process_times1()                                                  # retreving process times for each order in each unit
        procs_time_dict = {key: value for key, value in zip(units, p_t)}                        # generating dictionary of processing of products in each unit
        """gym.envs.register(
                            id=env_name,
                            entry_point='envs.singlestageparallel:SingleStageParallelMaster')
                            #max_episode_steps=150
                            #kwargs={'size' : 1, 'init_state' : 10., 'state_bound' : np.inf},)
        """
        env = gym.make(env_name, env_config =  {} )
        return env

    def test_make(self, config):
        # Ensures that environments are instantiated
        env_name = config #['env_name']
        try:
            _ = self._build_env(env_name)
            success = True
        except Exception as e:
            tb = e.__traceback__
            success = False
        assert success, ''.join(traceback.format_tb(tb))

    def plot(self, x, object):
        x = plt.hist(x)
        plt.xlabel(f'{object}')
        plt.ylabel('Value')
        plt.title(f'Histogram of {object}')
        plt.grid(True)
        plt.show()


    def test_episode(self, config):
        # Run 100 episodes and check observation space
    
        env_name = config #['env_name']
        EPISODES = 1
        #success = self.test_make(env_name)
        #assert success == True, f'fault in making environment -- {success}'
        env = self._build_env(env_name)
        rewards = []
        states  = []
        controls = []
        termin  = []
        rave    = []

       
        for ep in range(EPISODES):
            state, info = env.reset()
            print('available controls', info['control_set'])
            r = 0
            t= 0
            while True:
                #assert (env.observation_space.contains(state), 
                 #   f"State out of range of observation space: {state}")
                action = env.sample_action()
                print('control', np.array([8,8,8,8]))
                state, reward, done, info = env.step(np.array([8,8,8,8]))  #action
                print('processing', env.op_processing[t])
                r+=reward
                t+=1
                rewards.append(reward), states.append(state), controls.append(action), termin.append(done)
                
                if done:
                    print(f'completed episode @ timestep {t}')
                    break
            rave.append(r)
      
                

        assert done
        #env.generate_schedule()

        print('average reward', np.mean(rave))
        objects = [rewards]
        handles = ['rewards', 'controls']

        for i in range(len(objects)):
            self.plot(objects[i], handles[i])

        return
        
        
if __name__ == "__main__": 
    x = TestEnv()


