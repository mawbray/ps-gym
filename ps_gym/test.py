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




class TestEnv:
    def __init__(self):
        env = 'SingleStageParallel-v0'  # ENV_LIST
        print('initialised')
        self.test_episode(env)
        

    def _build_env(self, env_name):
        
        env = gym.make(env_name, env_config =  {} )         # env_config is a dictionary which takes environment attributes as keys and corresponding settings as values
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
                state, reward, done, info = env.step(action)  #action
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


