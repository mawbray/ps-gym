import ps_gym
import gym
import numpy as np

def assign_env_config(self, kwargs):
    for key, value in kwargs.items():
        setattr(self, key, value)
    if hasattr(self, 'env_config'):
        # print(self.env_config)
        for key, value in self.env_config.items():
            # Check types based on default settings
            if hasattr(self, key):
                if type(getattr(self,key)) == np.ndarray:
                    setattr(self, key, value)
                else:
                    setattr(self, key,
                        type(getattr(self, key))(value))
            else:
                setattr(self, key, value)
                
# Get Ray to work with gym registry
def create_env(config, *args, **kwargs):
    
    if type(config) == dict: env_name = config['env']
    else: env_name = config

       
    if env_name == 'SingleStageParallel-v0': from ps_gym.envs.singlestageparallel import SingleStageParallelMaster as env        
    if env_name == 'SingleStageParallel-v1': from ps_gym.envs.singlestageparallel import SingleStageParallelSO1 as env
        
    else: raise NotImplementedError('Environment {} not recognized.'.format(env_name))
    
    return env
