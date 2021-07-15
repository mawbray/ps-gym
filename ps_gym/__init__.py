
from gym.envs.registration import register, registry, make, spec

# Production Scheduling Environments

register(id='SingleStageParallel-v0',
	entry_point='ps_gym.envs.singlestageparallel:SingleStageParallelMaster'
)

register(id='SingleStageParallel-v1',
	entry_point='ps_gym.envs.singlestageparallel:SingleStageParallelSO1'
)

# Bin Packing Environments

# Newsvendor Envs


# Virtual Machine Packing Envs

# Inventory Management Envs
