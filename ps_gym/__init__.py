
from gym.envs.registration import register, registry, make, spec

# Production Scheduling Environments

register(id='SingleStageParallel-v0',
	entry_point='ps_gym.envs.singlestageparallel:SingleStageParallelMaster'
)

register(id='SingleStageParallel-v1',
	entry_point='ps_gym.envs.singlestageparallel:SingleStageParallelSO1'
)

register(id='SingleStageParallel-v2',
	entry_point='ps_gym.envs.singlestageparallel_large:SingleStageParallelLMaster'
)


