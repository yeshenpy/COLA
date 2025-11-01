from gym.envs.registration import register


register(
    id = 'MO-Ant-v3',
    entry_point = 'environments.ant_v3:AntEnv',
    max_episode_steps=500,
)


register(
    id = 'MO-Ant-v2',
    entry_point = 'environments.ant:AntEnv',
    max_episode_steps=500,
)

register(
    id = 'MO-Hopper-v2',
    entry_point = 'environments.hopper:HopperEnv',
    max_episode_steps=500,
)

register(
    id = 'MO-Hopper-v3',
    entry_point = 'environments.hopper_v3:HopperEnv',
    max_episode_steps=500,
)

register(
    id = 'MO-Hopper-v5',
    entry_point = 'environments.hopper_v5:HopperEnv',
    max_episode_steps=500,
)

register(
    id = 'MO-HalfCheetah-v5',
    entry_point = 'environments.half_cheetah_v5:HalfCheetahEnv',
    max_episode_steps=500,
)

register(
    id = 'MO-Ant-v5',
    entry_point = 'environments.ant_v5:AntEnv',
    max_episode_steps=500,
)

register(
    id = 'MO-New-HalfCheetah-v2',
    entry_point = 'environments.new_half_cheetah:HalfCheetahEnv',
    max_episode_steps=500,
)

register(
    id = 'MO-HalfCheetah-v2',
    entry_point = 'environments.half_cheetah:HalfCheetahEnv',
    max_episode_steps=500,
)

register(
    id = 'MO-Walker2d-v2',
    entry_point = 'environments.walker2d:Walker2dEnv',
    max_episode_steps=500,
)

register(
    id = 'MO-Swimmer-v2',
    entry_point = 'environments.swimmer:SwimmerEnv',
    max_episode_steps=500,
)

register(
    id = 'MO-Humanoid-v2',
    entry_point = 'environments.humanoid:HumanoidEnv',
    max_episode_steps=1000,
)


register(
    id = 'MO-Humanoid-v6',
    entry_point = 'environments.humanoid_v6:HumanoidEnv',
    max_episode_steps=1000,
)

register(
    id = 'MO-Humanoid-v5',
    entry_point = 'environments.humanoid_v5:HumanoidEnv',
    max_episode_steps=1000,
)

register(
    id = 'MO-Ant-v4',
    entry_point = 'environments.ant_v4:AntEnv',
    max_episode_steps=500,
)