# Swimmer-v2 env
# two objectives
# forward speed, energy efficiency

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from os import path

class SwimmerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.obj_dim = 2
        self.reward_num = 2
        self.max_episode_steps = 500
        self._max_episode_steps = 500
        self.steps = 0
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, model_path = path.join(path.abspath(path.dirname(__file__)), "assets/swimmer.xml"), frame_skip = 4)



    def step(self, a):
        done = False
        self.steps +=1
        ctrl_cost_coeff = 0.15
        xposbefore = self.sim.data.qpos[0]
        a = np.clip(a, -1, 1)
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = 0.3 - ctrl_cost_coeff * np.square(a).sum()
        ob = self._get_obs()
        # if self.steps > self._max_episode_steps:
        #     done = True
        return ob, np.array([reward_fwd, reward_ctrl]), done, {'obj': np.array([reward_fwd, reward_ctrl])}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat[2:], qvel.flat])

    def reset_model(self):
        self.steps = 0
        self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()
