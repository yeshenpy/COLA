# HalfCheetah-v2 env
# two objectives
# running speed, energy efficiency

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from os import path

class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.obj_dim = 5
        self.reward_num = 5
        self.max_episode_steps = 500
        self._max_episode_steps = 500
        self.steps = 0
        mujoco_env.MujocoEnv.__init__(self, model_path = path.join(path.abspath(path.dirname(__file__)), "assets/half_cheetah.xml"), frame_skip = 5)
        utils.EzPickle.__init__(self)


    def step(self, action):
        self.steps +=1
        xposbefore = self.sim.data.qpos[0]


        action = np.clip(action, -1.0, 1.0)
        self.do_simulation(action, self.frame_skip)
        ang =  self.sim.data.qpos[2]

        xposafter = self.sim.data.qpos[0]
        height = self.sim.data.qpos[1]
       # print("height", len(self.sim.data.qpos),height,  12. * (height - self.init_qpos[1]), self.init_qpos[1])
        ob = self._get_obs()
        alive_bonus = 1.0

        reward_run_x = 0.5*(xposafter - xposbefore)/self.dt +alive_bonus
        reward_jump = 20 * (height - self.init_qpos[1]) + alive_bonus
        reward_energy_1 = 4.0 - 3.0 * np.square([action[0], action[3]]).mean() + alive_bonus
        reward_energy_2 = 4.0 - 3.0 * np.square(action[1:3]).mean() + alive_bonus
        reward_energy_3 = 4.0 - 3.0 * np.square(action[3:5]).mean() + alive_bonus
        done = not (abs(ang) < np.deg2rad(50))
        # if self.steps > self._max_episode_steps:
        #     done = True
        return ob, np.array([reward_run_x, reward_jump, reward_energy_1, reward_energy_2, reward_energy_3]), done, {'obj': np.array([reward_run_x, reward_jump, reward_energy_1, reward_energy_2, reward_energy_3])}
    
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])
    
    def reset_model(self):
        self.steps = 0
        self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()
    
    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
