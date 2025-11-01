import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils
from os import path

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

class HumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.obj_dim = 5
        self.reward_num = 5
        self.max_episode_steps = 1000
        self._max_episode_steps = 1000
        self.steps = 0

        mujoco_env.MujocoEnv.__init__(self, model_path = path.join(path.abspath(path.dirname(__file__)), "assets/humanoid.xml"), frame_skip = 5)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

    def step(self, a):
        self.steps +=1
        pos_before = mass_center(self.model, self.sim)
        self.do_simulation(a, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)
        ob = self._get_obs()
        data = self.sim.data
        
        alive_bonus = 3.0
        reward_run = (pos_after - pos_before) / self.dt

        reward_energy_0 = (3.0 - 4.0 * np.square(data.ctrl[0:3]).mean() * 17 + alive_bonus)/3  + reward_run
        reward_energy_1 = (3.0 - 4.0 * np.square(data.ctrl[3:7]).mean() * 17 + alive_bonus)/3  + reward_run
        reward_energy_2 = (3.0 - 4.0 * np.square(data.ctrl[7:11]).mean() * 17 + alive_bonus)/3 + reward_run
        reward_energy_3 = (3.0 - 4.0 * np.square(data.ctrl[11:14]).mean() * 17 + alive_bonus)/3 + reward_run
        reward_energy_4 = (3.0 - 4.0 * np.square(data.ctrl[14:17]).mean() * 17 + alive_bonus)/3 + reward_run

        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        # if self.steps > self._max_episode_steps:
        #     done = True
        return ob,  np.array([reward_energy_0, reward_energy_1, reward_energy_2, reward_energy_3 , reward_energy_4]), done, {'obj': np.array([reward_energy_0, reward_energy_1, reward_energy_2, reward_energy_3 , reward_energy_4]), 'speed': reward_run }

    def reset_model(self):
        self.steps = 0
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20
