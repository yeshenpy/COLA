import os
import visdom
import numpy as np
import torch
import copy
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from rltorch.memory import MultiStepMemory, PrioritizedMemory
from base import QMemory

from model import TwinnedQNetwork, GaussianPolicy, Latent_Encoder
from utils import grad_false, hard_update, soft_update, to_batch,\
    update_params, RunningMeanStats
import random
from multi_step import *
from datetime import datetime
import time

from hypervolume import InnerHyperVolume
from copy import deepcopy
#PREF = [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8],[0.1,0.9]]

def check_dominated(obj_batch, obj):
    return (np.logical_and(
        (obj_batch >= obj).all(axis=1),
        (obj_batch > obj).any(axis=1))
    ).any()

#
## return sorted indices of nondominated objs
def real_get_ep_indices(obj_batch_input):
   if len(obj_batch_input) == 0: return np.array([])
   obj_batch = np.array(obj_batch_input)
   sorted_indices = np.argsort(obj_batch.T[0])
   ep_indices = []
   for idx in sorted_indices:
       if (obj_batch[idx] >= 0).all() and not check_dominated(obj_batch, obj_batch[idx]):
           ep_indices.append(idx)
   return ep_indices
# return sorted indices of nondominated objs
def get_ep_indices(obj_batch_input):
    if len(obj_batch_input) == 0: return np.array([])
    obj_batch = np.array(obj_batch_input)
    sorted_indices = np.argsort(obj_batch.T[0])
    ep_indices = []
    for idx in sorted_indices:
        if not check_dominated(obj_batch, obj_batch[idx]):
            ep_indices.append(idx)
    return ep_indices
class EP:
    def __init__(self):
        self.obj_batch = np.array([])
        self.sample_batch = np.array([])

    def index(self, indices, inplace=True):
        if inplace:
            self.obj_batch, self.sample_batch = \
                map(lambda batch: batch[np.array(indices, dtype=int)], [self.obj_batch, self.sample_batch])
        else:
            return map(lambda batch: deepcopy(batch[np.array(indices, dtype=int)]), [self.obj_batch, self.sample_batch])

    def update(self, sample_batch):
        self.sample_batch = np.append(self.sample_batch, np.array(deepcopy(sample_batch)))
        
        #print(" 1  ??? ", len(self.sample_batch ))
        for sample in sample_batch:
            self.obj_batch = np.vstack([self.obj_batch, sample.objs]) if len(self.obj_batch) > 0 else np.array([sample.objs])
        #print(" 2  ??? ", len(self.obj_batch ))
        if len(self.obj_batch) == 0: return
        ep_indices = get_ep_indices(self.obj_batch)

        self.index(ep_indices)

        return len(ep_indices) > 0


class Real_EP:
    def __init__(self):
        self.obj_batch = np.array([])
        self.sample_batch = np.array([])

    def index(self, indices, inplace=True):
        if inplace:
            self.obj_batch, self.sample_batch = \
                map(lambda batch: batch[np.array(indices, dtype=int)], [self.obj_batch, self.sample_batch])
        else:
            return map(lambda batch: deepcopy(batch[np.array(indices, dtype=int)]), [self.obj_batch, self.sample_batch])

    def update(self, sample_batch):
        self.sample_batch = np.append(self.sample_batch, np.array(deepcopy(sample_batch)))

        # print(" 1  ??? ", len(self.sample_batch ))
        for sample in sample_batch:
            self.obj_batch = np.vstack([self.obj_batch, sample.objs]) if len(self.obj_batch) > 0 else np.array(
                [sample.objs])
        # print(" 2  ??? ", len(self.obj_batch ))
        if len(self.obj_batch) == 0: return
        ep_indices = real_get_ep_indices(self.obj_batch)

        self.index(ep_indices)

class Sample:
    def __init__(self, policy,optimizer, weight,  objs = None):
        self.policy = policy
        self.weight = weight
        #self.optimizer = Adam(self.policy.parameters(), lr=0.0003)
        self.objs = objs
        self.link_policy_agent(optimizer)
        #self.optgraph_id = optgraph_id

    @classmethod
    def copy_from(cls, sample):
        policy = deepcopy(sample.policy)
        objs = deepcopy(sample.objs)
        weight = deepcopy(sample.weight)
        optimizer =  deepcopy(sample.optimizer)
        
        return cls(policy, optimizer ,weight, objs)

    def link_policy_agent(self, optimizer):
        optim_state_dict = deepcopy(optimizer.state_dict())
        self.optimizer = Adam(self.policy.parameters(),  lr = 3e-4, eps = 1e-5)
        self.optimizer.load_state_dict(optim_state_dict)

def compute_hypervolume_sparsity_3d(obj_batch, ref_point):
    HV = InnerHyperVolume(ref_point)
    hv = HV.compute(obj_batch)

    sparsity = 0.0
    m = len(obj_batch[0])
    for dim in range(m):
        objs_i = np.sort(deepcopy(obj_batch.T[dim]))
        for i in range(1, len(objs_i)):
            sparsity += np.square(objs_i[i] - objs_i[i - 1])
    if len(obj_batch) == 1:
        sparsity = 0.0
    else:
        sparsity /= (len(obj_batch) - 1)

    return hv, sparsity

def evluate_Hv_UT_and_spa(obj_num, obj_batch, PREF_):

    ref_point = np.zeros([obj_num])

    hv, sparsity = compute_hypervolume_sparsity_3d(obj_batch, -np.array(ref_point))

    print("obj_batch", obj_batch, ref_point)

    UT = 0.0
    for ref in PREF_:
        res = np.max(np.sum(np.array(ref) * obj_batch, axis=-1))
        UT += res
    UT /= len(PREF_)
    return hv, sparsity, UT


p_name= ['9505','9010','8515','8020','7525','7030','6535','6040','5545','5050','4555','4060','3565','3070','2575','2080','1585','1090','0595']
PREF = [[0.95,0.05],[0.9, 0.1], [0.85, 0.15], [0.8, 0.2], [0.75, 0.25], [0.7, 0.3], [0.65, 0.35], [0.6, 0.4], [0.55, 0.35], [0.5, 0.5], [0.45, 0.55], [0.4, 0.6], [0.35, 0.65], [0.3, 0.7], [0.25, 0.75],[0.2, 0.8], [0.15,0.85] ,[0.1,0.9]]

from pymoo.factory import get_performance_indicator

def compute_hv(objs, ref_point):
    x, hv = ref_point[0], 0.0
    for i in range(len(objs)):
        hv += (max(ref_point[0], objs[i][0]) - x) * (max(ref_point[1], objs[i][1]) - ref_point[1])
        x = max(ref_point[0], objs[i][0])
    return hv

class QMonitor(object):
    def __init__(self,train=True):
        a=1
    def update(self, eps, a, b, c, d):
        a=1

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

def compute_sparsity(obj_batch):
    non_dom = NonDominatedSorting().do(obj_batch, only_non_dominated_front=True)
    objs = obj_batch[non_dom]
    sparsity_sum = 0
    for objective in range(objs.shape[-1]):
        objs_sort = np.sort(objs[:, objective])
        sp = 0
        for i in range(len(objs_sort) - 1):
            sp += np.power(objs_sort[i] - objs_sort[i + 1], 2)
        sparsity_sum += sp
    if len(objs) > 1:
        sparsity = sparsity_sum / (len(objs) - 1)
    else:
        sparsity = 0

    return sparsity


class Monitor(object):

    def __init__(self, spec,path):

        env_name = spec['env_name']
        set_num = spec['set_num']
        buf_num = spec['buf_num']
        self.vis = visdom.Visdom(env = f'MOSAC{datetime.now().strftime("%m%d")}-{env_name}_pref{set_num}_buf{buf_num}' ,port = 8097)
        self.spec = spec
        if spec['pref'][0] == 0.9:
            print(999)
            self.path = os.path.join(path,'reward_log91.npz')
        elif spec['pref'][0] == 0.5:
            print(555)
            self.path = os.path.join(path,'reward_log55.npz')
        elif spec['pref'][0] == 0.1:
            print(111)
            self.path = os.path.join(path,'reward_log19.npz')


        self.value_window = None
        self.text_window = None

    def update(self, eps, tot_reward, Rew_1, Rew_2, loss):

        if self.value_window == None:
            self.tot_t = np.array([tot_reward])
            self.rew_1_t = np.array([Rew_1])
            self.rew_2_t = np.array([Rew_2])
            self.loss_t = np.array([loss])
            self.value_window = self.vis.line(X=torch.Tensor([eps]).cpu(),
                                              Y=torch.Tensor([tot_reward, Rew_1, Rew_2, loss]).unsqueeze(0).cpu(),
                                              opts=dict(xlabel='steps_per10000',
                                                        ylabel='Reward value',
                                                        title='Value Dynamics ' + str(self.spec['pref']) + ' ' + str(self.spec['seed']),
                                                        legend=['Total Reward', 'forward_reward', 'ctrl cost','loss']))
        else:
            #Smoothing
            self.tot_t = np.append(self.tot_t, tot_reward)
            tot_reward = np.mean(self.tot_t[-20:])
            
            self.rew_1_t = np.append(self.rew_1_t, Rew_1)
            Rew_1 = np.mean(self.rew_1_t[-20:])
            
            self.rew_2_t = np.append(self.rew_2_t, Rew_2)
            Rew_2 = np.mean(self.rew_2_t[-20:])
            
            if hasattr(self, 'path'):
                np.savez(self.path,tot=self.tot_t, rew_1 = self.rew_1_t, rew_2 = self.rew_2_t)
            
            self.loss_t = np.append(self.loss_t, loss)
            loss = np.mean(self.loss_t[-20:])

            self.vis.line(
                X=torch.Tensor([eps]).cpu(),
                Y=torch.Tensor([tot_reward, Rew_1, Rew_2, loss]).unsqueeze(0).cpu(),
                win=self.value_window,
                update='append')
import itertools
def generate_w_batch_test(reward_num, step_size):
    mesh_array = []
    step_size = step_size
    for i in range(reward_num):
        mesh_array.append(np.arange(0, 1 + step_size, step_size))
    w_batch_test = np.array(list(itertools.product(*mesh_array)))
    w_batch_test = w_batch_test[w_batch_test.sum(axis=1) == 1, :]
    w_batch_test = np.unique(w_batch_test, axis=0)
    return w_batch_test
from population_2d import Population as  Population2d
from mod_neuro_evo import SSNE


def rl_to_evo(rl_agent, evo_net):
    for target_param, param in zip(evo_net.parameters(), rl_agent.parameters()):
        target_param.data.copy_(param.data)

from model import Conflict_caculate, PCGrad
import pickle
class SacAgent:

    def __init__(self, env_id, env, log_dir, num_steps=3000000, batch_size=256,
                 lr=0.0003, hidden_units=[256, 256], memory_size=1e6, prefer_num = 8, buf_num = 0,
                 gamma=0.99, tau=0.005, entropy_tuning=True, ent_coef=0.2,
                 multi_step=1, per=False, alpha=0.6, beta=0.4,
                 beta_annealing=0.0001, grad_clip=None, updates_per_step=1,
                 start_steps=10000, log_interval=10, target_update_interval=1,
                 eval_interval=1000, cuda=True, seed=0, cuda_device=0, q_frequency=1000, ref_point =[0.0,-300.0], model_saved_step=100000,Use_Policy_Preference=True, Use_Critic_Preference=True,train_with_fixed_preference=False,
                 pop_size=5,iso_sigma=0.005,line_sigma=0.05,  EA_policy_num=1, warm_steps=10000, RL_policy_num=1, latent_dim=50, reward_coef = 1.0, dynamic_coef = 1.0, value_coef = 1.0, Policy_use_latent=False, Policy_use_s=False, Policy_use_w = False,Critic_use_s = False, Critic_use_a = False, Policy_use_target=False, encoder_update_freq=1,use_avg=False, Critic_use_both=False, use_encoder_hardupdate=False, regular_alpha=0.1, Wandb_name="_", Use_pc_grad=False, step_random=False, old_Q_update_freq=1, regular_bar=0.0, consider_other=True):
        self.env_id = env_id
        self.consider_other = consider_other
        self.regular_bar = regular_bar
        self.old_Q_update_freq = old_Q_update_freq
        self.step_random =step_random
        self.Use_pc_grad = Use_pc_grad
        self.Wandb_name = Wandb_name
        self.regular_alpha = regular_alpha
        self.use_encoder_hardupdate =use_encoder_hardupdate
        self.Critic_use_both = Critic_use_both
        self.Policy_use_target = Policy_use_target
        self.encoder_update_freq = encoder_update_freq
        self.Critic_use_s = Critic_use_s
        self.Critic_use_a = Critic_use_a


        self.Policy_use_s = Policy_use_s
        self.Policy_use_w = Policy_use_w
        self.Policy_use_latent = Policy_use_latent
        self.reward_coef = reward_coef
        self.dynamic_coef = dynamic_coef
        self.value_coef = value_coef

        self.latent_dim = latent_dim
        self.iso_sigma = iso_sigma
        self.line_sigma = line_sigma
        self.EA_policy_num = EA_policy_num
        self.RL_policy_num = RL_policy_num

        self.Use_Policy_Preference = Use_Policy_Preference
        self.Use_Critic_Preference = Use_Critic_Preference
        self.ref_point = ref_point
        self.env = env
        self.previous_evluate = 0
        print("--------------------------",Use_Policy_Preference,Use_Critic_Preference,train_with_fixed_preference)
        self.train_with_fixed_preference= train_with_fixed_preference


        self.max_action = self.env.action_space.high
        self.model_saved_step = model_saved_step

        self.save_objs_freq = 0
        torch.manual_seed(seed)
        if cuda:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        self.env.seed(seed)
        self.env.action_space.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True  # It harms a performance.
        torch.backends.cudnn.benchmark = False
        self.q_frequency = q_frequency
        self.QM = QMonitor()

        self.device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")

        print(self.device)
        print(self.env.observation_space.shape[0])
        print(self.env.reward_num)

        if self.Policy_use_latent :
            input_dim = self.latent_dim
            if self.Policy_use_s:
                input_dim += self.env.observation_space.shape[0]

            if self.Policy_use_w:
                input_dim += self.env.reward_num

            self.policy = GaussianPolicy(
                input_dim,
                self.env.action_space.shape[0],
                hidden_units=[128, 128], Use_Policy_Preference=self.Use_Policy_Preference).to(self.device)

        else :
            self.policy = GaussianPolicy(
                self.env.observation_space.shape[0]+self.env.reward_num,
                self.env.action_space.shape[0],
                hidden_units=[128,128], Use_Policy_Preference=self.Use_Policy_Preference).to(self.device)


        self.latent_encoder = Latent_Encoder(use_avg, self.env.observation_space.shape[0], self.env.action_space.shape[0] , self.env.reward_num, self.latent_dim)

        self.latent_encoder_target = copy.deepcopy(self.latent_encoder)

        if self.Critic_use_both:
            input_dim = self.latent_dim*2
        else :
            input_dim = self.latent_dim


        if self.Critic_use_s:
            input_dim += self.env.observation_space.shape[0]
        if self.Critic_use_a:
            input_dim += self.env.action_space.shape[0]

        self.reward_num =   self.env.reward_num
        self.critic = TwinnedQNetwork(
            input_dim,
            self.env.reward_num,
            self.env.reward_num,
            hidden_units=hidden_units, Use_Critic_Preference=self.Use_Critic_Preference).to(self.device)
        self.critic_target = TwinnedQNetwork(
            input_dim,
            self.env.reward_num,
            self.env.reward_num,
            hidden_units=hidden_units,Use_Critic_Preference=self.Use_Critic_Preference).to(self.device).eval()

        self.old_critic = copy.deepcopy(self.critic)

        # copy parameters of the learning network to the target network
        hard_update(self.critic_target, self.critic)
        # disable gradient calculations of the target network
        grad_false(self.critic_target)

        self.policy_optim = Adam(self.policy.parameters(), lr=lr)

        self.latent_encoder_optim = Adam(self.latent_encoder.parameters(), lr=lr)
        self.q1_optim = Adam(self.critic.Q1.parameters(), lr=lr)
        self.q2_optim = Adam(self.critic.Q2.parameters(), lr=lr)

        if self.Use_pc_grad:
            self.pc_q1_optim = PCGrad(Adam(self.critic.Q1.parameters(), lr=lr))
            self.pc_q2_optim = PCGrad(Adam(self.critic.Q2.parameters(), lr=lr))

        self.conflict_caculates = Conflict_caculate(self.q1_optim)
        self.conflict_caculates_q2 = Conflict_caculate(self.q2_optim)
        self.policy_conflict_caculates = Conflict_caculate(self.policy_optim)



        if entropy_tuning:
            # Target entropy is -|A|.
            self.target_entropy = -torch.prod(torch.Tensor(
                self.env.action_space.shape).to(self.device)).item()
            # We optimize log(alpha), instead of alpha.
            self.log_alpha = torch.zeros(
                1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=lr)
        else:
            # fixed alpha
            self.alpha = torch.tensor(ent_coef).to(self.device)

        if per:
            # replay memory with prioritied experience replay
            # See https://github.com/ku2482/rltorch/blob/master/rltorch/memory
            self.memory = PrioritizedMemory(
                memory_size, self.env.observation_space.shape,
                self.env.action_space.shape, self.device, gamma, multi_step,
                alpha=alpha, beta=beta, beta_annealing=beta_annealing)
        else:

            # replay memory without prioritied experience replay
            # See https://github.com/ku2482/rltorch/blob/master/rltorch/memory
            self.memory = MOMultiStepMemory(
                memory_size, self.env.observation_space.shape, self.env.reward_num,
                self.env.action_space.shape, self.device, gamma, multi_step)



        #Q Replay Buffer
        self.Q_memory = QMemory(buf_num)
        self.cur_p = 0
        self.cur_e = 0
        self.qmem_p = 0
        self.qmem_e = 0


        self.population = Population2d(self.env.reward_num)

        self.EP = EP()
        self.Real_EP = Real_EP()
        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model')
        self.summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        
        self.monitor = []
        self.tot_t = []
        self.reward_v = []
        if self.env.reward_num == 3:
            PREF_=np.load("3pref_table.npy")
        elif self.env.reward_num == 4:
            PREF_=np.load("4pref_table.npy")
        elif self.env.reward_num == 5:
            PREF_=np.load("5pref_table.npy")
        else:
            PREF_ = PREF
        for i in PREF_:
            self.tot_t.append([])
            self.reward_v.append([])

        
        self.set_num = prefer_num # set of ω'
        self.record_fau = 0
        self.steps = 0


        self.previous_best_hv = 0
        self.previous_best_ut = 0
        self.previous_save = -100000

        self.record_steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.num_steps = num_steps
        self.tau = tau
        self.per = per
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.gamma_n = gamma ** multi_step
        self.entropy_tuning = entropy_tuning
        self.grad_clip = grad_clip
        self.updates_per_step = updates_per_step
        self.log_interval = log_interval
        self.target_update_interval = target_update_interval
        self.eval_interval = eval_interval
        self.save_loss_inter = 0
        self.rl_updated = False


        self.warm_steps = warm_steps
        self.total_count = 1e-20
        self.insert_pop = 0.0
        self.insert_ep = 0.0


    def get_pref(self):

        if self.env_id == "MO-Ant-v5":
            assert self.env.reward_num == 5
            preference = np.random.rand(2)
            preference = preference.astype(np.float32)
            preference /= preference.sum()

            second_preference = np.random.rand(4)
            second_preference = second_preference.astype(np.float32)
            second_preference /= second_preference.sum()

            preference = np.array(
                [preference[0], preference[1] * second_preference[0], preference[1] * second_preference[1],
                 preference[1] * second_preference[2], preference[1] * second_preference[3]])
        else:
            preference = np.random.rand( self.env.reward_num)
            preference = preference.astype(np.float32)
            preference /= preference.sum()
        return preference


    def run(self,our_wandb):
        if self.env.reward_num == 2:
            Preference_for_HV = generate_w_batch_test(self.env.reward_num, step_size=0.005)
        elif self.env.reward_num == 3:
            Preference_for_HV = generate_w_batch_test(self.env.reward_num, step_size=0.05)
        elif self.env.reward_num == 4:
            Preference_for_HV = generate_w_batch_test(self.env.reward_num, step_size=0.1)
        elif self.env.reward_num == 5:
            Preference_for_HV = generate_w_batch_test(self.env.reward_num, step_size=0.2)
        elif self.env.reward_num == 6:
            Preference_for_HV = generate_w_batch_test(self.env.reward_num, step_size=0.25)
        else:
            assert 1 == 2

        self.our_wandb = our_wandb
        while True:
            self.train_episode(Preference_for_HV)
            if self.steps > self.num_steps:
                break

    def is_update(self):

        return len(self.memory) > self.batch_size and\
            self.steps >= self.start_steps


    def evluate(self, preference, policy, RL_agent=False):

        episode_steps = 0
        state = self.env.reset()
        done = False
        episode_reward = 0.
        # Sample preference from prefernence space
        while not done:
            ## Just fixed
            if self.start_steps > self.steps:
                action = self.env.action_space.sample()
                next_state, reward, done, _ = self.env.step(action)
            else:
                tp_state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                tp_preference = torch.FloatTensor(preference).unsqueeze(0).to(self.device)

                if self.Policy_use_latent:

                    if self.Policy_use_target :
                        input = self.latent_encoder_target.get_latent_features(tp_state)
                    else :
                        input = self.latent_encoder.get_latent_features(tp_state)

                    if self.Policy_use_s:
                       input = torch.cat([input, tp_state],-1)
                    if self.Policy_use_w:
                        input = torch.cat([input, tp_preference], -1)
                else :
                    input = torch.cat([tp_state,tp_preference],-1)

                if RL_agent:
                    self.rl_updated = True
                    with torch.no_grad():
                        action, _, _ = policy.sample(input)
                else:
                    with torch.no_grad():
                        _, _, action = policy.sample(input)
                action =action.cpu().numpy().reshape(-1)

                # action = self.act(state)
                next_state, reward, done, _ = self.env.step(action * self.max_action)
            self.steps += 1
            episode_steps += 1
            episode_reward += reward

            # ignore done if the agent reach time horizons
            # (set done=True only when the agent fails)
            if episode_steps >= self.env.max_episode_steps:
                masked_done = False
            else:
                masked_done = done
            # We need to give true done signal with addition to masked done
            # signal to calculate multi-step rewards.


            #print(np.array(state).shape, preference, action,reward,masked_done, done )
            self.memory.append(
                state, preference, action, reward, next_state, masked_done,
                episode_done=done)
            # self.big_memory.append(
            #     state, preference, action, reward, next_state, masked_done,
            #     episode_done=done)

            state = next_state
        return episode_reward,episode_steps

    def calc_current_q(self,  s_z, states, preference, actions, rewards, next_states, dones):

        sa_z = self.latent_encoder.get_dynamic(s_z, actions).detach()
        if self.Critic_use_both:
            input = torch.cat([s_z, sa_z], -1)
        else :
            input = sa_z
        if self.Critic_use_s:
            input = torch.cat([input,states],-1)
        if self.Critic_use_a:
            input = torch.cat([input,actions], -1)

        curr_r1, curr_r2= self.critic.forward(input, preference, preference)

        return curr_r1, curr_r2

    def calc_current_q_for_analyze(self,   s_z, states, preference, actions, rewards, next_states, dones):

        sa_z = self.latent_encoder.get_dynamic(s_z, actions).detach()
        if self.Critic_use_both:
            input = torch.cat([s_z, sa_z], -1)
        else :
            input = sa_z
        if self.Critic_use_s:
            input = torch.cat([input,states],-1)
        if self.Critic_use_a:
            input = torch.cat([input,actions], -1)

        curr_r1, curr_r2= self.critic.forward(input, preference, preference)



        return curr_r1, curr_r2


    def calc_target_q(self, states, preference, actions, rewards, next_states, dones):
        with torch.no_grad():

            next_s_z = self.latent_encoder_target.get_latent_features(next_states).detach()
            if self.Policy_use_latent:

                input = next_s_z
                if self.Policy_use_s:
                    input = torch.cat([input, next_states],-1)
                if self.Policy_use_w:
                    input = torch.cat([input, preference],-1)
                next_actions, next_entropies, _ = self.policy.sample(input)
            else :
                next_actions, next_entropies, _ = self.policy.sample(torch.cat([next_states, preference],-1))

            next_sa_z = self.latent_encoder_target.get_dynamic(next_s_z, next_actions).detach()

            if self.Critic_use_both:
                critic_input = torch.cat([next_s_z, next_sa_z], -1)
            else:
                critic_input = next_sa_z

            if self.Critic_use_s:
                critic_input = torch.cat([critic_input, next_states], -1)
            if self.Critic_use_a:
                critic_input = torch.cat([critic_input, next_actions], -1)

            next_q1, next_q2 = self.critic_target(critic_input, preference, preference)
            
            #We choose argmin_Q (ωTQ)
            w_q1 = torch.einsum('ij,j->i',[next_q1, preference[0] ])
            w_q2 = torch.einsum('ij,j->i',[next_q2, preference[0] ])
            mask = torch.lt(w_q1,w_q2)
            mask = mask.repeat([1,self.env.reward_num])
            mask = torch.reshape(mask, next_q1.shape)

            minq = torch.where( mask, next_q1, next_q2)
                
            next_q = minq + self.alpha * next_entropies

        target_q = rewards + (1.0 - dones) * self.gamma_n * next_q

        return target_q


    def get_similarity(self, representation_a, representation_b):
        '''
        calculate the gradient of the parameters

        input:
        - objectives: a list of objectives
        '''
        # print("????",  len(grads))
        # pc_grad = self._project_conflicting(grads, has_grads)
        # pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        # self._set_grad(pc_grad)

        res = 0.0
        res2 = 0.0
        for _ in range(representation_a.shape[0]):

            dot_product = torch.dot(representation_a[_], representation_b[_])
            res +=dot_product / (representation_a[_].norm() * representation_b[_].norm())
            res2 +=dot_product
        res  /= representation_a.shape[0]
        res2 /= representation_a.shape[0]
        return res.cpu().data.numpy(), res2.cpu().data.numpy()




    def train_episode(self,Preference_for_HV):


        start = time.time()
        self.episodes += 1
        episode_steps = 0

        if self.env.reward_num == 3:
            PREF_= Preference_for_HV
        elif self.env.reward_num == 4:
            PREF_= Preference_for_HV
        elif self.env.reward_num == 5:
            PREF_= Preference_for_HV
        elif self.env.reward_num == 6:
            PREF_= Preference_for_HV
        else:
            PREF_ = PREF

        preference = self.get_pref()
        rl_reward, rl_ep_length = self.evluate(preference, self.policy, RL_agent=True)
        episode_steps +=rl_ep_length


        sample = Sample(policy=deepcopy(self.policy),optimizer= deepcopy(self.policy_optim), weight= preference,objs=rl_reward)
        self.population.update([sample])
        self.EP.update([sample])
        self.Real_EP.update([sample])

#        if self.steps - self.previous_save >= 50000:
#            if not os.path.exists("./logs/" + self.Wandb_name):
#                os.makedirs("./logs/" + self.Wandb_name)
#
#            self.previous_save = self.steps
#            torch.save(self.policy.state_dict(), "./logs/" + self.Wandb_name + "/policy_" + str(int(self.steps/50000))+".pkl")
#            torch.save(self.latent_encoder.state_dict(), "./logs/" + self.Wandb_name + "/encoder_"+str(int(self.steps/50000))+".pkl")
#            torch.save(self.critic.state_dict(), "./logs/" + self.Wandb_name + "/critic"+str(int(self.steps/50000))+".pkl")
#            torch.save(self.critic_target.state_dict(), "./logs/" + self.Wandb_name + "/critic_target_"+str(int(self.steps/50000))+".pkl")
#            torch.save(self.latent_encoder_target.state_dict(), "./logs/" + self.Wandb_name + "/encoder_target_" + str(int(self.steps /50000)) + ".pkl")
            # with open("./logs/" + self.Wandb_name + "champion_buffer.pkl", 'wb+') as buffer_file:
            #     pickle.dump(self.big_memory, buffer_file)


        if self.steps - self.previous_evluate >= self.eval_interval:
            self.previous_evluate = self.steps

            hv, sparsity, UT = evluate_Hv_UT_and_spa(self.env.reward_num, self.EP.obj_batch, PREF_)
            self.our_wandb.log(
                {'Ep_hypervolume': hv, 'Ep_sparsity': sparsity, 'Ep_UT': UT, 'time_steps': self.steps})
            if len(self.Real_EP.obj_batch) > 0:
                hv, sparsity, UT = evluate_Hv_UT_and_spa(self.env.reward_num, self.Real_EP.obj_batch, PREF_)
                self.our_wandb.log(
                {'Real_Ep_hypervolume': hv, 'Real_Ep_sparsity': sparsity, 'Real_Ep_UT': UT, 'time_steps': self.steps})

            objs = self.get_objs(Preference_for_HV)
            np.save(os.path.join(self.summary_dir, 'objs_'+str(self.steps)), objs)
            hv, sparsity, UT = evluate_Hv_UT_and_spa(self.env.reward_num, objs, PREF_)
            insert_pop_rate = self.insert_pop / self.total_count
            insert_ep_rate = self.insert_ep / self.total_count

            self.our_wandb.log(
                {'EA_insert_pop_rate':insert_pop_rate, 'EA_insert_ep_rate':insert_ep_rate, 'Population_num': len(self.population.sample_batch), 'Ep_num': len(self.EP.sample_batch),
                 'hypervolume': hv, 'sparsity': sparsity, 'UT': UT, 'time_steps': self.steps})


            for num in range(self.env.reward_num):
                self.our_wandb.log({'lowest_' + str(num): np.min(objs[:, num]), 'time_steps': self.steps})
            for num in range(self.env.reward_num):
                self.our_wandb.log({'max_' + str(num): np.max(objs[:, num]), 'time_steps': self.steps})

            #    self.evaluate_(PREF_[i],i)
            if self.steps - self.save_objs_freq > 50000:
                self.save_objs_freq = self.steps
                list_as_string = str(self.steps) + ": "+ str(objs.tolist())
                with open(self.Wandb_name + "_obj_list.txt", "a") as file:
                    file.write(list_as_string + '\n\n')
                self.our_wandb.save(self.Wandb_name + "_obj_list.txt")
                list_as_string =str(self.steps) + ": " + str(self.EP.obj_batch.tolist())
                with open(self.Wandb_name + "_ep_obj_list.txt", "a") as file:
                    file.write(list_as_string + '\n\n')
                self.our_wandb.save(self.Wandb_name + "_ep_obj_list.txt")


        if self.is_update() and self.rl_updated:
            fixed_preference = self.get_pref()
            other_fixed_preference = self.get_pref()
            for _ in range(self.updates_per_step*episode_steps):
                self.learn(fixed_preference, other_fixed_preference)

        if self.steps > self.num_steps:
            hv, sparsity, UT = evluate_Hv_UT_and_spa(self.env.reward_num, self.EP.obj_batch, PREF_)
            self.our_wandb.log(
                {'Ep_hypervolume': hv, 'Ep_sparsity': sparsity, 'Ep_UT': UT, 'time_steps': self.steps})
            if len(self.Real_EP.obj_batch) > 0:
                hv, sparsity, UT = evluate_Hv_UT_and_spa(self.env.reward_num, self.Real_EP.obj_batch, PREF_)
                self.our_wandb.log(
                    {'Real_Ep_hypervolume': hv, 'Real_Ep_sparsity': sparsity, 'Real_Ep_UT': UT,
                     'time_steps': self.steps})

            objs = self.get_objs(Preference_for_HV)
            np.save(os.path.join(self.summary_dir, 'objs_' + str(self.steps)), objs)

            hv, sparsity, UT = evluate_Hv_UT_and_spa(self.env.reward_num, objs, PREF_)
            self.our_wandb.log({'Population_num':  len(self.population.sample_batch), 'Ep_num': len(self.EP.sample_batch), 'hypervolume': hv, 'sparsity': sparsity, 'UT': UT, 'time_steps': self.steps})
        
        print(f'episode: {self.episodes:<4}  '
              f'episode steps: {episode_steps:<4}  '
              f'episode weight: {preference}  '
              f'rl reward:', rl_reward, " cost avg time", (time.time() - start)/episode_steps )

    def learn(self,fixed_preference, other_fixed_preference):
        self.learning_steps += 1
        if self.learning_steps % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        temp_critic = copy.deepcopy(self.critic)
        if self.use_encoder_hardupdate :
            hard_update(self.latent_encoder_target, self.latent_encoder)
        else :
            if self.learning_steps % self.encoder_update_freq == 0:
                soft_update(self.latent_encoder_target, self.latent_encoder, self.tau)

        if self.learning_steps % self.q_frequency == 0 and self.learning_steps > 20000:
            co = copy.deepcopy(self.critic)
            self.Q_memory.append(co)
        
        if self.per:
            # batch with indices and priority weights
            batch, indices, weights = \
                self.memory.sample(self.batch_size)
        else:
            batch = self.memory.sample(self.batch_size)
            # set priority weights to 1 when we don't use PER.
            weights = 1.

        PREF_SET = []
        # Form preference set W containing the updating preference
        if self.train_with_fixed_preference:
            preference = fixed_preference
        else :
            preference = self.get_pref()

        best_policy = None
        best_weight_reward = -1e9
        best_policy_weight = None
        for sample in self.EP.sample_batch:
            if np.dot(sample.objs, preference) > best_weight_reward:
                best_weight_reward = np.dot(sample.objs, preference)
                best_policy = sample.policy
                best_policy_weight = sample.weight


        preference = torch.tensor(preference, device=self.device)

        if self.step_random:
            random_preference = self.get_pref()
            random_preference = torch.tensor(random_preference, device=self.device)
        else :
            random_preference = torch.tensor(other_fixed_preference, device=self.device)

        PREF_SET.append(preference)
        for _ in range(self.set_num-1):
            p = self.get_pref()
            p = torch.tensor(p ,device = self.device)
            PREF_SET.append(p)

        states, _, actions, rewards, next_states, dones = batch


        current_latent = self.latent_encoder.get_latent_features(states)
        next_latent_target = self.latent_encoder_target.get_latent_features(next_states).detach()

        pre_next_latent = self.latent_encoder.get_dynamic(current_latent, actions)

        dynamic_loss = torch.mean((pre_next_latent - next_latent_target).pow(2))

        self.latent_encoder_optim.zero_grad()
        dynamic_loss.backward()
        self.latent_encoder_optim.step()
        new_latent = self.latent_encoder.get_latent_features(states).detach()

        q1_loss, q2_loss, errors, mean_q1, mean_q2 = self.calc_critic_loss( new_latent ,batch, weights, preference, PREF_SET,best_policy ,best_policy_weight)

        if self.Use_pc_grad:
            second_q1_loss, second_q2_loss, _, _, _ = self.calc_critic_loss( new_latent, batch, weights, random_preference,PREF_SET, best_policy,best_policy_weight)
            self.pc_q1_optim.zero_grad()
            self.pc_q2_optim.zero_grad()
            self.pc_q1_optim.pc_backward([q1_loss, second_q1_loss])
            self.pc_q2_optim.pc_backward([q2_loss, second_q2_loss])
        else :
            total_loss = self.value_coef * (q1_loss + q2_loss)
            self.q1_optim.zero_grad()
            self.q2_optim.zero_grad()
            total_loss.backward()
            self.q2_optim.step()
            self.q1_optim.step()

        policy_loss, entropies = self.calc_policy_loss(new_latent.detach(), batch, weights, preference, PREF_SET)

        if self.steps - self.save_loss_inter > 10000:
            self.save_loss_inter = self.steps
            self.our_wandb.log({'dynamic_loss': dynamic_loss.cpu().data.numpy(),'Q_loss': q1_loss.cpu().data.numpy() ,'policy_loss':policy_loss.cpu().data.numpy(), 'time_steps':self.steps})

        # update_params(
        #     self.q1_optim, self.critic.Q1, q1_loss, self.grad_clip)
        # update_params(
        #     self.q2_optim, self.critic.Q2, q2_loss, self.grad_clip)
        update_params(
            self.policy_optim, self.policy, policy_loss, self.grad_clip)

        if self.entropy_tuning:
            entropy_loss = self.calc_entropy_loss(entropies, weights)
            update_params(self.alpha_optim, None, entropy_loss)
            self.alpha = self.log_alpha.exp()
        if self.per:
            # update priority weights
            self.memory.update_priority(indices, errors.cpu().numpy())

        if self.learning_steps % self.old_Q_update_freq==0:
            self.old_critic.load_state_dict(temp_critic.state_dict())

        self.QM.update(self.steps, self.cur_p, self.cur_e, self.qmem_p, self.qmem_e)

    def calc_old_q(self,  s_z, states, preference, actions, rewards, next_states, dones):

        sa_z = self.latent_encoder.get_dynamic(s_z, actions).detach()
        if self.Critic_use_both:
            input = torch.cat([s_z, sa_z], -1)
        else :
            input = sa_z
        if self.Critic_use_s:
            input = torch.cat([input,states],-1)
        if self.Critic_use_a:
            input = torch.cat([input,actions], -1)

        #curr_q1, curr_q2 = self.critic(input, preference, preference)

        curr_r1, curr_r2= self.old_critic.forward(input, preference, preference)

        return curr_r1, curr_r2

    def calc_critic_loss(self , s_z, batch, weights, preference, PREF, best_policy, best_policy_weight):


        states, _, actions, rewards, next_states, dones = batch

        D_pref = preference.repeat(self.batch_size,1)


        curr_q1, curr_q2= self.calc_current_q( s_z, states, D_pref, actions, rewards, next_states, dones)

        target_q = self.calc_target_q(states, D_pref, actions, rewards, next_states, dones)
        # TD errors for updating priority weights
        errors = torch.abs(curr_q1.detach() - target_q)
        # We log means of Q to monitor training.
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        # Critic loss is mean squared TD errors with priority weights.
        q1_loss = torch.mean(torch.tensordot((curr_q1 - target_q).pow(2), preference,dims=1) * weights)
        q2_loss = torch.mean(torch.tensordot((curr_q2 - target_q).pow(2), preference,dims=1) * weights)


        sampled_preference_without_repeat = torch.tensor(self.get_pref(), device=self.device)
        sampled_preference = torch.tensor(sampled_preference_without_repeat, device=self.device).repeat(states.shape[0],
                                                                                                        1)
        sampled_curr_q1, sampled_curr_q2 = self.calc_current_q(s_z, states, sampled_preference, actions, rewards,
                                                               next_states, dones)

        sampled_target_q = self.calc_target_q(states, sampled_preference, actions, rewards, next_states, dones)

        if self.consider_other:
            sampled_old_q1, sampled_old_q2 = self.calc_old_q(s_z, states, sampled_preference, actions, rewards,next_states,dones)
            regular_q1 = torch.mean((sampled_curr_q1 - sampled_old_q1.detach()).pow(2))
            regular_q2 = torch.mean((sampled_curr_q2 - sampled_old_q2.detach()).pow(2))
        else :
            curr_old_q1, curr_old_q2 = self.calc_old_q(s_z, states, D_pref, actions, rewards,next_states, dones)
            regular_q1 = torch.mean((curr_q1 - curr_old_q1.detach()).pow(2))
            regular_q2 = torch.mean((curr_q2 - curr_old_q2.detach()).pow(2))

        sampled_td_error_1 = torch.mean(torch.tensordot((sampled_curr_q1 - sampled_target_q).pow(2), sampled_preference_without_repeat,dims=1) * weights)
        sampled_td_error_2 = torch.mean(torch.tensordot((sampled_curr_q2 - sampled_target_q).pow(2), sampled_preference_without_repeat,dims=1) * weights)

        grads_Q1, shapes = self.conflict_caculates.get_gardients_vector([sampled_td_error_1, q1_loss])
        shapes = shapes[0]
        the_third_start = shapes[0][0] * shapes[0][1] + shapes[1][0] + shapes[2][0] * shapes[2][1] + shapes[3][0]
        the_third_end = shapes[0][0] * shapes[0][1] + shapes[1][0] + shapes[2][0] * shapes[2][1] + shapes[3][0] + shapes[4][0] * shapes[4][1] + shapes[5][0]

        grads_Q2, _ = self.conflict_caculates_q2.get_gardients_vector([sampled_td_error_2, q2_loss])

        stiffness_Q1 = self.conflict_caculates.get_stiffness([grads_Q1[0][the_third_start: the_third_end], grads_Q1[1][the_third_start: the_third_end]])
        stiffness_Q2 = self.conflict_caculates_q2.get_stiffness([grads_Q2[0][the_third_start: the_third_end], grads_Q2[1][the_third_start: the_third_end]])


        q1_total_loss = q1_loss + max(self.regular_bar - stiffness_Q1, 0.0) * self.regular_alpha * regular_q1
        q2_total_loss = q2_loss + max(self.regular_bar - stiffness_Q2, 0.0) * self.regular_alpha * regular_q2

        return q1_total_loss, q2_total_loss, errors, mean_q1, mean_q2


    def calc_critic_loss_for_amalyze(self, s_z, batch, weights, preference):

        states, _, actions, rewards, next_states, dones = batch

        D_pref = preference.repeat(s_z.shape[0], 1)


        curr_q1, curr_q2 = self.calc_current_q_for_analyze( s_z, states, D_pref, actions, rewards, next_states, dones)

        target_q = self.calc_target_q(states, D_pref, actions, rewards, next_states, dones)

        q1_loss = torch.mean(torch.tensordot((curr_q1 - target_q).pow(2), preference, dims=1) * weights)
        q2_loss = torch.mean(torch.tensordot((curr_q2 - target_q).pow(2), preference, dims=1) * weights)

        q1_total_loss = q1_loss
        q2_total_loss = q2_loss

        return q1_total_loss, q2_total_loss, curr_q1, curr_q2




    def calc_policy_loss(self, current_latent,  batch, weights, preference, PREF):
        states, _, actions, rewards, next_states, dones = batch
        preference_batch = preference.repeat(self.batch_size, 1)
        
        losses = []

        for a, c in enumerate([ self.critic]+self.Q_memory.sample() ): # Use critic from Q Replay Buffer
            for b, i in enumerate(PREF): #Get Q from preference set W
                p_batch = torch.tensor(i, device = self.device).repeat(self.batch_size, 1)

                if self.Policy_use_latent:
                    input = current_latent
                    if self.Policy_use_s:
                        input = torch.cat([input, states], -1)
                    if self.Policy_use_w:
                        input = torch.cat([input, p_batch], -1)
                    sampled_action, entropy, _ = self.policy.sample(input)
                else :
                    sampled_action, entropy, _ = self.policy.sample(torch.cat([states, p_batch],-1))
                if a == 0 and b == 0:
                    e = entropy


                sa_z = self.latent_encoder.get_dynamic(current_latent, sampled_action)
                if self.Critic_use_both:
                    critic_input = torch.cat([current_latent, sa_z], -1)
                else:
                    critic_input = sa_z
                if self.Critic_use_s:
                    critic_input = torch.cat([critic_input, states], -1)
                if self.Critic_use_a:
                    critic_input = torch.cat([critic_input, sampled_action], -1)

                q1, q2 = c(critic_input, preference_batch, preference_batch)
                
                q1 = torch.tensordot(q1, preference, dims = 1)
                q2 = torch.tensordot(q2, preference, dims = 1)
                q = torch.min(q1, q2)
                
                l = - q - self.alpha * entropy
                losses.append(l)

        losses = torch.stack(losses, dim = 1)
        policy_loss, idx =  torch.min(losses, 1)
        ll=idx.detach().cpu()[:,0].tolist()
        policy_loss = torch.mean(policy_loss)

        if self.Policy_use_latent:
            input = current_latent
            if self.Policy_use_s:
                input = torch.cat([input, states], -1)
            if self.Policy_use_w:
                input = torch.cat([input, preference_batch], -1)
            sampled_action, e, _ = self.policy.sample(input)
        else:
            sampled_action, e, _ = self.policy.sample(torch.cat([states, preference_batch], -1))

        return policy_loss, e

    def calc_entropy_loss(self, entropy, weights):
        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropy).detach()
            * weights)
        return entropy_loss

    def exploit(self, state, preference):
        # act without randomness

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        preference = torch.FloatTensor(preference).unsqueeze(0).to(self.device)

        if self.Policy_use_latent:

            if self.Policy_use_target:
                input = self.latent_encoder_target.get_latent_features(state)
            else:
                input = self.latent_encoder.get_latent_features(state)

            if self.Policy_use_s:
                input = torch.cat([input, state], -1)
            if self.Policy_use_w:
                input = torch.cat([input, preference], -1)
        else:
            input = torch.cat([state, preference], -1)

        with torch.no_grad():
            _, _, action = self.policy.sample(input)
        return action.cpu().numpy().reshape(-1)


    def get_objs(self, Preference_for_HV, eval_episodes=3):
        hypervolume, sparsity = np.zeros((eval_episodes,)), np.zeros((eval_episodes,))
        recovered_objs =  np.zeros((len(Preference_for_HV), self.env.reward_num))
        for eval_ep in range(eval_episodes):

            # Evaluate agent for the preferences in w_batch
            for p_index, evalPreference in enumerate(Preference_for_HV):
                eval_Pre = evalPreference
                state = self.env.reset()
                terminal = False
                tot_rewards = 0
                while not terminal:
                    action = self.exploit(state, evalPreference)
                    next_state, reward, terminal, _ = self.env.step(action*self.max_action)
                    tot_rewards += reward
                    state = next_state
                recovered_objs[p_index] +=tot_rewards
        recovered_objs /=eval_episodes

        return recovered_objs

    def __del__(self):
        #self.writer.close()
        self.env.close()
