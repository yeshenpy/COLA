import os
import torch
import wandb
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
cpu_num = 1
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
os.environ["WANDB_API_KEY"] = "b26af16f098a021f99a7de6c84df44c8a5728033"
#os.environ["WANDB_MODE"] = "offline"
torch.set_num_threads(cpu_num)
import argparse
from datetime import datetime
import gym
import numpy as np
import random

from environments import *

from agent import SacAgent
def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='dst_d-v0')
    parser.add_argument('--cuda', action='store_true', default=False)

    parser.add_argument('--Use_Policy_Preference', action='store_true', default=False)
    parser.add_argument('--Use_Critic_Preference', action='store_true', default=False)
    parser.add_argument('--Policy_use_latent', action='store_true', default=False)
    parser.add_argument('--Policy_use_s', action='store_true', default=False)
    parser.add_argument('--Policy_use_w', action='store_true', default=False)
    parser.add_argument('--Critic_use_s', action='store_true', default=False)
    parser.add_argument('--Critic_use_a', action='store_true', default=False)
    parser.add_argument('--Use_pc_grad', action='store_true', default=False)

    parser.add_argument('--Critic_use_both', action='store_true', default=False)

    parser.add_argument('--step_random', action='store_true', default=False)

    parser.add_argument('--use_encoder_hardupdate', action='store_true', default=False)

    parser.add_argument('--Policy_use_target', action='store_true', default=False)

    parser.add_argument('--train_with_fixed_preference', action='store_true', default=False)

    parser.add_argument('--encoder_update_freq', type=int, default=1)

    parser.add_argument('--pop_size', type=int, default=4)
    parser.add_argument('--cuda_device', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--prefer', type=int, default=0)
    parser.add_argument('--buf_num', type=int, default=0)
    parser.add_argument('--q_freq', type=int, default=1000)
    parser.add_argument('--ref_point', type=float, nargs='+', default=[0., 0.])
    parser.add_argument('--ent_coef', type=float,default=0.2)
    parser.add_argument('--gamma', type=float, default=0.99)

    parser.add_argument('--iso_sigma', type=float, default=0.01)
    parser.add_argument('--line_sigma', type=float, default=0.2)
    parser.add_argument('--EA_policy_num', type=int, default=1)
    parser.add_argument('--RL_policy_num', type=int, default=1)
    parser.add_argument('--warm_steps', type=int, default=8000000)

    parser.add_argument('--latent_dim', type=int, default=50)

    parser.add_argument('--regular_alpha', type=float, default=0.1)
    parser.add_argument('--reward_coef', type=float, default=0.2)
    parser.add_argument('--dynamic_coef', type=float, default=0.2)
    parser.add_argument('--value_coef', type=float, default=0.2)

    parser.add_argument('--use_avg', action='store_true', default=False)

    parser.add_argument('--old_Q_update_freq', type=int, default=1)

    parser.add_argument('--regular_bar', type=float, default=0.2)

    parser.add_argument('--consider_other', action='store_true', default=False)

    args = parser.parse_args()

    name = "COLA_Consider_other_"+str(args.consider_other)+"_Regular_bar_"+ str(args.regular_bar)+"_Freq_"+str(args.old_Q_update_freq)+"_Regular_" + str(args.regular_alpha) + "_Critic_infos_"  + str(args.Critic_use_both) + "_"+str(args.use_avg)+"_"+str(args.Critic_use_s) + "_"+ str(args.Critic_use_a)+"_"+str(args.Policy_use_latent)+str(args.Policy_use_s)+"_"+str(args.Policy_use_w)+ "_Env_" + str(args.env_id)

    # You can define configs in the external json or yaml file.
    configs = {
        'num_steps': 8000000,
        'batch_size': 256,#256
        'lr': 0.0003,
        'hidden_units': [256, 256],
        'memory_size': 1e6,
        'prefer_num': args.prefer,
        'buf_num': args.buf_num,
        'gamma': args.gamma,
        'tau': 0.005,
        'entropy_tuning': True,
        'ent_coef': args.ent_coef, #0.2,  # It's ignored when entropy_tuning=True.
        'multi_step': 1,
        'per': False,  # prioritized experience replay
        'alpha': 0.6,  # It's ignored when per=False.
        'beta': 0.4,  # It's ignored when per=False.
        'beta_annealing': 0.0001,  # It's ignored when per=False.
        'grad_clip': None,
        'updates_per_step': 1,
        'start_steps': 10000,
        'log_interval': 10,
        'target_update_interval': 1,
        'eval_interval': 50000,
        'cuda': args.cuda,
        'seed': args.seed,
        'cuda_device': args.cuda_device,
        'q_frequency': args.q_freq,
        'model_saved_step': 100000,
        'Use_Policy_Preference': args.Use_Policy_Preference,
        'Use_Critic_Preference': args.Use_Critic_Preference,
        'train_with_fixed_preference': args.train_with_fixed_preference,
        'iso_sigma':args.iso_sigma,
        'line_sigma':args.line_sigma,
        'EA_policy_num': args.EA_policy_num,
        'warm_steps':args.warm_steps,
        'RL_policy_num': args.RL_policy_num,
        'latent_dim': args.latent_dim,
        'reward_coef': args.reward_coef,
        'dynamic_coef': args.dynamic_coef,
        'value_coef': args.value_coef,
        'Policy_use_latent': args.Policy_use_latent,
        'Policy_use_s': args.Policy_use_s,
        'Policy_use_w': args.Policy_use_w,
        'Critic_use_s': args.Critic_use_s,
        'Critic_use_a': args.Critic_use_a,
        'Policy_use_target': args.Policy_use_target,
        'encoder_update_freq': args.encoder_update_freq,
        'use_avg': args.use_avg,
        'Critic_use_both': args.Critic_use_both,
        'use_encoder_hardupdate': args.use_encoder_hardupdate,
        'regular_alpha': args.regular_alpha,
        'Wandb_name': name + "_" + str(args.seed),
        'Use_pc_grad': args.Use_pc_grad,
        'step_random': args.step_random,
        'old_Q_update_freq': args.old_Q_update_freq,
        'regular_bar': args.regular_bar,
        'consider_other': args.consider_other
    }

    env = gym.make(args.env_id)
    configs['ref_point'] = [0.0,0.0]

    log_dir = os.path.join(
        'logs', args.env_id,
        f'MOSAC_with_Ep_recorder-set{args.prefer}-buf{args.buf_num}-seed{args.seed}_freq{args.q_freq}')
    our_wandb = wandb.init(project="COLA",name=name)
    agent = SacAgent(env_id=args.env_id, env=env, log_dir=log_dir, **configs)
    agent.run(our_wandb)


if __name__ == '__main__':
    run()
