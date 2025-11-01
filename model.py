import torch
import torch.nn as nn
from torch.distributions import Normal
from rltorch.network import create_linear_network
import os
import copy
import random

class PCGrad():
    def __init__(self, optimizer, reduction='mean'):
        self._optim, self._reduction = optimizer, reduction
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''

        return self._optim.zero_grad()

    def step(self):
        '''
        update the parameters with the gradient
        '''

        return self._optim.step()

    def pc_backward(self, objectives):
        '''
        calculate the gradient of the parameters

        input:
        - objectives: a list of objectives
        '''

        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return

    def _project_conflicting(self, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    g_i -= (g_i_g_j) * g_j / (g_j.norm() ** 2)
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction:
            merged_grad[shared] = torch.stack([g[shared]
                                               for g in pc_grad]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared]
                                               for g in pc_grad]).sum(dim=0)
        else:
            exit('invalid reduction method')

        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in pc_grad]).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads):
        '''
        set the modified gradients to the network
        '''

        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives):
        '''
        pack the gradient of the parameters of the network for each objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad()
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network with specific
        objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad


class Conflict_caculate():
    def __init__(self, optimizer, reduction='mean'):
        self._optim, self._reduction = optimizer, reduction
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''

        return self._optim.zero_grad()

    def step(self):
        '''
        update the parameters with the gradient
        '''

        return self._optim.step()


    def get_gardients_vector(self, objectives):
        '''
               calculate the gradient of the parameters

               input:
               - objectives: a list of objectives
               '''

        grads, shapes, has_grads = self._pack_grad(objectives)

        #print("shapes,",shapes[0])
        self._optim.zero_grad()

        #reshaped_grad = self._unflatten_grad(np.array(grads), shapes[0])

        return grads, shapes


    def get_stiffness(self, grads):
        '''
        calculate the gradient of the parameters

        input:
        - objectives: a list of objectives
        '''
        #print("????",  len(grads))

        assert len(grads) == 2
        grads_0 = grads[0]
        grads_1 = grads[1]
        #pc_grad = self._project_conflicting(grads, has_grads)
        #pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        #self._set_grad(pc_grad)

        return torch.dot(grads_0, grads_1)/(grads_0.norm()*grads_1.norm())




    def _project_conflicting(self, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    g_i -= (g_i_g_j) * g_j / (g_j.norm() ** 2)
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction:
            merged_grad[shared] = torch.stack([g[shared]
                                               for g in pc_grad]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared]
                                               for g in pc_grad]).sum(dim=0)
        else:
            exit('invalid reduction method')

        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in pc_grad]).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads):
        '''
        set the modified gradients to the network
        '''

        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives):
        '''
        pack the gradient of the parameters of the network for each objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            #print("0")
            self._optim.zero_grad()
            obj.backward(retain_graph=True)
            #print("1")
            grad, shape, has_grad = self._retrieve_grad()
            #print("2")
            grads.append(self._flatten_grad(grad, shape))

            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0

        for grad in grads:
            for shape in shapes:
                length = np.prod(shape).astype(np.int32)

                #print("???", idx,length, shape)
                unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
                idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network with specific
        objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad

class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))



class QNetwork(BaseNetwork):
    def __init__(self,latent_dim, num_actions, num_weights, hidden_units=[256, 256],
                 initializer='xavier',Use_Critic_Preference=True):
        super(QNetwork, self).__init__()

        # https://github.com/ku2482/rltorch/blob/master/rltorch/network/builder.py

        self.Q = nn.Sequential(nn.Linear(latent_dim+num_actions, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, num_weights))

    def forward(self, x):
        q = self.Q(x)
        return q


    def get_fau(self, x):
        top = 0
        bottom = 0
        for i in range(len(self.Q)):
            x = self.Q[i](x)
            if isinstance(self.Q[i], nn.ReLU):
                top += (x > 0).sum().item()
                bottom += x.numel()
                # print((x1>0).sum().item(), (x2>0).sum().item(), x1.numel())
        return top / bottom


# class QNetwork(BaseNetwork):
#     def __init__(self, latent_dim, num_actions, num_weights, hidden_units=[256, 256],
#                  initializer='xavier',Use_Critic_Preference=True):
#         super(QNetwork, self).__init__()
#
#         # https://github.com/ku2482/rltorch/blob/master/rltorch/network/builder.py
#         self.Q = create_linear_network(
#             latent_dim+num_actions, num_weights, hidden_units=hidden_units,
#             initializer=initializer)
#
#
#     def forward(self, x):
#         q = self.Q(x)
#         return q


def AvgL1Norm(x, eps=1e-8):
    return x/x.abs().mean(-1,keepdim=True).clamp(min=eps)


class Latent_Encoder(BaseNetwork):  # Critic

    def __init__(self, use_avg , state_dim_with_weights, num_actions, num_weights, latent_dim):
        super(Latent_Encoder, self).__init__()

        self.use_avg = use_avg
        self.z_encoder = nn.Sequential(*[nn.Linear(state_dim_with_weights, 256), nn.ELU(),  nn.Linear(256, latent_dim)])

        self.z_dynamic_pre = nn.Sequential(nn.Linear(latent_dim + num_actions, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(),nn.Linear(256, latent_dim))

    def get_latent_features(self, states_weights):
        if self.use_avg:
            latent_features = AvgL1Norm(self.z_encoder(states_weights))
        else :
            latent_features = self.z_encoder(states_weights)

        return latent_features

    def get_dynamic(self, z, a):

        pre_next_latent_features = self.z_dynamic_pre(torch.cat([z, a], -1))

        return pre_next_latent_features




class TwinnedQNetwork(BaseNetwork):#Critic

    def __init__(self, latent_dim, num_actions, num_weights, hidden_units=[256, 256],
                 initializer='xavier',Use_Critic_Preference=True):
        super(TwinnedQNetwork, self).__init__()
        #
        #
        # self.Use_Critic_Preference = Use_Critic_Preference
        self.Q1 = QNetwork(
            latent_dim, num_actions, num_weights, hidden_units, initializer,Use_Critic_Preference )
        self.Q2 = QNetwork(
            latent_dim, num_actions, num_weights, hidden_units, initializer,Use_Critic_Preference)


    def forward(self, z, actions, preferences):

        q1 = self.Q1(torch.cat([z,actions],-1))
        q2 = self.Q2(torch.cat([z,actions],-1))

        return q1, q2


    def get_fau(self,  z, actions, preferences):
        fau_1 = self.Q1.get_fau(torch.cat([z,actions],-1))
        fau_2 = self.Q2.get_fau(torch.cat([z,actions],-1))
                # print((x1>0).sum().item(), (x2>0).sum().item(), x1.numel())
        return (fau_1 + fau_2)/2.0

    def get_representation(self, states, actions):

        x = torch.cat([states, actions], dim=1)

        representation, q = self.Q1.get_only_representation(x)

        return representation, q


from copy import deepcopy
import math
def is_lnorm_key(key):
    return key.startswith('lnorm')
def to_numpy(var):
    return var.data.numpy()
import numpy as np


class GaussianPolicy(BaseNetwork):#Policy
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20
    eps = 1e-6

    def __init__(self, num_inputs, num_actions, hidden_units=[256, 256],
                 initializer='xavier',Use_Policy_Preference =True):
        super(GaussianPolicy, self).__init__()

        # https://github.com/ku2482/rltorch/blob/master/rltorch/network/builder.py
        self.Use_Policy_Preference = Use_Policy_Preference

        self.policy = create_linear_network(
            num_inputs, num_actions*2, hidden_units=hidden_units,
            initializer=initializer)
       # print(self.Use_Policy_Preference, "??? ~~~~~~~~~~~~", num_inputs)
    def forward(self, inputs):
        x = inputs
      #  print("??? ", x.shape)
        mean, log_std = torch.chunk(self.policy(x), 2, dim=-1)
        log_std = torch.clamp(
            log_std, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)

        return mean, log_std

    def get_fau(self, x):
        top = 0
        bottom = 0
        for i in range(len(self.policy)):
            x = self.policy[i](x)
            # print(f"trunk{i}:{self.trunk[i]}\noutput shape:{x.shape}")
            # print(f"trunk{i}:{type(self.trunk[i])}\n{self.trunk[i]}\noutput shape:{x.shape}")
            # print(isinstance(self.trunk[i],nn.ReLU))
            if isinstance(self.policy[i], nn.ReLU):
                top += (x > 0).sum().item()
                bottom += x.numel()
                # print((x>0).sum().item(), x.numel())
        return top / bottom

    def sample(self, inputs):
        # calculate Gaussian distribusion of (mean, std)
        means, log_stds = self.forward(inputs)
        stds = log_stds.exp()
        normals = Normal(means, stds)
        # sample actions
        xs = normals.rsample()
        actions = torch.tanh(xs)
        # calculate entropies
        log_probs = normals.log_prob(xs)\
            - torch.log(1 - actions.pow(2) + self.eps)
        entropies = -log_probs.sum(dim=1, keepdim=True)

        return actions, entropies, torch.tanh(means)
    # function to return current pytorch gradient in same order as genome's flattened parameter vector
    def extract_grad(self):
        tot_size = self.count_parameters()
        pvec = torch.zeros(tot_size, dtype=torch.float32).to(self.args.device)
        count = 0
        for name, param in self.named_parameters():
            if is_lnorm_key(name) or len(param.shape) != 2:
                continue
            sz = param.numel()
            pvec[count:count + sz] = param.grad.view(-1)
            count += sz
        return pvec.detach().clone()

    # function to grab current flattened neural network weights
    def extract_parameters(self):
        tot_size = self.count_parameters()
        pvec = torch.zeros(tot_size, dtype=torch.float32).to(self.args.device)
        count = 0
        for name, param in self.named_parameters():
            if is_lnorm_key(name) or len(param.shape) != 2:
                continue
            sz = param.numel()
            pvec[count:count + sz] = param.view(-1)
            count += sz
        return pvec.detach().clone()

    # function to inject a flat vector of ANN parameters into the model's current neural network weights
    def inject_parameters(self, pvec):
        count = 0
        for name, param in self.named_parameters():
            if is_lnorm_key(name) or len(param.shape) != 2:
                continue
            sz = param.numel()
            raw = pvec[count:count + sz]
            reshaped = raw.view(param.size())
            param.data.copy_(reshaped.data)
            count += sz

    # count how many parameters are in the model
    def count_parameters(self):
        count = 0
        for name, param in self.named_parameters():
            if is_lnorm_key(name) or len(param.shape) != 2:
                continue
            count += param.numel()
        return count
    def set_params(self, params):
        """
        Set the params of the network to the given parameters
        """
        cpt = 0
        for param in self.parameters():
            tmp = np.product(param.size())

            # if torch.cuda.is_available():
            #     param.data.copy_(torch.from_numpy(
            #         params[cpt:cpt + tmp]).view(param.size()).cuda())
            # else:
            param.data.copy_(torch.from_numpy(
                params[cpt:cpt + tmp]).view(param.size()))
            cpt += tmp

    def get_params(self):
        """
        Returns parameters of the actor
        """
        return deepcopy(np.hstack([to_numpy(v).flatten() for v in
                                   self.parameters()]))

    def get_grads(self):
        """
        Returns the current gradient
        """
        return deepcopy(np.hstack([to_numpy(v.grad).flatten() for v in self.parameters()]))

    def get_size(self):
        """
        Returns the number of parameters of the network
        """
        return self.get_params().shape[0]
