# -*- encoding: utf-8 -*-
#' '
#@file_name    :optimiser.py
#@description    :
#@time    :2020/02/23 16:22:45
#@author    :Cindy, xd Zhang 
#@version   :0.1
import numpy as np
class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, init_lr, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.init_lr = init_lr
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0
    def load_state_dict(self,dict):
        self._optimizer.load_state_dict(dict)
        for param_group in self._optimizer.param_groups:
            self.init_lr=param_group['lr']
            break      
    def state_dict(self):
        return self._optimizer.state_dict()
    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()
    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()


    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        chs= min(n_steps ** (-0.5), n_steps * (n_warmup_steps ** (-1.5)) )
        #n<n_warmup_steps 取 n_steps * (n_warmup_steps ** (-1.5))~3.952847075210474e-06
        #n>n_warmup_steps 取n_steps ** (-0.5)~1e-2
        dms=(d_model ** -0.5) 
        return dms*chs


    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.init_lr * self._get_lr_scale()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr