import numpy as np
import torch
from math import cos, sin, tanh
from omniisaacgymenvs.ES.hebbian_neural_net import HebbianNet
from omniisaacgymenvs.ES.rbf_neural_net import RBFNet
from omniisaacgymenvs.ES.robot_config import get_num_legjoints

class RBFHebbianNet:
    def __init__(self, 
                 popsize, 
                 num_basis, 
                 num_output, 
                 ARCHITECTURE=[47, 64, 32, 18], 
                 mode='parallel_Hebb',
                 hebb_init_wnoise=0.01,
                 hebb_norm_mode='clip',
                 robot='ant',):
        """
        sizes: [input_size, hid_1, ..., output_size]
        """
        num_legs, num_joints, motor_mapping = get_num_legjoints(robot)

        self.rbf_net = RBFNet(popsize=popsize,
                              num_basis=num_basis,
                              num_output=num_output,
                              robot=robot,
                              motor_encode='semi-indirect',
                              )

        self.hebb_net = HebbianNet(popsize=popsize,
                                sizes=ARCHITECTURE,
                                init_noise=hebb_init_wnoise,
                                norm_mode=hebb_norm_mode,
                                )
    def forward(self, pre):

        out_rbf = self.rbf_net.forward(pre)
        out_hebb = self.hebb_net.forward(pre)

        post = out_rbf + out_hebb

        return post
    
    def get_n_params_a_model(self):
        return len(self.get_a_model_params())

    def get_hebb_weights(self):
        return self.hebb_net.get_weights()

    def get_models_params(self):
        return self.hebb_net.get_models_params()

    def get_a_model_params(self):
        return self.hebb_net.get_a_model_params()
            
    def get_rbf_params(self):
        return self.rbf_net.get_models_params()

    def get_a_rbf_params(self):
        return self.rbf_net.get_a_model_params()

    def set_models_params(self, flat_params):
        self.hebb_net.set_models_params(flat_params)

    def set_a_model_params(self, flat_params):
        self.hebb_net.set_a_model_params(flat_params)

    def set_rbf_params(self, flat_params):
        self.rbf_net.set_models_params(flat_params)

    def set_a_rbf_params(self, flat_params):
        self.rbf_net.set_a_model_params(flat_params)