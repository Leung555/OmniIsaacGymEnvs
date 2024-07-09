import numpy as np
import torch
from math import cos, sin, tanh
from omniisaacgymenvs.ES.feedforward_neural_net_gpu import FeedForwardNet
from omniisaacgymenvs.ES.rbf_neural_net import RBFNet
from omniisaacgymenvs.ES.robot_config import get_num_legjoints

class RBFFFNet:
    def __init__(self, 
                 popsize, 
                 num_basis, 
                 num_output, 
                 ARCHITECTURE,
                 robot='ant',):
        """
        sizes: [input_size, hid_1, ..., output_size]
        """
        self.architecture = ARCHITECTURE
        
        num_legs, num_joints, motor_mapping = get_num_legjoints(robot)

        self.rbf_net = RBFNet(popsize=popsize,
                              num_basis=num_basis,
                              num_output=num_output,
                              robot=robot,
                              motor_encode='semi-indirect',
                              )

        self.ff_net = FeedForwardNet(popsize=popsize,
                                    sizes=ARCHITECTURE,
                                    )
    def forward(self, pre):

        out_rbf = self.rbf_net.forward(pre)
        out_ff = self.ff_net.forward(pre)

        post = out_rbf + out_ff

        return post
    
    def get_n_params_a_model(self):
        return len(self.get_a_model_params())

    # def get_ff_weights(self):
    #     return self.ff_net.get_weights()

    def get_models_params(self):
        return self.ff_net.get_models_params()

    def get_a_model_params(self):
        return self.ff_net.get_a_model_params()
            
    def get_rbf_params(self):
        return self.rbf_net.get_models_params()

    def get_a_rbf_params(self):
        return self.rbf_net.get_a_model_params()

    def set_models_params(self, flat_params):
        self.ff_net.set_models_params(flat_params)

    def set_a_model_params(self, flat_params):
        self.ff_net.set_a_model_params(flat_params)

    def set_rbf_params(self, flat_params):
        self.rbf_net.set_models_params(flat_params)

    def set_a_rbf_params(self, flat_params):
        self.rbf_net.set_a_model_params(flat_params)