import numpy as np
import torch
from math import cos, sin, tanh

class RBFNet:
    def __init__(self, POPSIZE, num_output, num_basis=10):
        """
        sizes: [input_size, hid_1, ..., output_size]
        """
        self.POPSIZE = POPSIZE
        # cpg params
        self.omega = 0.01*np.pi
        self.alpha = 1.01
        # self.o1 = torch.Tensor.repeat(torch.Tensor([0.00]), POPSIZE).unsqueeze(1)
        # self.o2 = torch.Tensor.repeat(torch.Tensor([0.18]), POPSIZE).unsqueeze(1)

        self.O = torch.Tensor([[0.01, 0.18]]).expand(POPSIZE, 2).cuda()
        # Rbf network
        self.input_range = (-1, 1)
        self.num_basis = num_basis
        self.num_output = num_output
        self.centers = torch.linspace(self.input_range[0], 
                                      self.input_range[1], num_basis).cuda()
        self.variance = 1/0.04
        self.weights = torch.randn(self.POPSIZE, self.num_basis, self.num_output,).cuda()
        # print(self.weights)
        self.W = torch.Tensor([[ cos(self.omega) ,  -sin(self.omega)], 
                                [ sin(self.omega) ,  cos(self.omega)]]).cuda()


    def forward(self, pre):

        with torch.no_grad():
            self.O = torch.tanh(self.alpha*torch.matmul(self.O, self.W))
            post = torch.matmul(torch.exp(-self.variance*torch.sum((self.O.reshape(self.POPSIZE, 2, 1).
                    expand(self.POPSIZE,2,self.num_basis) - self.centers) ** 2, dim=1)).double(), self.weights.double())[:, 0]
            # post = torch.matmul(a.double(), self.weights.double())[:, 0]
        return post.float().detach()

    def get_params(self):
        p = torch.cat([ params.flatten() for params in self.weights] )

        return p.cpu().flatten().numpy()

    def get_params_a_model(self):
        p = torch.cat([ params.flatten() for params in self.weights[0]] )

        return p.cpu().flatten().numpy()
    
    def set_params(self, flat_params):
        flat_params = torch.from_numpy(flat_params)

        # print('---- set_params ---------------------------------')
        # print('self.test: ', self.weights[0])
        # print('----------------------------------------------')
        # print('w: ', w)
        popsize, basis, num_out = self.weights.shape
        self.weights = flat_params.reshape(popsize, basis, num_out).cuda()
        # self.weights[i] = self.weights[i].cuda()
        # print('---- set_params ouput ---------------------------------')
        # print('self.test: ', self.weights[0])
        # print('----------------------------------------------')


    def get_weights(self):
        return self.weights
