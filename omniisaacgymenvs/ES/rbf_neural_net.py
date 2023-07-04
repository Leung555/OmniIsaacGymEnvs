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
        self.a = torch.Tensor([[0.01, 0.18]])
        print(self.a)
        self.O = self.a.expand(POPSIZE, 2).cuda()
        # Rbf network
        self.input_range = (-1, 1)
        self.num_basis = 10
        self.num_output = 1
        self.centers = torch.linspace(self.input_range[0], 
                                      self.input_range[1], self.num_basis).cuda()
        self.variance = 1/0.04
        self.weights = torch.randn(self.num_basis, self.POPSIZE, self.num_output,).cuda()
        self.W = torch.Tensor([[ cos(self.omega) ,  -sin(self.omega)], 
                                [ sin(self.omega) ,  cos(self.omega)]]).cuda()


    def forward(self, pre):

        with torch.no_grad():
            self.O = torch.tanh(self.alpha*torch.matmul(self.O, self.W))
            post = torch.sum(self.weights*torch.exp(-self.variance*(self.O - 
                self.centers.expand(self.POPSIZE*2,self.num_basis).transpose(0,1).
                reshape(self.num_basis, self.POPSIZE,2)) ** 2), dim=[0,2]).unsqueeze(1)
        return post.detach()

    def get_params(self):
        p = torch.cat([ params.flatten() for params in self.weights] )

        return p.cpu().flatten().numpy()

    def get_params_a_model(self):
        p = torch.cat([ params.flatten() for params in self.weights[:, 0]] )

        return p.cpu().flatten().numpy()
    
    def set_params(self, flat_params):
        flat_params = torch.from_numpy(flat_params)

        # print('---- set_params ---------------------------------')
        # print('self.test: ', self.weights[0])
        # print('----------------------------------------------')
        # print('w: ', w)
        basis, popsize, num_out = self.weights.shape
        self.weights = flat_params.reshape(basis, popsize, num_out).cuda()
        # self.weights[i] = self.weights[i].cuda()
        # print('---- set_params ouput ---------------------------------')
        # print('self.test: ', self.weights[0])
        # print('----------------------------------------------')


    def get_weights(self):
        return self.weights
