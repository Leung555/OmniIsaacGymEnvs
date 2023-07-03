import numpy as np
import torch
from math import cos, sin, tanh

class RBFNet:
    def __init__(self, num_output, num_basis=10):
        """
        sizes: [input_size, hid_1, ..., output_size]
        """
        # cpg params
        self.omega = 0.01*np.pi
        self.alpha = 1.01
        self.o1 = torch.Tensor.repeat(torch.Tensor([]))
        self.o2 = 0.18
        # Rbf network
        input_range = (-1, 1)
        self.num_basis = num_basis
        self.num_output = num_output
        self.centers = np.linspace(input_range[0], input_range[1], num=num_basis)
        self.variance = 1/0.04
        self.weights = np.random.randn(num_output, num_basis, )
        self.rbf_array = []
        # self.weights = [torch.Tensor(sizes[i], sizes[i + 1]).uniform_(-1.0,1.0)
        #                     for i in range(len(sizes) - 1)]
          


    def forward(self):

        with torch.no_grad():
            self.o1 = tanh(self.alpha*( self.o1*cos(self.omega) + self.o2*sin(self.omega)))
            self.o2 = tanh(self.alpha*(-self.o1*sin(self.omega) + self.o2*cos(self.omega)))
            cpg_out = [[self.o1], [self.o2]]
            # print('o1', o1)
            post = np.sum(self.weights*np.exp(self.variance*(cpg_out - self.centers) ** 2), axis=1)

        return post #post.detach()

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
        pop, a, b = self.weights.shape
        self.weights = flat_params.reshape(self.popsize, a, b)
        # self.weights[i] = self.weights[i].cuda()
        # print('---- set_params ouput ---------------------------------')
        # print('self.test: ', self.weights[0])
        # print('----------------------------------------------')


    def get_weights(self):
        return [w for w in self.weights]
