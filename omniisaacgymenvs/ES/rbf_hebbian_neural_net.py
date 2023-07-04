import numpy as np
import torch
from math import cos, sin, tanh

def WeightStand(w, eps=1e-5):

    mean = torch.mean(input=w, dim=[1,2], keepdim=True)
    var = torch.var(input=w, dim=[1,2], keepdim=True)

    w = (w - mean) / torch.sqrt(var)

    return w

class RBFHebbianNet:
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
        self.A = torch.randn(self.POPSIZE, self.num_basis, self.num_output,).cuda()
        self.B = torch.randn(self.POPSIZE, self.num_basis, self.num_output,).cuda()
        self.C = torch.randn(self.POPSIZE, self.num_basis, self.num_output,).cuda()
        self.D = torch.randn(self.POPSIZE, self.num_basis, self.num_output,).cuda()
        self.lr = torch.randn(self.POPSIZE, self.num_basis, self.num_output,).cuda()
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

    def hebbian_update(self, hid_num ,weights, pre, post, A, B, C, D, lr):

        i = self.one_array[hid_num] * pre.unsqueeze(2)
        j = post.unsqueeze(2).expand(-1,-1, weights.shape[1]).transpose(1,2)
        ij = i * j

        new_weights = weights + lr * (A*ij + B*i + C*j + D)
        # print('weights update: ', weights)
        new_weights = WeightStand(weights)        
        
        return new_weights

    def get_params(self):
        p = torch.cat([ params.flatten() for params in self.A]  
                +[ params.flatten() for params in self.B] 
                +[ params.flatten() for params in self.C]
                +[ params.flatten() for params in self.D]
                +[ params.flatten() for params in self.lr]
                )
        return p.cpu().flatten().numpy()

    def get_params_a_model(self):
        p = torch.cat([ params[0].flatten() for params in self.A]  
                +[ params[0].flatten() for params in self.B] 
                +[ params[0].flatten() for params in self.C]
                +[ params[0].flatten() for params in self.D]
                +[ params[0].flatten() for params in self.lr]
                )
        return p.flatten().numpy()
        
    def set_params(self, flat_params):
        flat_params = torch.from_numpy(flat_params)

        m = 0
        for i, hebb_A in enumerate(self.A):
            pop, a, b = hebb_A.shape
            self.A[i] = flat_params[:, m:m + a * b].reshape(pop, a, b).cuda()
            m += a * b 
        
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
