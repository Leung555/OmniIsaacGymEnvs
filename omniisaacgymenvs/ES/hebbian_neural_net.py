import numpy as np
import torch


def WeightStand(w, eps=1e-5):

    mean = torch.mean(input=w, dim=[0,1], keepdim=True)
    var = torch.var(input=w, dim=[0,1], keepdim=True)

    w = (w - mean) / torch.sqrt(var + eps)

    return w


class HebbianNet:
    def __init__(self, sizes):
        """
        sizes: [input_size, hid_1, ..., output_size]
        """
        self.weights = [torch.Tensor(sizes[i], sizes[i + 1]).uniform_(-0.1,0.1) for i in range(len(sizes) - 1)]

        self.A = [torch.normal(0,.1, (sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]
        self.B = [torch.normal(0,.1, (sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]
        self.C = [torch.normal(0,.1, (sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]
        self.D = [torch.normal(0,.1, (sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]
        self.lr = [torch.normal(0,.1, (sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]
        


    def forward(self, pre):

        with torch.no_grad():
            pre = torch.from_numpy(pre)
            """
            pre: (n_in, )
            """
            
            for i, W in enumerate(self.weights):
                post = torch.tanh(pre @ W.float())
                # post = torch.tanh(pre @ W.double())

                self.weights[i] = self.hebbian_update(W, pre, post, self.A[i], self.B[i], self.C[i], self.D[i], self.lr[i])
                pre = post

        return post.detach()


    def hebbian_update(self, weights, pre,post, A, B, C, D, lr):


        i = torch.ones(weights.shape) * pre.unsqueeze(1)
        j = torch.ones(weights.shape) * post
        ij = i * j


        weights = weights + lr * (A*ij + B*i + C*j + D)
        weights = WeightStand(weights)
        return weights


    def get_params(self):
        p = torch.cat([ params.flatten() for params in self.A]  
                +[ params.flatten() for params in self.B] 
                +[ params.flatten() for params in self.C]
                +[ params.flatten() for params in self.D]
                +[ params.flatten() for params in self.lr]
                )
        return p.flatten().numpy()


    def set_params(self, flat_params):
        flat_params = torch.from_numpy(flat_params)

        m = 0
        for i, hebb_A in enumerate(self.A):
            a, b = hebb_A.shape
            self.A[i] = flat_params[m:m + a * b].reshape(a, b)
            m += a * b 

        for i, hebb_B in enumerate(self.B):
            a, b = hebb_B.shape
            self.B[i] = flat_params[m:m + a * b].reshape(a, b)
            m += a * b 

        for i, hebb_C in enumerate(self.C):
            a, b = hebb_C.shape
            self.C[i] = flat_params[m:m + a * b].reshape(a, b)
            m += a * b 

        for i, hebb_D in enumerate(self.D):
            a, b = hebb_D.shape
            self.D[i] = flat_params[m:m + a * b].reshape(a, b)
            m += a * b 

        for i, hebb_lr in enumerate(self.lr):
            a, b = hebb_lr.shape
            self.lr[i] = flat_params[m:m + a * b].reshape(a, b)
            m += a * b 


    def get_weights(self):
        return [w for w in self.weights]
