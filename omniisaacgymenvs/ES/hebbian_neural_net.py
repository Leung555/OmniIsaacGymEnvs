import numpy as np
import torch


def WeightStand(w, eps=1e-5):

    mean = torch.mean(input=w, dim=[0,1], keepdim=True)
    var = torch.var(input=w, dim=[0,1], keepdim=True)

    w = (w - mean) / torch.sqrt(var + eps)

    return w


class HebbianNet:
    def __init__(self, sizes, popsize):
        """
        sizes: [input_size, hid_1, ..., output_size]
        """
        self.weights = [torch.Tensor(popsize, sizes[i], sizes[i + 1]).uniform_(-1.0,1.0).cuda()
                            for i in range(len(sizes) - 1)]
        self.architecture = sizes
        self.one_array = [torch.ones(popsize, sizes[i], sizes[i + 1]).cuda()
                            for i in range(len(sizes) - 1)]
        # print('self.one_array', self.one_array)
        self.A = [torch.normal(0,.1, (popsize, sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]
        self.B = [torch.normal(0,.1, (popsize, sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]
        self.C = [torch.normal(0,.1, (popsize, sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]
        self.D = [torch.normal(0,.1, (popsize, sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]
        self.lr = [torch.normal(0,.1, (popsize, sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]
        


    def forward(self, pre):

        with torch.no_grad():
            # pre = torch.from_numpy(pre)
            """
            pre: (n_in, )
            """
            
            for i, W in enumerate(self.weights):
                # pre = torch.Tensor([[1., 0], 
                #                     [1., 0]]).cuda()
                # W = torch.ones(weights.shape).cuda()
                post =  torch.einsum('ij, ijk -> ik', pre, W.float())
                # post = torch.tanh(pre @ W.float())
                # post = torch.tanh(pre @ W.double())

                self.weights[i] = self.hebbian_update(i, W, pre, post, self.A[i], self.B[i], self.C[i], self.D[i], self.lr[i])
                pre = post

        return post.detach()


    def hebbian_update(self, hid_num ,weights, pre,post, A, B, C, D, lr):

        # print('\n------- Hebbian_Update --------------')
        # print('hidden layer: ', hid_num+1)
        # print('pre: ', pre)
        # print()
        # print('weights: ', weights.shape)
        # print('torch.ones(weights.shape): ', torch.ones(weights.shape))
        # print()
        # print('post: ', post)
        # print()
        # i = pre.reshape(weights.shape)
        i = self.one_array[hid_num] * pre.unsqueeze(2)
        # i = torch.ones(weights.shape).cuda() * pre.unsqueeze(2)
        # print('i: ', i)
        # print()
        # post_unsq = post.unsqueeze(1)
        # print('post_unsq: ', post_unsq)
        # print()
        # print('torch.cat(): ', torch.cat((post_unsq)*weights.shape[0], 1))
        # print()
        # j = torch.ones(weights.shape).cuda() * post
        # j = post.repeat(1, weights.shape[1]).unsqueeze(2)
        # j = post.unsqueeze(2).repeat(1,1,weights.shape[1]).transpose(1,2)
        j = post.unsqueeze(2).expand(-1,-1, weights.shape[1]).transpose(1,2)
        ij = i * j
        # print('j: ', j)
        # print()
        # print('ij: ', ij)


        weights = weights + lr * (A*ij + B*i + C*j + D)
        weights = WeightStand(weights)
        # print('------------- Hebb-------------------\n')

        return weights


    def get_params(self):
        p = torch.cat([ params.flatten() for params in self.A]  
                +[ params.flatten() for params in self.B] 
                +[ params.flatten() for params in self.C]
                +[ params.flatten() for params in self.D]
                +[ params.flatten() for params in self.lr]
                )
        return p.flatten().numpy()

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

        for i, hebb_B in enumerate(self.B):
            pop, a, b = hebb_B.shape
            self.B[i] = flat_params[:, m:m + a * b].reshape(pop, a, b).cuda()
            m += a * b 

        for i, hebb_C in enumerate(self.C):
            pop, a, b = hebb_C.shape
            self.C[i] = flat_params[:, m:m + a * b].reshape(pop, a, b).cuda()
            m += a * b 

        for i, hebb_D in enumerate(self.D):
            pop, a, b = hebb_D.shape
            self.D[i] = flat_params[:, m:m + a * b].reshape(pop, a, b).cuda()
            m += a * b 

        for i, hebb_lr in enumerate(self.lr):
            pop, a, b = hebb_lr.shape
            self.lr[i] = flat_params[:, m:m + a * b].reshape(pop, a, b).cuda()
            m += a * b 


    def get_weights(self):
        return [w for w in self.weights]