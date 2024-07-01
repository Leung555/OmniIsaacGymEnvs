import numpy as np
import torch


def WeightStand(w, eps=1e-5):

    mean = torch.mean(input=w, dim=[1,2], keepdim=True)
    var = torch.var(input=w, dim=[1,2], keepdim=True)

    w = (w - mean) / torch.sqrt(var + eps)

    return w


class FeedForwardNet:
    def __init__(self, popsize, sizes):
        """
        sizes: [input_size, hid_1, ..., output_size]
        """
        self.weights = [torch.Tensor(popsize, sizes[i], sizes[i + 1]).uniform_(10, 10).cuda()
                            for i in range(len(sizes) - 1)]
        self.architecture = sizes 
        # print('Weight: ', self.weights)   


    def forward(self, pre):
        # print('pre: ', pre)

        with torch.no_grad():
            # pre = torch.from_numpy(pre)
            """
            pre: (n_in, )
            """
            # print('---- forward ----------------------------------')
            # print('self.test: ', self.weights[0])
            # print('----------------------------------------------')
            # c = 0
            for i, W in enumerate(self.weights):
                # W = W.cuda()
                # print('pre: ', i, pre.shape)
                # print('W: ', i, W.shape)
                post =  torch.tanh(torch.einsum('ij, ijk -> ik', pre, W.float()))
                # post = torch.tanh(pre @ W.float())
                # post = torch.tanh(pre @ W.double())

                pre = post
                # c+=1
            
            # print('c: ', c)
                # print('post: ', post)
            # print('post: ', post.detach())

        return post.detach()

    def get_n_params_a_model(self):
        return len(self.get_a_model_params())

    def get_models_params(self):
        p = torch.cat([ params.flatten() for params in self.weights] )

        return p.cpu().flatten().numpy()
    
    def get_a_model_params(self):
        p = torch.cat([ params[0].flatten() for params in self.weights] )

        return p.cpu().flatten().numpy()

    def set_models_params(self, flat_params):
        flat_params = torch.from_numpy(flat_params)
        m = 0
        for i, w in enumerate(self.weights):
            pop, a, b = w.shape
            self.weights[i] = flat_params[:, m:m + a * b].reshape(pop, a, b).cuda()
            m += a * b 

    def set_a_model_params(self, flat_params):
        flat_params = torch.from_numpy(flat_params)
        m = 0
        for i, w in enumerate(self.weights):
            pop, a, b = w.shape
            self.weights[i] = flat_params[m:m + a * b].repeat(pop, 1, 1).reshape(pop, a, b).cuda()
            m += a * b 

    def get_weights(self):
        return [w for w in self.weights]