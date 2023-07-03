import numpy as np
import torch


def WeightStand(w, eps=1e-5):

    mean = torch.mean(input=w, dim=[1,2], keepdim=True)
    var = torch.var(input=w, dim=[1,2], keepdim=True)

    w = (w - mean) / torch.sqrt(var + eps)

    return w


class FeedForwardNet:
    def __init__(self, sizes, popsize):
        """
        sizes: [input_size, hid_1, ..., output_size]
        """
        self.weights = [torch.Tensor(popsize, sizes[i], sizes[i + 1]).uniform_(-1.0,1.0)
                            for i in range(len(sizes) - 1)]
        self.architecture = sizes 
        # print('Weight: ', self.weights)   


    def forward(self, pre):

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
                # print('pre: ', i, pre)
                # print('W: ', i, W)
                post =  torch.tanh(torch.einsum('ij, ijk -> ik', pre, W.float()))
                # post = torch.tanh(pre @ W.float())
                # post = torch.tanh(pre @ W.double())

                pre = post
                # c+=1
            
            # print('c: ', c)
            # print('post: ', post)
            # print('post: ', post.detach())

        return post.detach()

    def get_params(self):
        p = torch.cat([ params.flatten() for params in self.weights] )

        return p.cpu().flatten().numpy()
    
    def get_params_a_model(self):
        p = torch.cat([ params[0].flatten() for params in self.weights] )

        return p.cpu().flatten().numpy()


    def set_params(self, flat_params):
        flat_params = torch.from_numpy(flat_params)
        # print('flat_params: ', flat_params)

        m = 0
        # print('---- set_params ---------------------------------')
        # print('flat_params: ', flat_params)
        # print('flat_params_slice: ', flat_params[0:4])
        # print('self.weights[0]: ', self.weights[0])
        # print('----------------------------------------------')
        for i, w in enumerate(self.weights):
            # print('w: ', w)
            pop, a, b = w.shape
            # print('pop, a, b', pop, a, b)
            self.weights[i] = flat_params[:, m:m + a * b].reshape(pop, a, b).cuda()
            # self.weights[i] = self.weights[i].cuda()
            m += a * b 
        # print('---- set_params ouput ---------------------------------')
        # print('self.weights_already set: ', self.weights)
        # print('----------------------------------------------')


    def get_weights(self):
        return [w for w in self.weights]
