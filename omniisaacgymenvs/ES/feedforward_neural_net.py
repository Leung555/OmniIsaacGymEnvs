import numpy as np
import torch


def WeightStand(w, eps=1e-5):

    mean = torch.mean(input=w, dim=[0,1], keepdim=True)
    var = torch.var(input=w, dim=[0,1], keepdim=True)

    w = (w - mean) / torch.sqrt(var + eps)

    return w


class FeedForwardNet:
    def __init__(self, sizes):
        """
        sizes: [input_size, hid_1, ..., output_size]
        """
        self.weights = [torch.Tensor(sizes[i], sizes[i + 1]).uniform_(-0.5,0.5) 
                        for i in range(len(sizes) - 1)]
        # self.weights_gpu = torch.stack(self.weights).cuda()


    def forward(self, pre):

        with torch.no_grad():
            pre = torch.from_numpy(pre)
            """
            pre: (n_in, )
            """
            
            for i, W in enumerate(self.weights):
                post = torch.tanh(pre @ W.float())
                # post = torch.tanh(pre @ W.double())

                pre = post

        return post.detach()

    def get_params(self):
        p = torch.cat([ params.flatten() for params in self.weights] )
        return p.flatten().numpy()


    def set_params(self, flat_params):
        flat_params = torch.from_numpy(flat_params)

        m = 0
        for i, w in enumerate(self.weights):
            a, b = w.shape
            self.weights[i] = flat_params[m:m + a * b].reshape(a, b)
            m += a * b 


    def get_weights(self):
        return [w for w in self.weights]
