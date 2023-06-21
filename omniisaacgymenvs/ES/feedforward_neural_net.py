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
        self.weights = [torch.Tensor(sizes[i], sizes[i + 1]).uniform_(-1.0,1.0)
                            for i in range(len(sizes) - 1)]
          


    def forward(self, pre):

        with torch.no_grad():
            pre = torch.from_numpy(pre)
            """
            pre: (n_in, )
            """
            # print('---- forward ----------------------------------')
            # print('self.test: ', self.weights[0])
            # print('----------------------------------------------')
            for i in range(len(self.weights)):
                # W = self.weights[i]
                # print('pre: ', pre)
                # print('W: ', self.weights[i])
                post = torch.tanh(pre @ self.weights[i].float())
                # post = torch.tanh(pre @ W.double())

                pre = post
            
            # for i, W in enumerate(self.weights):
            #     # W = W.cuda()
            #     # print('pre: ', i, pre)
            #     # print('W: ', i, W)
            #     post = torch.tanh(pre @ W.float())
            #     # post = torch.tanh(pre @ W.double())

            #     pre = post
            # print('post: ', post)
            # print('post: ', post.detach())

        return post.detach()

    def get_params(self):
        p = torch.cat([ params.flatten() for params in self.weights] )

        return p.cpu().flatten().numpy()


    def set_params(self, flat_params):
        flat_params = torch.from_numpy(flat_params)

        m = 0
        # print('---- set_params ---------------------------------')
        # print('self.test: ', self.weights[0])
        # print('----------------------------------------------')
        for i, w in enumerate(self.weights):
            # print('w: ', w)
            a, b = w.shape
            self.weights[i] = flat_params[m:m + a * b].reshape(a, b)
            # self.weights[i] = self.weights[i].cuda()
            m += a * b 
        # print('---- set_params ouput ---------------------------------')
        # print('self.test: ', self.weights[0])
        # print('----------------------------------------------')


    def get_weights(self):
        return [w for w in self.weights]
