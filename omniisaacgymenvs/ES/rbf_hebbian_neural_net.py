import numpy as np
import torch

def WeightStand(w, eps=1e-5):

    mean = torch.mean(input=w, dim=[0,1], keepdim=True)
    var = torch.var(input=w, dim=[0,1], keepdim=True)

    w = (w - mean) / torch.sqrt(var + eps)

    return w


class RBFHebbianNet:
    def __init__(self, num_output, num_basis=10):
        """
        sizes: [input_size, hid_1, ..., output_size]
        """
        # cpg params
        # self.omega = 0.01*np.pi
        # self.o1, self.o2 = 0.01, 0.01
        # # Rbf network
        # input_range = (-1, 1)
        # self.num_basis = num_basis
        # self.num_output = num_output
        # self.centers = np.linspace(input_range[0], input_range[1], num=num_basis)
        # self.variance = 1/0.04
        # self.weights = np.random.randn(num_output, num_basis, )

        # self.weights = [torch.Tensor(sizes[i], sizes[i + 1]).uniform_(-1.0,1.0)
        #                     for i in range(len(sizes) - 1)]
          


    def forward(self):

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
