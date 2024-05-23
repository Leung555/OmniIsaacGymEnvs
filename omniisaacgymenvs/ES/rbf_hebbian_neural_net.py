import numpy as np
import torch
from math import cos, sin, tanh

def WeightStand(w, eps=1e-5):
    max_val = torch.max(torch.abs(w).flatten(start_dim=1, end_dim=2), dim=1)
    max_val = max_val[0].unsqueeze(1).unsqueeze(2)    # print('w - mean: ', w - mean)
    w = w / max_val

    return w

class RBFHebbianNet:
    def __init__(self, POPSIZE, num_output, num_basis=10, 
                 ARCHITECTURE=[47, 64, 32, 18], 
                 mode='simple_RBF_hebb',
                 hebb_init_wnoise=0.1,
                 training_option=2):
        """
        sizes: [input_size, hid_1, ..., output_size]
        """

        self.mode = mode
        self.POPSIZE = POPSIZE
        self.training_option = training_option
        self.O = torch.Tensor([[0.0, 0.18]]).expand(POPSIZE, 2).cuda()
        self.t, self.x, self.y, self.period = self.pre_compute_cpg()
        print('self.period: ', self.period)

        # Rbf network
        self.num_basis = num_basis
        self.num_output = num_output
        self.variance = 25.0
        self.phase = 0
        self.p1 = torch.zeros(num_basis).cuda()
        self.p2 = torch.zeros(num_basis).cuda()
        
        self.ci, self.cx, self.cy, self.rx, self.ry, self.KENNE = self.pre_rbf_centers(
            self.period, self.num_basis, self.x, self.y, self.variance)
        self.KENNE = self.KENNE.cuda()

        # self.centers = torch.linspace(self.input_range[0], 
        #                               self.input_range[1], num_basis).cuda()
        # self.RBFweights = torch.randn(self.POPSIZE, self.num_basis, self.num_output,).cuda()
        x1 = np.linspace(0, 2*np.pi, self.num_basis)
        y1 = np.sin(x1) * 0.0
        # x2 = np.linspace(np.pi/2, 3*(np.pi)-np.pi/2, self.num_basis)
        # y2 = np.sin(x2) + 1.4
        # print('y2: ', y2)
        ze = np.zeros_like(y1)
        # self.RBFweights = torch.Tensor([  y1,-y1, y1, -y1, y1,-y1, 
        #                               -y1,-y1, y1,  y1, y1,-y1,
        #                                y1, y1,-y1, -y1,-y1, y1]).T.repeat(self.POPSIZE, 1, 1).cuda()
        self.RBFweights = torch.Tensor([  ze,-ze, ze, 
                                      -ze,-ze, ze,
                                      -ze,-ze, ze ]).T.repeat(self.POPSIZE, 1, 1).cuda()

        # print(self.RBFweights)
        self.indices = torch.tensor([1, 2, 4, 5, 7, 8, 10, 11, 0, 3, 13, 14, 16, 17, 6, 9, 12, 15]).cuda()
        self.hyper_architecture = [27, 64, 180]
        self.architecture = ARCHITECTURE

        self.motor_encode = 'semi-indirect' # 'direct', 'indirect'
        if self.motor_encode == 'semi-indirect':
            phase_2 = int(self.period//2)
            self.phase = torch.Tensor([0, phase_2])
            # self.RBFweights = torch.zeros(POPSIZE, num_basis, 9).cuda()
            self.RBFweights = torch.Tensor(POPSIZE, num_basis, self.num_output//2).uniform_(-0.2, 0.2).cuda()
            # self.indices = torch.tensor([3, 6, 12, 15, 4, 7, 13, 16, 0, 9, 5, 8, 14, 17, 1, 10, 2, 11]).cuda()
            
            # Dung beetle robot configuration
            # self.indices = torch.tensor([ 2, 0, 3, 1, 6, 4, 7, 5, 10, 8, 11, 9]).cuda()
            # Ant robot configuration
            self.indices = torch.tensor([0, 2, 6, 4, 1, 3, 7, 5]).cuda()
            
            print('self.RBFweights: ', self.RBFweights.shape)
            if self.mode == 'parallel_Hebb':
                sizes = self.architecture
                sizes[0] += num_basis
                self.architecture = sizes
                self.weights = [torch.Tensor(POPSIZE, sizes[i], sizes[i + 1]).uniform_(-hebb_init_wnoise, hebb_init_wnoise).cuda()
                                    for i in range(len(sizes) - 1)]
                print('self.weights.shape: ', sizes)
                self.one_array = [torch.ones(POPSIZE, sizes[i], sizes[i + 1]).cuda()
                                    for i in range(len(sizes) - 1)]
                # print('self.one_array', self.one_array)
                self.A = [torch.normal(0,.01, (POPSIZE, sizes[i], sizes[i + 1])).cuda() for i in range(len(sizes) - 1)]
                self.B = [torch.normal(0,.01, (POPSIZE, sizes[i], sizes[i + 1])).cuda() for i in range(len(sizes) - 1)]
                self.C = [torch.normal(0,.01, (POPSIZE, sizes[i], sizes[i + 1])).cuda() for i in range(len(sizes) - 1)]
                self.D = [torch.normal(0,.01, (POPSIZE, sizes[i], sizes[i + 1])).cuda() for i in range(len(sizes) - 1)]
                self.lr = [torch.normal(0,.01, (POPSIZE, sizes[i], sizes[i + 1])).cuda() for i in range(len(sizes) - 1)]


    def forward(self, pre):

        with torch.no_grad():
            # Indirect encoding ##################################
            self.p1 = self.KENNE[int(self.phase[0])]
            self.p2 = self.KENNE[int(self.phase[1])]
            # print(p1.shape)
            # print(self.weights.shape)
            # out_p1 = torch.matmul(p1, self.weights)
            # out_p2 = torch.matmul(p2, self.weights)
            out_p1 = torch.tanh(torch.matmul(self.p1.float(), self.RBFweights.float()))
            out_p2 = torch.tanh(torch.matmul(self.p2.float(), self.RBFweights.float()))
            # print(out_p1.shape)
            # 4 legs
            # outL = torch.concat([out_p1[:, 0:3], out_p2[:, 3:6]], dim=1)
            # outR = torch.concat([out_p2[:, 0:3], out_p1[:, 3:6]], dim=1)
            # 6 leg
            outL = torch.concat([out_p1[:, 0:2], out_p2[:, 2:4]], dim=1)
            outR = torch.concat([out_p2[:, 0:2], out_p1[:, 2:4]], dim=1)
            # outL = torch.concat([out_p1[:, 0:3], out_p2[:, 3:6], out_p1[:, 6:9]], dim=1)
            # outR = torch.concat([out_p2[:, 0:3], out_p1[:, 3:6], out_p2[:, 6:9]], dim=1)
            # print(outL.shape)
            post = torch.concat([outL, outR], dim=1)
            post = torch.index_select(post, 1, self.indices)
            # print('post.shape: ', post.shape)
            
            self.phase = self.phase + 1
            self.phase = torch.where(self.phase > self.period, 0, self.phase)
            ####################################################

            if self.mode == 'parallel_Hebb':
                # print('pre: ', pre.shape)
                # print('self.p1: ', self.p1.repeat(self.POPSIZE, 1).shape)
                pre = torch.concat((pre, self.p1.repeat(self.POPSIZE, 1)), dim=1)
                # print('pre: ', pre.shape)
                for i, W in enumerate(self.weights):
                    out =  torch.tanh(torch.einsum('ij, ijk -> ik', pre.float(), W.float()))
                    if i == 2:
                        out = post + out
                    self.weights[i] = self.hebbian_update(i, W, pre, out, self.A[i], self.B[i], self.C[i], self.D[i], self.lr[i])
                    pre = out
                post = out

            # simple change RBF weight to hebbian

            # Direct encoding ##################################
            # post = torch.matmul(self.KENNE[self.phase], self.weights)
            # post = torch.index_select(post, 1, self.indices)
            # self.phase += 1
            # if self.phase > self.period:
            #     self.phase = 0
            ####################################################

            # print(post[0])

        return post.float().detach()
    
    def hebbian_update(self, hid_num ,weights, pre, post, A, B, C, D, lr):

        i = self.one_array[hid_num] * pre.unsqueeze(2)
        j = post.unsqueeze(2).expand(-1,-1, weights.shape[1]).transpose(1,2)
        ij = i * j

        weights = weights + lr * (A*ij + B*i + C*j + D)
        # weights = weights + lr * (C*j + D)

        weights = WeightStand(weights)

        return weights

    def get_n_params_a_model(self):
        return np.sum(self.architecture)*5

    def get_hebb_weights(self):
        return [w for w in self.weights][0]

    def get_models_params(self):
        return self.get_hebb_params().cpu().flatten().numpy()

    def get_a_model_params(self):
        return self.get_a_hebb_params().cpu().flatten().numpy()
            
    def get_rbf_params(self):
        p = torch.cat([params.flatten() for params in self.RBFweights])
        return p

    def get_a_rbf_params(self):
        p = torch.cat([params.flatten() for params in self.RBFweights[0]])
        return p

    def get_hebb_params(self):
        p = torch.cat([ params.flatten() for params in self.A]  
                     +[ params.flatten() for params in self.B] 
                     +[ params.flatten() for params in self.C]
                     +[ params.flatten() for params in self.D]
                     +[ params.flatten() for params in self.lr]
                     )
        return p

    def get_a_hebb_params(self):
        p = torch.cat([ params[0].flatten() for params in self.A]  
                     +[ params[0].flatten() for params in self.B] 
                     +[ params[0].flatten() for params in self.C]
                     +[ params[0].flatten() for params in self.D]
                     +[ params[0].flatten() for params in self.lr]
                     )
        return p

    def set_models_params(self, flat_params):
        flat_params = torch.from_numpy(flat_params)
        self.set_hebb_params(flat_params)

    def set_a_model_params(self, flat_params):
        flat_params = torch.from_numpy(flat_params)
        self.set_a_hebb_params(flat_params)

    def set_rbf_params(self, flat_params):
        flat_params = torch.from_numpy(flat_params)

        popsize, basis, num_out = self.RBFweights.shape
        self.RBFweights = flat_params.reshape(popsize, basis, num_out).cuda()

    def set_a_rbf_params(self, flat_params):
        flat_params = torch.from_numpy(flat_params)

        popsize, basis, num_out = self.RBFweights.shape
        self.RBFweights = flat_params.repeat(popsize, 1, 1).reshape(popsize, basis, num_out).cuda()

    def set_hebb_params(self, flat_params):
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

    def set_a_hebb_params(self, flat_params):
        m = 0
        for i, hebb_A in enumerate(self.A):
            pop, a, b = hebb_A.shape
            self.A[i] = flat_params[m:m + a * b].repeat(pop, 1, 1).reshape(pop, a, b).cuda()
            m += a * b 

        for i, hebb_B in enumerate(self.B):
            pop, a, b = hebb_B.shape
            self.B[i] = flat_params[m:m + a * b].repeat(pop, 1, 1).reshape(pop, a, b).cuda()
            m += a * b 

        for i, hebb_C in enumerate(self.C):
            pop, a, b = hebb_C.shape
            self.C[i] = flat_params[m:m + a * b].repeat(pop, 1, 1).reshape(pop, a, b).cuda()
            m += a * b 

        for i, hebb_D in enumerate(self.D):
            pop, a, b = hebb_D.shape
            self.D[i] = flat_params[m:m + a * b].repeat(pop, 1, 1).reshape(pop, a, b).cuda()
            m += a * b 

        for i, hebb_lr in enumerate(self.lr):
            pop, a, b = hebb_lr.shape
            self.lr[i] = flat_params[m:m + a * b].repeat(pop, 1, 1).reshape(pop, a, b).cuda()
            m += a * b 

    def pre_compute_cpg(self):
        # Run for one period
        phi   = 0.03*np.pi # SO(2) Frequency
        alpha = 1.01         # SO(2) Alpha term
        w11   = alpha*cos(phi)
        w12   = alpha*sin(phi)
        w21   =-w12
        w22   = w11
        x     = []
        y     = []
        t     = []
        t.append(0)
        x.append(-0.197)
        y.append(0.0)
        period = 0
        while y[period] >= y[0]:
            period = period+1
            t.append(period*0.0167)
            x.append(tanh(w11*x[period-1]+w12*y[period-1]))
            y.append(tanh(w22*y[period-1]+w21*x[period-1]))
            
        while y[period] <= y[0]:
            period = period+1
            t.append(period*0.0167)
            x.append(tanh(w11*x[period-1]+w12*y[period-1]))
            y.append(tanh(w22*y[period-1]+w21*x[period-1]))
        period = period
        return t, x, y, period
    
    def pre_rbf_centers(self, period, num_basis, x, y, var):
        KENNE  = [0]*num_basis  # Kernels
        ci = np.asarray(np.around(np.linspace(1, period, num_basis+1)), dtype=int)

        ci = ci[:-1]

        cx = [0] * (len(ci))
        cy = [0] * (len(ci))
        cxy = [0] * (len(ci))

        xy = x+y

        for k in range(len(ci)):
            cx[k] = x[ci[k]]
            cy[k] = y[ci[k]]

        for i in range(num_basis):
            rx   = [q - cx[i] for q in x]
            ry   = [q - cy[i] for q in y]
            KENNE[i] = np.exp(-(np.power((rx),2) + np.power((ry),2))*var)

        return ci, cx, cy, rx, ry, torch.from_numpy(np.array(KENNE).T)