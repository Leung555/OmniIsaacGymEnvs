import numpy as np
import torch
from math import cos, sin, tanh

class RBFNet:
    def __init__(self, POPSIZE, num_output, num_basis=10):
        """
        sizes: [input_size, hid_1, ..., output_size]
        """
        self.POPSIZE = POPSIZE
        # cpg params
        # self.omega = 0.03*np.pi
        # self.alpha = 1.01
        # self.W = torch.Tensor([[ cos(self.omega) ,  -sin(self.omega)], 
        #                         [ sin(self.omega) ,  cos(self.omega)]]).cuda()
        self.O = torch.Tensor([[0.0, 0.18]]).expand(POPSIZE, 2).cuda()
        self.t, self.x, self.y, self.period = self.pre_compute_cpg()
        # self.o1 = torch.Tensor.repeat(torch.Tensor([0.00]), POPSIZE).unsqueeze(1)
        # self.o2 = torch.Tensor.repeat(torch.Tensor([0.18]), POPSIZE).unsqueeze(1)
        print('self.period: ', self.period)

        # Rbf network
        self.num_basis = num_basis
        self.num_output = num_output
        self.variance = 25.0
        self.phase = 0
        
        self.ci, self.cx, self.cy, self.rx, self.ry, self.KENNE = self.pre_rbf_centers(
            self.period, self.num_basis, self.x, self.y, self.variance)
        self.KENNE = self.KENNE.cuda()

        # self.centers = torch.linspace(self.input_range[0], 
        #                               self.input_range[1], num_basis).cuda()
        # self.weights = torch.randn(self.POPSIZE, self.num_basis, self.num_output,).cuda()
        x1 = np.linspace(0, 2*np.pi, self.num_basis)
        y1 = np.sin(x1) * 0.0
        # x2 = np.linspace(np.pi/2, 3*(np.pi)-np.pi/2, self.num_basis)
        # y2 = np.sin(x2) + 1.4
        # print('y2: ', y2)
        ze = np.zeros_like(y1)
        # self.weights = torch.Tensor([  y1,-y1, y1, -y1, y1,-y1, 
        #                               -y1,-y1, y1,  y1, y1,-y1,
        #                                y1, y1,-y1, -y1,-y1, y1]).T.repeat(self.POPSIZE, 1, 1).cuda()
        self.weights = torch.Tensor([  ze,-ze, ze, 
                                      -ze,-ze, ze,
                                      -ze,-ze, ze ]).T.repeat(self.POPSIZE, 1, 1).cuda()

        # print(self.weights)
        self.indices = torch.tensor([1, 2, 4, 5, 7, 8, 10, 11, 0, 3, 13, 14, 16, 17, 6, 9, 12, 15]).cuda()
        
        self.motor_encode = 'semi-indirect' # 'direct', 'indirect'
        if self.motor_encode == 'semi-indirect':
            phase_2 = int(self.period//2)
            self.phase = torch.Tensor([0, phase_2])
            # self.weights = torch.zeros(POPSIZE, num_basis, 9).cuda()
            self.weights = torch.Tensor(POPSIZE, num_basis, 9).uniform_(-0.2, 0.2).cuda()
            self.indices = torch.tensor([3, 6, 12, 15, 4, 7, 13, 16, 0, 9, 5, 8, 14, 17, 1, 10, 2, 11]).cuda()

    def forward(self, pre):

        with torch.no_grad():
            # Indirect encoding ##################################
            p1 = self.KENNE[int(self.phase[0])]
            p2 = self.KENNE[int(self.phase[1])]
            # out_p1 = torch.matmul(p1, self.weights)
            # out_p2 = torch.matmul(p2, self.weights)
            out_p1 = torch.tanh(torch.matmul(p1, self.weights)) * 0.5
            out_p2 = torch.tanh(torch.matmul(p2, self.weights)) * 0.5
            outL = torch.concat([out_p1[:, 0:3], out_p2[:, 3:6], out_p1[:, 6:9]], dim=1)
            outR = torch.concat([out_p2[:, 0:3], out_p1[:, 3:6], out_p2[:, 6:9]], dim=1)
            post = torch.concat([outL, outR], dim=1)
            post = torch.index_select(post, 1, self.indices)
            
            self.phase = self.phase + 1
            self.phase = torch.where(self.phase > self.period, 0, self.phase) 
            ####################################################
            
            # Direct encoding ##################################
            # post = torch.matmul(self.KENNE[self.phase], self.weights)
            # post = torch.index_select(post, 1, self.indices)
            # self.phase += 1
            # if self.phase > self.period:
            #     self.phase = 0            
            ####################################################

            # print(post[0])

        return post.float().detach()

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
        popsize, basis, num_out = self.weights.shape
        self.weights = flat_params.reshape(popsize, basis, num_out).cuda()
        # self.weights[i] = self.weights[i].cuda()
        # print('---- set_params ouput ---------------------------------')
        # print('self.test: ', self.weights[0])
        # print('----------------------------------------------')

    def set_params_single_model(self, flat_params):
        flat_params = torch.from_numpy(flat_params)
        # print('flat_params: ', flat_params)

        popsize, basis, num_out = self.weights.shape
        self.weights = flat_params.repeat(popsize, 1, 1).reshape(popsize, basis, num_out).cuda()
            
    def get_weights(self):
        return self.weights
    
    def pre_compute_cpg(self):
        # Run for one period
        phi   = 0.01*np.pi # SO(2) Frequency
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