import numpy as np
import torch
from math import cos, sin, tanh
from omniisaacgymenvs.ES.robot_config import get_num_legjoints

class RBFNet:
    def __init__(self, 
                 popsize, 
                 num_basis,
                 num_output,
                 robot,
                 motor_encode='semi-indirect'):
        """
        sizes: [input_size, hid_1, ..., output_size]
        """
        self.architecture = [num_basis, num_output]
        self.popsize = popsize

        # Initialize CPG
        self.O = torch.Tensor([[0.0, 0.18]]).expand(popsize, 2).cuda()
        self.t, self.x, self.y, self.period = self.pre_compute_cpg()

        # Rbf network
        self.num_basis = num_basis
        self.num_output = num_output
        self.variance = 25.0
        self.phase = 0

        # Pre calculated rbf layers output 
        self.ci, self.cx, self.cy, self.rx, self.ry, self.KENNE = self.pre_rbf_centers(
            self.period, self.num_basis, self.x, self.y, self.variance)
        self.KENNE = self.KENNE.cuda()

        # Get number of legs, joints, and motor mapping from model --> sim robot
        self.num_legs, self.num_joints, motor_mapping = get_num_legjoints(robot)
        self.indices = motor_mapping.cuda()

        # initilize motor encoding type (weights, CPGs' phase)
        self.motor_encode = motor_encode # 'direct', 'indirect'
        if self.motor_encode == 'semi-indirect':
            self.weights = torch.Tensor(popsize, num_basis, num_output//2).uniform_(-0.1, 0.1).cuda()
            # Initilize phase of each CPG
            phase_2 = int(self.period//2)
            self.phase = torch.Tensor([0, phase_2])

        step = self.num_joints
        self.indices_L = [(0, i * step) if i % 2 == 0 else (1, i * step) 
                          for i in range(self.num_legs//2)]
        self.indices_R = [(1, i * step) if i % 2 == 0 else (0, i * step) 
                          for i in range(self.num_legs//2)]


    def forward(self, pre):
        # print('pre: ', pre)
        
        with torch.no_grad():
            # Indirect encoding ##################################
            p1 = self.KENNE[int(self.phase[0])]
            p2 = self.KENNE[int(self.phase[1])]
            out_p1 = torch.tanh(torch.matmul(p1, self.weights))
            out_p2 = torch.tanh(torch.matmul(p2, self.weights))
            # print('out_p1: ', out_p1.shape)
            outL, outR = self.concat_slices([out_p1, out_p2])
            # print('outR: ', outR)
            post = torch.concat([outL, outR], dim=1)
            # print('post: ', post.shape)
            post = torch.index_select(post, 1, self.indices)
            # print('post: ', post)
            # print('post index: ', post.shape)

            self.phase = self.phase + 1
            self.phase = torch.where(self.phase > self.period, 0, self.phase) 
            ####################################################

        return post.float().detach()
    
    def concat_slices(self, tensors, dim=1):    
        outL = torch.cat([tensors[i][:, j:j+self.num_joints] for i, j in self.indices_L], dim=dim)
        outR = torch.cat([tensors[i][:, j:j+self.num_joints] for i, j in self.indices_R], dim=dim)
        return outL, outR    

    def get_n_params_a_model(self):
        return len(self.get_a_model_params())

    def get_models_params(self):
        p = torch.cat([ params.flatten() for params in self.weights] )

        return p.cpu().flatten().numpy()

    def get_a_model_params(self):
        p = torch.cat([ params.flatten() for params in self.weights[0]] )

        return p.cpu().flatten().numpy()
    
    def set_models_params(self, flat_params):
        flat_params = torch.from_numpy(flat_params)
        # print('flat_params: ', flat_params.shape)

        popsize, basis, num_out = self.weights.shape
        self.weights = flat_params.reshape(popsize, basis, num_out).cuda()

    def set_a_model_params(self, flat_params):
        flat_params = torch.from_numpy(flat_params)
        # print('flat_params: ', flat_params.shape)

        popsize, basis, num_out = self.weights.shape
        # print('flat_params.repeat(popsize, 1, 1): ', flat_params.repeat(popsize, 1, 1).shape)
        self.weights = flat_params.repeat(popsize, 1, 1).reshape(popsize, basis, num_out).cuda()
            
    
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