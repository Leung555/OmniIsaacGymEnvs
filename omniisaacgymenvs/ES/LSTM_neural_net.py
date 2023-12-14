import torch

class LSTMs():

    def __init__(self, popsize, arch): #in_channels, hid_size, out_channels):
        super(LSTMs, self).__init__()


        self.arch = arch
        in_channels = arch[0]
        hid_size = arch[1]
        out_channels = arch[2]
        
        self.popsize = popsize

        self.in_channels = in_channels
        self.hid_size = hid_size
        self.out_channels = out_channels

        self.hidden_state = torch.zeros(popsize,hid_size, 1).cuda()
        self.cell_state = torch.zeros(popsize,hid_size, 1).cuda()

    def forward(self, inp):
        with torch.no_grad():        

            x = torch.cat((inp.unsqueeze(-1), self.hidden_state), dim=1)

            f = torch.sigmoid( torch.einsum('lbn,lbc->lnc', self.Wf.float(), x.float())+self.Bf.float())
            i = torch.sigmoid( torch.einsum('lbn,lbc->lnc', self.Wi.float(), x.float())+self.Bi.float())

            c = torch.tanh( torch.einsum('lbn,lbc->lnc', self.Wc.float(), x.float())+self.Bc.float())

            self.cell_state = f * self.cell_state + i * c

            o = torch.sigmoid( torch.einsum('lbn,lbc->lnc', self.Wo.float(), x.float())+self.Bo.float())

            self.hidden_state = o * torch.tanh(self.cell_state)

            x = torch.cat((inp.unsqueeze(-1), self.hidden_state), dim=1)
            out = torch.tanh( torch.einsum('lbn,lbc->lnc', self.Wout.float(), x.float())) ## (popsize, out_size, 1)

        return out.squeeze_()


    def set_params(self, pop):
        
        ##population shape: (popsize, n_total_params)


        n_i, n_h, n_o = self.arch
        popsize = pop.shape[0]

        m = 0
        self.Wf = pop[:,m:m+(n_i+n_h)*n_h].reshape(popsize, n_i+n_h, n_h).cuda()
        m += (n_i+n_h)*n_h
        self.Wi = pop[:,m:m+(n_i+n_h)*n_h].reshape(popsize, n_i+n_h, n_h).cuda()
        m += (n_i+n_h)*n_h
        self.Wc = pop[:,m:m+(n_i+n_h)*n_h].reshape(popsize, n_i+n_h, n_h).cuda()
        m += (n_i+n_h)*n_h
        self.Wo = pop[:,m:m+(n_i+n_h)*n_h].reshape(popsize, n_i+n_h, n_h).cuda()
        m += (n_i+n_h)*n_h
        self.Wout = pop[:,m:m+(n_i+n_h)*n_o].reshape(popsize, n_i+n_h, n_o).cuda()
        m += (n_i+n_h)*n_o

        self.Bf = pop[:,m:m+n_h].unsqueeze(-1).cuda()
        m += n_h
        self.Bi = pop[:,m:m+n_h].unsqueeze(-1).cuda()
        m += n_h
        self.Bc = pop[:,m:m+n_h].unsqueeze(-1).cuda()
        m += n_h
        self.Bo = pop[:,m:m+n_h].unsqueeze(-1).cuda()
        m += n_h

    def get_n_params(self):
        n_i, n_h, n_o = self.arch
        return (n_i+n_h)*n_h * 4 + (n_i+n_h)*n_o + n_h * 4

    def get_params_a_model(self):
        p = torch.cat([ self.Wf[0].flatten()]  
                     +[ self.Wi[0].flatten()] 
                     +[ self.Wc[0].flatten()]
                     +[ self.Wo[0].flatten()]
                     +[ self.Wout[0].flatten()]
                     +[ self.Bf[0].flatten()]
                     +[ self.Bi[0].flatten()]
                     +[ self.Bc[0].flatten()]
                     +[ self.Bo[0].flatten()]
                     )
        return p.flatten().cpu().numpy()