import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

# GCN basic operation
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        self.bias = nn.Parameter(torch.FloatTensor(output_dim).cuda())
        # self.act = nn.ReLU()
    def forward(self, x, adj):
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y,self.weight) + self.bias
        return y

class GcnEncoderGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            pred_hidden_dim=None, concat=True):
        super(GcnEncoderGraph, self).__init__()
        self.concat = concat
        add_self = not concat
        self.num_layers = num_layers
        self.conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, add_self=add_self)
        self.conv_block = nn.ModuleList(
                [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, add_self=add_self) 
                 for i in range(num_layers-2)])
        self.conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self)
        self.act = nn.ReLU()

        if concat:
            pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            pred_input_dim = embedding_dim

        if pred_hidden_dim is None:
            pred_hidden_dim = embedding_dim // 2
            
        self.pred_model = nn.Sequential(
                nn.Linear(pred_input_dim, pred_hidden_dim),
                self.act,
                nn.Linear(pred_hidden_dim, label_dim))
        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                m.bias.data = init.constant(m.bias.data, 0.0)
                # init_range = np.sqrt(6.0 / (m.input_dim + m.output_dim))
                # m.weight.data = torch.rand([m.input_dim, m.output_dim]).cuda()*init_range
                # print('find!')
                
    def forward(self, x, adj):
        x = self.conv_first(x, adj)
        x = self.act(x)
        out_all = []
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        for i in range(self.num_layers-2):
            x = self.conv_block[i](x,adj)
            x = self.act(x)
            out,_ = torch.max(x, dim=1)
            out_all.append(out)
        x = self.conv_last(x,adj)
        x = self.act(x)
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out
        ypred = self.pred_model(output)
        #print(output.size())
        return ypred

    def loss(self, pred, label):
        # softmax + CE
        return F.cross_entropy(pred, label, size_average=True)


