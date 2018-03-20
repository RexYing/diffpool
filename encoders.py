import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

# GCN basic operation
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        # self.act = nn.ReLU()
    def forward(self, x, adj):
        y = torch.matmul(adj, x)
        y = torch.matmul(y,self.weight)
        return y

class GcnEncoderGraph(nn.Module):
    def __init__(self,input_dim, hidden_dim, output_dim, num_layers):
        super(GCN_encoder_graph, self).__init__()
        self.num_layers = num_layers
        self.conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim)
        self.conv_block = nn.ModuleList([GraphConv(input_dim=hidden_dim, output_dim=hidden_dim) for i in range(num_layers-2)])
        self.conv_last = GraphConv(input_dim=hidden_dim, output_dim=output_dim)
        self.act = nn.ReLU()
        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                # init_range = np.sqrt(6.0 / (m.input_dim + m.output_dim))
                # m.weight.data = torch.rand([m.input_dim, m.output_dim]).cuda()*init_range
                # print('find!')
    def forward(self, x, adj, concat_all_layers=False):
        x = self.conv_first(x, adj)
        x = self.act(x)
        out_all = []
        out, _ = torch.max(x, dim=1, keepdim=True)
        out_all.append(out)
        for i in range(self.num_layers-2):
            x = self.conv_block[i](x,adj)
            x = self.act(x)
            out,_ = torch.max(x, dim=1, keepdim = True)
            out_all.append(out)
        x = self.conv_last(x,adj)
        x = self.act(x)
        out, _ = torch.max(x, dim=1, keepdim = True)
        out_all.append(out)
        if concat_all_layers:
            output = torch.cat(out_all, dim = 1)
        else:
            output = out
        output = output.permute(1,0,2)
        # print(out)
        return output

