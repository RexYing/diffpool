import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

# GCN basic operation
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        self.bias = nn.Parameter(torch.FloatTensor(output_dim).cuda())
    def forward(self, x, adj):
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y,self.weight) + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
            #print(y[0][0])
        return y

class GcnEncoderGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            pred_hidden_dims=[], concat=True):
        super(GcnEncoderGraph, self).__init__()
        self.concat = concat
        add_self = not concat
        self.num_layers = num_layers

        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
                input_dim, hidden_dim, embedding_dim, num_layers, add_self)
        self.act = nn.ReLU()

        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim
        self.pred_model = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims, label_dim)
            
        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                m.bias.data = init.constant(m.bias.data, 0.0)

    def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, num_layers, add_self):
        conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, add_self=add_self)
        conv_block = nn.ModuleList(
                [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, add_self=add_self) 
                 for i in range(num_layers-2)])
        conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self)
        return conv_first, conv_block, conv_last

    def build_pred_layers(self, pred_input_dim, pred_hidden_dims, label_dim):

        if len(pred_hidden_dims) == 0:
            pred_model = nn.Linear(pred_input_dim, label_dim)
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                pred_layers.append(self.act)
                pred_input_dim = pred_dim
            pred_layers.append(nn.Linear(pred_dim, label_dim))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model

    def gcn_forward(self, x, adj, conv_first, conv_block, conv_last):
        x = self.conv_first(x, adj)
        x = self.act(x)
        x_all = [x]
        #out_all = []
        #out, _ = torch.max(x, dim=1)
        #out_all.append(out)
        for i in range(len(conv_block)):
            x = self.conv_block[i](x,adj)
            x = self.act(x)
            x_all.append(x)
            #out,_ = torch.max(x, dim=1)
            #out = torch.sum(x, dim=1)
            #out_all.append(out)
        x = self.conv_last(x,adj)
        x_all.append(x)
        # x_tensor: [batch_size x num_nodes x embedding]
        x_tensor = torch.cat(x_all, dim=2)
        print('xtensor: ', x_tensor.size())
        return x_tensor

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
            #out = torch.sum(x, dim=1)
            out_all.append(out)
        x = self.conv_last(x,adj)
        #x = self.act(x)
        out, _ = torch.max(x, dim=1)
        #out = torch.sum(x, dim=1)
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
        #return F.binary_cross_entropy(F.sigmoid(pred[:,0]), label.float())


class SoftPoolingGcnEncoder(GcnEncoderGraph):
    def __init__(self, max_num_nodes, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            assign_hidden_dim, assign_ratio=0.25, assign_num_layers=-1,
            pred_hidden_dims=[], concat=True):

        super(SoftPoolingGcnEncoder, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim,
                num_layers, pred_hidden_dims=pred_hidden_dims, concat=concat)
        add_self = not concat
        
        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
                input_dim, hidden_dim, embedding_dim, num_layers, add_self)
        self.conv_first_after_pool = GraphConv(input_dim=self.pred_input_dim, output_dim=hidden_dim, add_self=add_self)

        # assignment
        assign_dim = int(max_num_nodes * assign_ratio)
        if assign_num_layers == -1:
            assign_num_layers = num_layers
        self.assign_conv_first, self.assign_conv_block, self.assign_conv_last = self.build_conv_layers(
                input_dim, assign_hidden_dim, assign_dim, assign_num_layers, add_self)
        assign_pred_input_dim = assign_hidden_dim * (num_layers - 1) + assign_dim if concat else assign_dim
        self.assign_pred = self.build_pred_layers(assign_pred_input_dim, pred_hidden_dims, assign_dim)


    def forward(self, x, adj):
        assign_tensor = self.gcn_forward(x, adj, 
                self.assign_conv_first, self.assign_conv_block, self.assign_conv_last)
        # [batch_size x num_nodes x next_lvl_num_nodes]
        assign_tensor = self.assign_pred(assign_tensor)
        # [batch_size x num_nodes x embedding_dim]
        embedding_tensor = self.gcn_forward(x, adj,
                self.conv_first, self.conv_block, self.conv_last)

        # update pooled features and adj matrix
        x = torch.matmul(torch.transpose(assign_tensor, 1, 2), embedding_tensor)
        adj = torch.transpose(assign_tensor, 1, 2) @ adj @ assign_tensor
        
        embedding_tensor = self.gcn_forward(x, adj, 
                self.conv_first_after_pool, self.conv_block, self.conv_last)

        out = torch.max(embedding_tensor, dim=1)
        ypred = self.pred_model(out)
        return ypred



