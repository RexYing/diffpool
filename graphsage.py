import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

class SupervisedGraphSage(nn.Module):
    ''' GraphSage embeddings
    '''

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(enc.embed_dim, num_classes))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = embeds.mm(self.weight)
        return scores

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(nn.softmax(scores), labels.squeeze())

