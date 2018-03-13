import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import time

import gen.feat as featgen
import gen.data as datagen
import util

#class ClassifyGraphs():

def syntheticTask1(input_feat_dim=10):

    graphs1 = datagen.gen_ba(range(40, 60), range(4, 5), 20, 
                             featgen.ConstFeatureGen(np.ones(input_feat_dim)))
    for G in graphs1:
        G.graph['label'] = 0
    util.draw_graph_list(graphs1[:16], 4, 4, 'figs/ba')

    graphs2 = datagen.gen_2community_ba(range(20, 30), range(4, 5), 20, 0.3, 
                                        [featgen.ConstFeatureGen(np.ones(input_feat_dim))])
    for G in graphs2:
        G.graph['label'] = 1
    util.draw_graph_list(graphs2[:16], 4, 4, 'figs/ba2')

def main():
    syntheticTask1()

if __name__ == "__main__":
    main()

