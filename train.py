import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import time

import gen.feat as featgen
import gen.data as datagen
import util

#class ClassifyGraphs():

def synthetic_task_train(graphs, same_feat=True):
    feat_data, labels, adj_lists = load_cora()
    if same_feat:
        feat_data = graphs[0].node[0]['feat']
        feat_dim = feat_data.shape[0]
        features = nn.Embedding(feat_dim)
        features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=True)
        # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 1433, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 5
    enc2.num_samples = 5

    graphsage = SupervisedGraphSage(7, enc2)
#    graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in range(100):
        batch_nodes = train[:256]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print batch, loss.data[0]

    val_output = graphsage.forward(val) 
    print "Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro")
    print "Average batch time:", np.mean(times)


def synthetic_task1(input_feat_dim=10):

    # data
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

    graphs = graphs1 + graphs2
    synthetic_task_train(graphs)
    

def main():
    synthetic_task1()

if __name__ == "__main__":
    main()

