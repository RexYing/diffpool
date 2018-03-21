import torch
import torch.nn as nn
from torch.autograd import Variable

import argparse
import numpy as np
import os
import random
import time

import encoders
import gen.feat as featgen
import gen.data as datagen
from graph_sampler import GraphSampler
import util


def synthetic_task_train(dataset, args, same_feat=True):

    model = encoders.GcnEncoderGraph(args.input_dim, args.hidden_dim, args.output_dim, 2, 2).cuda()
    model.train()
    
    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, model.parameters()), lr=0.001)
    times = []
    for batch_idx, data in enumerate(dataset):
        model.zero_grad()
        adj = Variable(data['adj'].float(), requires_grad=False).cuda()
        h0 = Variable(data['feats'].float()).cuda()
        label = Variable(data['label'].long()).cuda()
        start_time = time.time()
        ypred = model(h0, adj)
        loss = model.loss(ypred, label)
        print(loss)
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print('Iter: ', batch_idx, ', loss: ', loss.data[0])

    #print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1),
    #    average="micro"))
    #print("Average batch time:", np.mean(times))


def synthetic_task1(args, export_graphs=False):

    # data
    graphs1 = datagen.gen_ba(range(40, 60), range(4, 5), 500, 
                             featgen.ConstFeatureGen(np.ones(args.input_dim)))
    for G in graphs1:
        G.graph['label'] = 0
    if export_graphs:
        util.draw_graph_list(graphs1[:16], 4, 4, 'figs/ba')

    graphs2 = datagen.gen_2community_ba(range(20, 30), range(4, 5), 500, 0.3, 
                                        [featgen.ConstFeatureGen(np.ones(args.input_dim))])
    for G in graphs2:
        G.graph['label'] = 1
    if export_graphs:
        util.draw_graph_list(graphs2[:16], 4, 4, 'figs/ba2')

    graphs = graphs1 + graphs2
    random.shuffle(graphs)
    print(len(graphs))

    # minibatch
    dataset_sampler = GraphSampler(graphs)
    dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.num_workers)
    synthetic_task_train(dataset_loader, args)
    
def arg_parse():
    parser = argparse.ArgumentParser(description='GraphPool arguments.')
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument('--dataset', dest='dataset', 
            help='Input dataset.')

    parser.add_argument('--cuda', dest='cuda',
            help='CUDA.')
    parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
            help='Batch size.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
            help='Number of workers to load data.')
    parser.add_argument('--feature', dest='feature_type',
            help='Feature used for encoder. Can be: id, deg')
    parser.add_argument('--input_dim', dest='input_dim', type=int,
            help='Input feature dimension')
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int,
            help='Hidden dimension')
    parser.add_argument('--output_dim', dest='output_dim', type=int,
            help='Output dimension')

    parser.set_defaults(cuda='1',
                        feature_type='default',
                        lr=0.001,
                        batch_size=10,
                        num_workers=1,
                        input_dim=10,
                        hidden_dim=20,
                        output_dim=30,
                       )
    return parser.parse_args()

def main():
    prog_args = arg_parse()

    os.environ['CUDA_VISIBLE_DEVICES'] = prog_args.cuda
    print('CUDA', prog_args.cuda)

    synthetic_task1(prog_args)

if __name__ == "__main__":
    main()

