import torch
import torch.nn as nn
from torch.autograd import Variable

import argparse
import numpy as np
import sklearn.metrics as metrics
import os
import random
import time

import encoders
import gen.feat as featgen
import gen.data as datagen
from graph_sampler import GraphSampler
import load_data
import util

def evaluate(dataset, model, args, name='Validation', max_num_examples=None):
    model.eval()

    labels = []
    preds = []
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).cuda()
        h0 = Variable(data['feats'].float()).cuda()

        labels.append(data['label'].long().numpy())
        ypred = model(h0, adj)
        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())

        if max_num_examples is not None:
            if (batch_idx+1)*args.batch_size > max_num_examples:
                break

    labels = np.hstack(labels)
    preds = np.hstack(preds)
    
    print(name, " F1:", metrics.f1_score(labels, preds, average="micro"))
    print(name, " prec:", metrics.precision_score(labels, preds))
    print(name, " recall:", metrics.recall_score(labels, preds))

def train(dataset, model, args, same_feat=True):
    
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=0.001)
    iter = 0
    for epoch in range(args.num_epochs):
        begin_time = time.time()
        avg_loss = 0.0
        model.train()
        print('Epoch: ', epoch)
        for batch_idx, data in enumerate(dataset):
            model.zero_grad()
            adj = Variable(data['adj'].float(), requires_grad=False).cuda()
            h0 = Variable(data['feats'].float(), requires_grad=False).cuda()
            label = Variable(data['label'].long()).cuda()
            ypred = model(h0, adj)
            loss = model.loss(ypred, label)
            loss.backward()
            optimizer.step()
            iter += 1
            avg_loss += loss.data[0]
            if iter % 10 == 0:
                #print(label)
                #print(ypred)
                print('Iter: ', iter, ', loss: ', loss.data[0])
        avg_loss /= batch_idx + 1
        elapsed = time.time() - begin_time
        print('Avg loss: ', avg_loss, '; epoch time: ', elapsed)
        evaluate(dataset, model, args, name='Train', max_num_examples=100)

    return model

def prepare_data(graphs, args):
    random.shuffle(graphs)

    train_idx = int(len(graphs) * args.train_ratio)
    train_graphs = graphs[:train_idx]
    #train_graphs = graphs[:10]
    test_graphs = graphs[train_idx:]
    print('Num training graphs: ', len(train_graphs), 
          '; Num testing graphs: ', len(test_graphs))

    print('Number of graphs: ', len(graphs))
    print('Number of edges: ', sum([G.number_of_edges() for G in graphs]))
    print('Max, avg, std of graph size: ', 
            max([G.number_of_nodes() for G in graphs]), ', '
            "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])), ', '
            "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])))

    # minibatch
    dataset_sampler = GraphSampler(train_graphs, normalize=False)
    train_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.num_workers)

    dataset_sampler = GraphSampler(test_graphs, normalize=False)
    test_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers)

    return train_dataset_loader, test_dataset_loader

def synthetic_task1(args, export_graphs=False):

    # data
    graphs1 = datagen.gen_ba(range(40, 60), range(4, 5), 500, 
            featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float)))
    for G in graphs1:
        G.graph['label'] = 0
    if export_graphs:
        util.draw_graph_list(graphs1[:16], 4, 4, 'figs/ba')

    graphs2 = datagen.gen_2community_ba(range(20, 30), range(4, 5), 500, 0.3, 
            [featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))])
    for G in graphs2:
        G.graph['label'] = 1
    if export_graphs:
        util.draw_graph_list(graphs2[:16], 4, 4, 'figs/ba2')

    graphs = graphs1 + graphs2

    max_node = max([G.number_of_nodes() for G in graphs])
    print('max_node', max_node)
    train_dataset, test_dataset = prepare_data(graphs, args)
    if args.model=='flex':
        model = encoders.GcnEncoderGraph_flex(args.input_dim, args.hidden_dim,max_node//2, args.output_dim, 2,
                args.num_gc_layers).cuda()
    else:
        model = encoders.GcnEncoderGraph(args.input_dim, args.hidden_dim, args.output_dim, 2,
                                         args.num_gc_layers).cuda()
    train(train_dataset, model, args)
    evaluate(train_dataset, model, args, "Train")
    evaluate(test_dataset, model, args, "Validation")

def benchmark_task(args, feat=None):
    graphs = load_data.read_graphfile(args.datadir, args.bmname, max_nodes=args.max_nodes)
    if feat == 'node-label':
        for G in graphs:
            for u in G.nodes():
                G.node[u]['feat'] = G.node[u]['label']
    else:
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)

    max_node = max([G.number_of_nodes() for G in graphs])
    print('max_node',max_node)
    train_dataset, test_dataset = prepare_data(graphs, args)
    if args.model=='flex':
        model = encoders.GcnEncoderGraph_flex(args.input_dim, args.hidden_dim, args.output_dim,max_node//2, 2, args.num_gc_layers).cuda()
    else:
        model = encoders.GcnEncoderGraph(args.input_dim, args.hidden_dim, args.output_dim, 2, args.num_gc_layers).cuda()
    train(train_dataset, model, args)
    evaluate(test_dataset, model, args, 'Validation')
    
def arg_parse():
    parser = argparse.ArgumentParser(description='GraphPool arguments.')
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument('--dataset', dest='dataset', 
            help='Input dataset.')
    benchmark_parser = io_parser.add_argument_group()
    benchmark_parser.add_argument('--datadir', dest='datadir',
            help='Directory where benchmark is located')
    benchmark_parser.add_argument('--bmname', dest='bmname',
            help='Name of the benchmark dataset')
    benchmark_parser.add_argument('--mdname', dest='mdname',
                                  help='Name of the model')

    parser.add_argument('--cuda', dest='cuda',
            help='CUDA.')
    parser.add_argument('--max-nodes', dest='max_nodes', type=int,
            help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.')
    parser.add_argument('--batch-size', dest='batch_size', type=int,
            help='Batch size.')
    parser.add_argument('--epochs', dest='num_epochs', type=int,
            help='Number of epochs to train.')
    parser.add_argument('--train-ratio', dest='train_ratio', type=float,
            help='Ratio of number of graphs training set to all graphs.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
            help='Number of workers to load data.')
    parser.add_argument('--feature', dest='feature_type',
            help='Feature used for encoder. Can be: id, deg')
    parser.add_argument('--input-dim', dest='input_dim', type=int,
            help='Input feature dimension')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int,
            help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', type=int,
            help='Output dimension')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int,
            help='Number of graph convolution layers before each pooling')

    parser.set_defaults(dataset='synthetic1',
                        mdname='normal',
                        max_nodes = 1000,
                        cuda='1',
                        feature_type='default',
                        lr=0.001,
                        batch_size=20,
                        num_epochs=20,
                        train_ratio=0.8,
                        num_workers=1,
                        input_dim=10,
                        hidden_dim=20,
                        output_dim=30,
                        num_gc_layers=4,
                       )
    return parser.parse_args()

def main():
    prog_args = arg_parse()

    os.environ['CUDA_VISIBLE_DEVICES'] = prog_args.cuda
    print('CUDA', prog_args.cuda)

    if prog_args.datadir is not None:
        benchmark_task(prog_args)
    elif prog_args.dataset is not None:
        synthetic_task1(prog_args)

if __name__ == "__main__":
    main()

