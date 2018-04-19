import networkx as nx
import numpy as np
import scipy as sc
import os

def read_graphfile(datadir, dataname):
    ''' Read data from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
        graph index starts with 1 in file

    Returns:
        List of networkx objects with graph and node labels
    '''
    prefix = os.path.join(datadir, dataname, dataname)
    filename_graph_indic = prefix + '_graph_indicator.txt'
    # index of graphs that a given node belongs to
    graph_indic={}
    with open(filename_graph_indic) as f:
        i=1
        for line in f:
            line=line.strip("\n")
            graph_indic[i]=int(line)
            i+=1

    filename_nodes=prefix + '_node_labels.txt'
    node_labels=[]
    with open(filename_nodes) as f:
        for line in f:
            line=line.strip("\n")
            node_labels+=[int(line) - 1]
    
    filename_graphs=prefix + '_graph_labels.txt'
    graph_labels=[]
    with open(filename_graphs) as f:
        for line in f:
            line=line.strip("\n")
            graph_labels.append(int(line) - 1)
    
    filename_adj=prefix + '_A.txt'
    adj_list={i:[] for i in range(1,len(graph_labels)+1)}    
    index_graph={i:[] for i in range(1,len(graph_labels)+1)}
    num_edges = 0
    with open(filename_adj) as f:
        for line in f:
            line=line.strip("\n").split(",")
            e0,e1=(int(line[0].strip(" ")),int(line[1].strip(" ")))
            adj_list[graph_indic[e0]].append((e0,e1))
            index_graph[graph_indic[e0]]+=[e0,e1]
            num_edges += 1
    for k in index_graph.keys():
        index_graph[k]=[u-1 for u in set(index_graph[k])]
    print('Number of graphs loaded: ', len(graph_labels))
    print('Number of edges loaded: ', num_edges)

    graphs=[None] * len(adj_list)
    for i in range(1,1+len(adj_list)):
        # indexed from 1 here
        G=nx.from_edgelist(adj_list[i])
      
        # add features and labels
        G.graph['label'] = graph_labels[i-1]
        for u in G.nodes():
            G.node[u]['label'] = node_labels[u-1]

        graphs[i-1] = G

        # relabeling
        mapping={}
        it=0
        if float(nx.__version__)<2.0:
            for n in graphs[i-1].nodes():
                mapping[n]=it
                it+=1
        else:
            for n in graphs[i-1].nodes:
                mapping[n]=it
                it+=1
            
        # indexed from 0
        graphs[i-1] = nx.relabel_nodes(graphs[i-1], mapping)
    return graphs

