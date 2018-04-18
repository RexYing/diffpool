import numpy as np
import scipy as sc
import networkx as nx
from graph_functions import *


def read_graphfile(datadir, dataname):
    filename_graph_indic=datadir+dataname+"/"+dataname+"_graph_indicator.txt"
    graph_indic={}
  
    with open(filename_graph_indic) as f:
            i=1
            for line in f:
                line=line.strip("\n")
                graph_indic[i]=int(line)
                i+=1
    filename_nodes=datadir+dataname+"/"+dataname+"_node_labels.txt"
    node_labels=[]
    with open(filename_nodes) as f:
        for line in f:
            line=line.strip("\n")
            node_labels+=[int(line)]
    
    filename_graphs=datadir+dataname+"/"+dataname+"_graph_labels.txt"
    graph_labels=[]
    with open(filename_graphs) as f:
        for line in f:
            line=line.strip("\n")
            graph_labels.append(int(line))
    
    adj_list={i:[] for i in range(1,len(graph_labels)+1)}    
    filename_adj="../data/"+dataname+"/"+dataname+"_A.txt"
    gen_list=[]
    index_graph={k:[] for k in range(1,1+len(graph_labels))}
    with open(filename_adj) as f:
        for line in f:
            line=line.strip("\n").split(",")
            gen_list.append((int(line[0].strip(" ")),int(line[1].strip(" "))))
            e0,e1=(int(line[0].strip(" ")),int(line[1].strip(" ")))
            adj_list[graph_indic[e0]].append((e0,e1))
            index_graph[graph_indic[e0]]+=[e0,e1]
    for k in index_graph.keys():
        index_graph[k]=[u-1 for u in set(index_graph[k])]
    #for i in range(1,len(graph_labels)+1):
    #    adj_list[i]=[(e[0],e[1]) for e in gen_list if graph_indic[e[0]]==i or graph_indic[e[1]]==i]
    print "check ", np.sum([len(adj_list[i]) for i in adj_list.keys()]),len(gen_list)
    graphs={}
    for i in range(1,1+len(adj_list)):
        G=nx.from_edgelist(adj_list[i])
      
        # add features and labels
        G.graph['label'] = graph_labels[i]
        

        graphs[i] = G

        # relabeling
        mapping={}
        it=0
        if float(nx.__version__)<2.0:
            for n in graphs[i].nodes():
                mapping[n]=it
                it+=1
        else:
            for n in graphs[i].nodes:
                mapping[n]=it
                it+=1
            
            
        graphs[i] = nx.relabel_nodes(graphs[i], mapping)
    return graphs,adj_list, np.array(node_labels),np.array(graph_labels),graph_indic,index_graph

def load_data(datadir, dataname):
    
    graphs,adj_list, node_labels,graph_labels,graph_indic,index_graph = read_graphfile(datadir, dataname)
    graph_indic_array=np.sort([v for k,v in graph_indic.iteritems()])
    
    ind=range(1,1+len(graphs))
    np.random.shuffle(ind)
    training_set=ind[:1000]
    test_set=ind[1000:]
    
    n_classes=np.max(node_labels)
    train_features=[None]*len(training_set)
    it=0
    for i in training_set:
        cand_nodes=index_graph[i]
        train_features[it]=np.zeros((len(cand_nodes),n_classes+1))
        train_features[it][np.arange(len(cand_nodes)), node_labels[cand_nodes]]=1
        it+=1
      
    
    test_features=[None]*len(test_set)
    it=0
    for i in test_set:
        cand_nodes=np.where(graph_indic_array==i)[0]
        #test_features[it]=np.eye(n_classes+1)[ node_labels[cand_nodes]]
        test_features[it]=np.zeros((len(cand_nodes),n_classes+1))
        test_features[it][np.arange(len(cand_nodes)), node_labels[cand_nodes]]=1
        it+=1
    
    
    train_graphs,train_targets=[graphs[i] for i in training_set], [graph_labels[i-1] for i in training_set] 
    test_graphs,test_targets=[graphs[i] for i in test_set], [graph_labels[i-1] for i in test_set]  
    return train_graphs,np.array(train_targets),train_features, test_graphs,np.array(test_targets),test_features

