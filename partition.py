import networkx
import numpy as np

def partition(embeddings):
    ''' Compute a partition of embeddings, where each partition is pooled together.
    Args:
        embeddings: N-by-D matrix, where N is the number of node embeddings, and D
            is the embedding dimension.
    '''
    dist = np.dot(embeddings)
    
def kruskal(adj):
    # initialize MST
    MST = set()
    edges = set()
    num_nodes = adj.shape[0]
    # collect all edges from graph G
    for j in range(num_nodes):
        for k in range(num_nodes):
            if G.graph[j][k] != 0 and (k, j) not in edges:
                edges.add((j, k))
    # sort all edges in graph G by weights from smallest to largest
    sorted_edges = sorted(edges, key=lambda e:G.graph[e[0]][e[1]])
    uf = UF(G.vertices)
    for e in sorted_edges:
        u, v = e
        # if u, v already connected, abort this edge
        if uf.connected(u, v):
            continue
        # if not, connect them and add this edge to the MST
        uf.union(u, v)
        MST.add(e)
    return MST

