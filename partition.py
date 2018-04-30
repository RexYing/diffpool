import networkx
import numpy as np

def partition(embeddings):
    ''' Compute a partition of embeddings, where each partition is pooled together.
    Args:
        embeddings: N-by-D matrix, where N is the number of node embeddings, and D
            is the embedding dimension.
    '''
    dist = np.dot(embeddings)
