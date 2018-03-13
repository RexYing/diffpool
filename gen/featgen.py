import abc
import networkx as nx
import numpy as np
import random

class FeatureGen(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def gen_node_features():
        pass

class UniformFeatureGen(FeatureGen):
    def __init__(self, val):
        self.val = val

    def gen_node_features(G):
        feat = np.ones(G.number_of_nodes(), 1) * self.val
        feat_dict = {i:feat[i] for i in range(feat.shape[0])}
        nx.set_node_attributes(G, 'feat', feat_dict)

class GaussianFeatureGen(FeatureGen):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def gen_node_features(G):
        feat = np.random.multivariate_normal(mu, sigma, G.number_of_nodes())
        feat_dict = {i:feat[i] for i in range(feat.shape[0])}
        nx.set_node_attributes(G, 'feat', feat_dict)

