# diffpool

This is the repo for Hierarchical Graph Representation Learning with Differentiable Pooling (NeurIPS 2018)

Recently, graph neural networks (GNNs) have revolutionized the field of graph
representation learning through effectively learned node embeddings, and achieved
state-of-the-art results in tasks such as node classification and link prediction.
However, current GNN methods are inherently flat and do not learn hierarchical
representations of graphs—a limitation that is especially problematic for the task
of graph classification, where the goal is to predict the label associated with an
entire graph. Here we propose DIFFPOOL, a differentiable graph pooling module
that can generate hierarchical representations of graphs and can be combined with
various graph neural network architectures in an end-to-end fashion. DIFFPOOL
learns a differentiable soft cluster assignment for nodes at each layer of a deep
GNN, mapping nodes to a set of clusters, which then form the coarsened input
for the next GNN layer. Our experimental results show that combining existing
GNN methods with DIFFPOOL yields an average improvement of 5–10% accuracy
on graph classification benchmarks, compared to all existing pooling approaches,
achieving a new state-of-the-art on four out of five benchmark data sets.


Paper link: https://arxiv.org/pdf/1806.08804.pdf

