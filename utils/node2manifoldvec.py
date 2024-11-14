from typing import Optional

import torch
import torch_geometric
import geoopt 



class ManifoldEmbedding(torch.nn.Module):
    """ 
        Pytorch Module for hyperbolic shallow Embeddings
    
        Attributes
        ----------
        num_embeddings: int
            Number of embeddings.
        embedding_dim: int
            Size of the embeddings.
        manifold: geoopt.Manifold
            Used manifold
        weight: geoopt.ManifoldParameter
            Model parameters on the used manifold.

        Methods
        -------
        init(num_embeddings, embedding_dim, manifold)
            Initialize the model.
        reset_parameters()
            Initialize the parameters with a normal distribution on the manifold.
        forward(x)
            Return the embeddings on the given index.
    """
    def __init__(self, num_embeddings, embedding_dim, manifold):
        super(ManifoldEmbedding, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.manifold = manifold

        self.reset_parameters()

    def reset_parameters(self):
        self.weight = self.manifold.random_normal((self.num_embeddings, self.embedding_dim),
                                                mean = self.manifold.origin((self.num_embeddings, self.embedding_dim), device = self.manifold.k.device), 
                                                std = 1.0)

        self.weight = geoopt.ManifoldParameter(self.weight) 

    def forward(self, x):
        return self.weight[x]



class Node2ManifoldVec(torch_geometric.nn.Node2Vec):
    """
        Hyperbolic Node2Vec model: Overriding the pytorch-geometric implementation of node2vec: https://github.com/pyg-team/pytorch_geometric

        Attributes
        ----------
        manifold: geoopt.base.Manifold
            Manifold of the learned embeddings.
        edge_index: torch.Tensor 
            The edge indices.
        embedding_dim: int
            The size of each embedding vector.
        walk_length: int
            The walk length.
        context_size: int
            The actual context size which is considered for positive samples. 
            This parameter increases the effective sampling rate by reusing samples across different source nodes.
        walks_per_node: int
            The number of walks to sample for each node. (default: 1)
        p: float: 
            Likelihood of immediately revisiting a node in the walk. (default: 1)
        q: float
            Control parameter to interpolate between breadth-first strategy and depth-first strategy (default: 1)
        num_negative_samples: int
            The number of negative samples to use for each positive sample. (default: 1)
        num_nodes: int
            The number of nodes. (default: None)

        Overrided Methods
        -------
        init(manifold, edge_index, embedding_dim, walk_length, context_size, walks_per_node, p, q, num_negative_samples, num_nodes)
            Initializing the model with embeddings on the given manifold. (Sparse gradients are currently not supported.)
        loss(pos_rw, neg_rw)
            Computes the loss given positive and negative random walks.
    """

    def __init__(self, manifold, edge_index: torch.Tensor, embedding_dim: int, walk_length: int, context_size: int, walks_per_node: int = 1,
                 p: float = 1.0, q: float = 1.0, num_negative_samples: int = 1, num_nodes: Optional[int] = None):
        super(Node2ManifoldVec, self).__init__(edge_index, embedding_dim, walk_length, context_size, walks_per_node, 
                                               p, q, num_negative_samples, num_nodes, sparse = False)
        
        self.manifold = manifold
        self.embedding = ManifoldEmbedding(self.num_nodes, embedding_dim, manifold)

    @torch.jit.export
    def loss(self, pos_rw: torch.Tensor, neg_rw: torch.Tensor) -> torch.Tensor:
        # Positive loss.
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()
        h_start = self.embedding(start).view(pos_rw.size(0), 1, self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(pos_rw.size(0), -1, self.embedding_dim)

        if str(self.embedding.manifold).split(" ")[0] == "Lorentz":
            dist = (-2 * self.embedding.manifold.k -2 * self.embedding.manifold.inner(None, h_start, h_rest)).view(-1)
        else: 
            dist = self.embedding.manifold.dist(h_start, h_rest).view(-1)
        #sim = torch.exp(-dist)  #pos_loss = dist
        #pos_loss = -torch.log(sim).mean()
        pos_loss = dist.mean()
 

        # Negative loss.
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()
        h_start = self.embedding(start).view(neg_rw.size(0), 1, self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(neg_rw.size(0), -1, self.embedding_dim)

        if str(self.embedding.manifold).split(" ")[0] == "Lorentz":
            dist = (-2 * self.embedding.manifold.k -2 * self.embedding.manifold.inner(None, h_start, h_rest)).view(-1)
        else: 
            dist = self.embedding.manifold.dist(h_start, h_rest).view(-1)
        eps = 1e-10 if torch.get_default_dtype() == torch.float64 else 1e-5
        sim = torch.exp(-dist - eps)
        neg_loss = -torch.log(1-sim).mean()

        return pos_loss + neg_loss