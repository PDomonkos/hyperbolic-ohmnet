import numpy as np

import torch

import networkx as nx 

###################################################################################
# Lorentz model utils
#
# Some of the code ode adopted from geoopt:
#   https://github.com/geoopt/geoopt/blob/master/geoopt/manifolds/lorentz/math.py
#   Apache License, Version 2.0
#   Copyright (c) 2018 Geoopt Developers
###################################################################################

@torch.jit.script 
def arcosh(x: torch.Tensor):
    dtype = x.dtype
    z = torch.sqrt(torch.clamp_min(x.double().pow(2) - 1.0, 1e-15))
    return torch.log(x + z).to(dtype)

def squared_lorentzian_distance_matrix(query_emb, ref_emb, k = 1):
    metric_tensor = np.eye(query_emb.shape[1])
    metric_tensor[0,0] = -1.0
    return np.clip(- 2 * k - 2 * np.matmul(np.matmul(query_emb, metric_tensor), ref_emb.T), 0, None)

def squared_lorentzian_distance(query_emb, ref_emb, manifold):
    return -2 * manifold.k -2 * manifold.inner(None,query_emb,ref_emb) 

def lorentzian_distance_matrix(query_emb, ref_emb, k):
    metric_tensor = np.eye(query_emb.shape[1])
    metric_tensor[0,0] = -1.0
    metric_tensor = torch.Tensor(metric_tensor)
    return torch.sqrt(k) * arcosh(-torch.matmul(torch.matmul(query_emb, metric_tensor), ref_emb.T) / k ) 

def lorentz_to_poincare(x, k, dim=-1):
    dn = x.size(dim) - 1
    return x.narrow(dim, 1, dn) / (x.narrow(-dim, 0, 1) + torch.sqrt(k))

def poincare_to_lorentz(x, k, dim=-1, eps=1e-6):
    x_norm_square = torch.sum(x * x, dim=dim, keepdim=True)
    res = (
        torch.sqrt(k)
        * torch.cat((1 + x_norm_square, 2 * x), dim=dim)
        / (1.0 - x_norm_square + eps)
    )
    return res

###################################################################################
# Lorentz cetroid minimizing the squared Lorentz distance
#
# Code is adapted from:
#   https://github.com/kschwethelm/HyperbolicCV/blob/main/code/lib/lorentz/manifold.py
#    MIT License
#    Copyright (c) 2023 Kristian Schwethelm
###################################################################################

def lorentz_centroid(x, manifold, w=None, eps=1e-8):
    if w is not None:
        avg = w.matmul(x)
    else:
        avg = x.mean(dim=-2)

    denom = (-manifold.inner(avg, avg, keepdim=True))
    denom = denom.abs().clamp_min(eps).sqrt()

    centroid = torch.sqrt(manifold.k) * avg / denom

    return centroid

###################################################################################
# Linear layer in a hyperboloid manifold
#
# Code is adapted from:
#   https://github.com/HazyResearch/hgcn/blob/master/layers/hyp_layers.py
###################################################################################

class HypLinear(torch.nn.Module):

    def __init__(self, in_features, out_features, manifold):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.manifold = manifold
        self.weight = torch.nn.Parameter(torch.empty(in_features, out_features))
        self.bias = torch.nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self): 
        torch.nn.init.xavier_uniform_(self.weight, gain=1.0)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        u = self.manifold.logmap0(x)
        u = u.narrow(-1, 1, x.size(-1) - 1)
        uw = u @ self.weight
        uw = torch.cat((torch.zeros((uw.size(0), 1)).to(self.weight.device), uw), axis = 1)
        xw = self.manifold.expmap0(uw)

        bu = torch.cat((torch.zeros(1).to(self.bias.device), self.bias))
        bu = self.manifold.transp0(xw, bu)
        xwb = self.manifold.expmap(xw, bu)
        return self.manifold.logmap0(xwb)[:, 1:]

###################################################################################
# Poincaré distance matrix
#
# Code is adapted from Poincaré Maps:
#   https://github.com/facebookresearch/PoincareMaps/blob/main/poincare_maps.py
#   Creative Commons Attribution-NonCommercial 4.0 International Public License
#   Copyright (c) 2017-present, Facebook, Inc.
###################################################################################

def euclidean_distance(x):
    nx = x.size(0)
    x = x.contiguous()
    
    x = x.view(nx, -1)

    norm_x = torch.sum(x ** 2, 1, keepdim=True).t()
    ones_x = torch.ones(nx, 1)

    xTx = torch.mm(ones_x, norm_x)
    xTy = torch.mm(x, x.t())
    
    d = (xTx.t() + xTx - 2 * xTy)
    d[d < 0] = 0

    return d

def poincare_distance(x):
    if torch.get_default_dtype() == torch.float64:
        x = torch.DoubleTensor(x)
    else:
        x = torch.FloatTensor(x)
    eps = 1e-5
    boundary = 1 - eps
    
    nx = x.size(0)
    x = x.contiguous()
    x = x.view(nx, -1)
    
    norm_x = torch.sum(x ** 2, 1, keepdim=True)
    sqdist = euclidean_distance(x) * 2    
    squnorm = 1 - torch.clamp(norm_x, 0, boundary)

    x = (sqdist / torch.mm(squnorm, squnorm.t())) + 1
    z = torch.sqrt(torch.pow(x, 2) - 1)
    
    dist_mat = torch.log(x + z)
    
    return dist_mat.detach().cpu().numpy()

###################################################################################
# Einstein midpoint in the Klein model.
# 
# Code adopted from:
#   Supplementary code for the paper "Hyperbolic Image Embeddings".
#   https://github.com/leymir/hyperbolic-image-embeddings/blob/master/hyptorch/pmath.py 
#   Apache License, Version 2.0
#   Copyright (c) 2019 Valentin Khrulkov
###################################################################################

def klein_2_poincare(x, k):
    denom = 1 + torch.sqrt(1 - k * x.pow(2).sum(-1, keepdim=True))
    return x / denom
 
def poincare_2_klein(x, k):
    denom = 1 + k * x.pow(2).sum(-1, keepdim=True)
    return 2 * x / denom
 
def lorenz_factor(x, *, k=1.0, dim=-1, keepdim=False):
    return 1 / torch.sqrt(1 - k * x.pow(2).sum(dim=dim, keepdim=keepdim))

def poincare_mean(x, dim=0, k=1.0):
    x = poincare_2_klein(x, k)
    lamb = lorenz_factor(x, k=k, keepdim=True)
    mean = torch.sum(lamb * x, dim=dim, keepdim=True) / torch.sum(lamb, dim=dim, keepdim=True)
    mean = klein_2_poincare(mean, k)
    return mean.squeeze(dim)

###################################################################################
# Approximate Gromov's delta hyperbolicity
#
# Code is adapted from:
#   https://github.com/HazyResearch/hgcn/blob/master/utils/hyperbolicity.py
###################################################################################

def hyperbolicity_sample(G, num_samples=50000):
    """
        Approximate Gromov's delta hyperbolicity
    """
    hyps = []
    for i in range(num_samples):
        node_tuple = np.random.choice(G.nodes(), 4, replace=False)
        s = []
        try:
            d01 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[1], weight=None)
            d23 = nx.shortest_path_length(G, source=node_tuple[2], target=node_tuple[3], weight=None)
            d02 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[2], weight=None)
            d13 = nx.shortest_path_length(G, source=node_tuple[1], target=node_tuple[3], weight=None)
            d03 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[3], weight=None)
            d12 = nx.shortest_path_length(G, source=node_tuple[1], target=node_tuple[2], weight=None)
            s.append(d01 + d23)
            s.append(d02 + d13)
            s.append(d03 + d12)
            s.sort()
            hyps.append((s[-1] - s[-2]) / 2)
        except Exception as e:
            continue
    return max(hyps)

###################################################################################
# Translation in the Poincaré model
#
# Code is adapted from:
#   https://github.com/facebookresearch/PoincareMaps/blob/main/model.py
#   Creative Commons Attribution-NonCommercial 4.0 International Public License
#   Copyright (c) 2017-present, Facebook, Inc.
###################################################################################

def poincare_translation(v, x):
    """
    Computes the translation of x  when we move v to the center.
    Hence, the translation of u with -u should be the origin.
    """
    xsq = (x ** 2).sum(axis=1)
    vsq = (v ** 2).sum()
    xv = (x * v).sum(axis=1)
    a = np.matmul((xsq + 2 * xv + 1).reshape(-1, 1),
                  v.reshape(1, -1)) + (1 - vsq) * x
    b = xsq * vsq + 2 * xv + 1
    return np.dot(np.diag(1. / b), a)