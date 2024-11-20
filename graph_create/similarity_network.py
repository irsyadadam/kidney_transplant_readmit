#os
import importlib.metadata
import json
import logging
import os
import re
import tempfile
import time
import ast
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, TypeVar, Union

#data handling
import pandas as pd
import numpy as np
from tqdm import tqdm

#stats
import scipy
import sklearn
from scipy.spatial.distance import pdist, squareform


#network
import networkx as nx

#vis
import matplotlib.pyplot as plt
import seaborn as sns

#nx to pyg
#nx to pyg
import torch_geometric as pyg
import torch

def gaussian_kernel(feature_matrix: Optional[np.ndarray] = None) -> np.ndarray:
    pairwise_dists = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(feature_matrix, 'euclidean'))
    gauss_kern = np.exp(-pairwise_dists ** 2 / pairwise_dists.std() ** 2)
    return gauss_kern

def similarity_network_creation(feature_matrix: Optional[np.ndarray] = None,
                                similarity_operator: Optional[Callable[[np.ndarray], np.ndarray]] = gaussian_kernel,
                                threshold: Optional[float] = 0.5) -> nx.Graph:
    r"""
    Similarity Network Creation

    ARGS:
        feature_matrix:
        similarity_operator: function to generate similarity matrix, Callable[np.ndarray] -> float
        threshold: float for edge chopping to introduce sparsity, default=0.5
    
    RETURNS: 
        nx.network with edges as similarity
    """
    print("------Constructing Similarity Network------")
    #network init
    print("\tsimilarity operator...", end = "")
    similarity_matrix = similarity_operator(feature_matrix)
    similarity_matrix = torch.Tensor(similarity_matrix)
    print("success")

    print("\tedge_index/edge_weight creation...", end = "")

    #change adj to edge index, edge_weight
    edge_index = (similarity_matrix > 0).nonzero().t()
    row, col = edge_index
    edge_weight = similarity_matrix[row, col]
    print("success")

    print("\tchopping edges...", end = "")
    #remove self loops
    edge_index, edge_weight = pyg.utils.remove_self_loops(edge_index, edge_weight)
    
    #filter based on topK threshold
    topk = int(threshold * len(edge_weight))
    topk_indices = torch.topk(edge_weight, topk, largest=True).indices
    edge_index = edge_index[:, topk_indices]
    edge_weight = edge_weight[topk_indices]
    edge_type = torch.Tensor([1 for i in range(len(edge_weight))])
    print("success")

    #stats
    print("Similarity Network Statistics:")
    print("\tnodes:", similarity_matrix.shape[0])
    print("\tedges:", edge_index.shape[1])
    print("\t(min edge, max edge): (%s, %s)" % (edge_weight.min().numpy(), edge_weight.max().numpy()))
    print("\t(avg, std): (%s, %s)" % (edge_weight.mean().numpy(), edge_weight.var().numpy()))

    return edge_index, edge_weight, edge_type