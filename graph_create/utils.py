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

#network
import networkx as nx

#vis
import matplotlib.pyplot as plt
import seaborn as sns

#pyg
import torch_geometric as pyg


def create_pyg_heterodata(x1: Optional[np.ndarray] = None,
                          x2: Optional[np.ndarray] = None,
                          node_type: Optional[str] = "patient"
                          m1_edge_index: Optional[Tuple[int, int]] = None, 
                          m1_edge_weight: Optional[np.ndarray] = None, 
                          m1_edge_name: Optional[str] = "d-r",
                          m2_edge_index: Optional[Tuple[int, int]], 
                          m2_edge_weight: Optional[np.ndarray], 
                          m2_edge_name: Optional[str] = "similarity") -> pyg.data.HeteroData:
    r"""
    takes in 2 modal edge types and returns heterogenous data. meant for pyg.to_hetero(GNN) encoder.

    ARGS:
        x1: node features of first node type
        x2: node features of second node type
        node_type: str
        m1_edge_index: edge index type 1
        m1_edge_weight: edge weight type 1
        m1_edge_name: edge 1 name
        m2_edge_index: edge index type 2
        m2_edge_weight: edge weight type 2
        m2_edge_name: edge name 2

    RETURNS:
        pyg.data.HeteroData

    """
                          
    #convert to right datatype
    m1_edge_index, m1_edge_weight = m1_edge_index.to(torch.long), m1_edge_weight.to(torch.long)
    m2_edge_index, m2_edge_weight = m2_edge_index.to(torch.long), m2_edge_weight.to(torch.long)

    #create dataset
    dataset = pyg.data.HeteroData()
    dataset[x1_type].x = x1
    dataset[x2_type].x = x2

    #edges
    dataset[node_type, m1_edge_name, node_type].edge_index = m1_edge_index
    dataset[node_type, m2_edge_name, node_type].edge_index = m2_edge_index

    #edge-weight
    dataset[node_type, m1_edge_name, node_type].edge_weight = m1_edge_weight
    dataset[node_type, m2_edge_name, node_type].edge_weight = m2_edge_weight

    return dataset
    
    

def create_pyg_data(x: Optional[np.ndarray],
                    m1_edge_index: Optional[Tuple[int, int]], 
                    m1_edge_weight: Optional[np.ndarray], 
                    m1_edge_type: Optional[np.ndarray],
                    m2_edge_index: Optional[Tuple[int, int]], 
                    m2_edge_weight: Optional[np.ndarray], 
                    m2_edge_type: Optional[str]) -> pyg.data.Data:
 
    r"""
    takes in 2 modal edge types and returns homogenous data with edge type. meant for rGCN or rGAT encoder.

    ARGS:
        x: node_features
        m1_edge_index: edge index type 1
        m1_edge_weight: edge weight type 1
        m1_edge_type: edge 1 name
        m2_edge_index: edge index type 2
        m2_edge_weight: edge weight type 2
        m2_edge_type: edge name 2

    RETURNS:
        pyg.data.Data(x, edge_index = [2, n], edge_weight = [n], edge_type = [,n])

    """
    #convert to right datatype
    m1_edge_index, m1_edge_weight, m1_edge_type = m1_edge_index.to(torch.long), m1_edge_weight.to(torch.long), m1_edge_type.to(torch.long)
    m2_edge_index, m2_edge_weight, m2_edge_type = dr_edge_index.to(torch.long), m2_edge_weight.to(torch.long), m2_edge_type.to(torch.long)

    #concat
    edge_index = torch.cat([m1_edge_index, m2_edge_index], dim=1)
    edge_weight = torch.cat([m1_edge_weight, m2_edge_weight], dim=0)
    edge_type = torch.cat([m1_edge_type, m2_edge_type], dim=0)

    #create dataset
    dataset = pyg.data.Data(x = x, edge_index = edge_index, edge_weight = edge_weight, edge_type = edge_type)

    return dataset