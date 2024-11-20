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

#import
from graph_create.similarity_network import similarity_network_creation
from graph_create.kidney_recipient_network import donor_recipient_network_creation
from graph_create.utils import create_pyg_heterodata, create_pyg_data, 
    

def create_kr_sim_network(subject: Optional[pd.DataFrame[Any]] = None,
                          object: Optional[pd.DataFrame[Any]] = None,
                          link_col: Optional[str] = "CASEID",
                          similarity_threshold: Optional[float] = 0.0001,
                          graph_type: Optional[str] = "hetero") -> Any:
    r"""
    creates kidney/recipient + patient similarity network

    ARGS:
        subject
        object
        link_col
        similarity_threshold
        graph_type
        
    RETURNS:
        torch_geometric.data

    """

    assert graph_type in ["hetero", "homo"]

    #TODO: accomodate for 1 single df for everything
    subject = _to_subject(df)
    object = _to_object(df)
    linlk = _to_link(df)

    #get first knowledge graph
    dr_edge_index, dr_edge_weight, dr_edge_type, subject, object = donor_recipient_network_creation(subject = subject, object = object, link_col = link_col)
    final_df = pd.concat([subject, object], ignore_index=True)

    #get second knowledge graph
    s_edge_index, s_edge_weight, s_edge_type = similarity_network_creation(feature_matrix = final_np, threshold = similarity_threshold) #similarity operator already defined as gaussian kernel

    if graph_type == "homo":
        #standard
        graph = create_pyg_data(x, dr_edge_index, dr_edge_weight, dr_edge_type, s_edge_index, s_edge_weight, s_edge_type)
    elif graph_type == "hetero":
        graph = create_pyg_data(x1, x2, dr_edge_index, dr_edge_weight, dr_edge_type, s_edge_index, s_edge_weight, s_edge_type)

    return graph