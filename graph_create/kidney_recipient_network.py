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

#warnings
import warnings
warnings.filterwarnings("ignore")

#tensor
import torch

def donor_recipient_network_creation(subject: Optional[pd.DataFrame] = None,
                                     object: Optional[pd.DataFrame] = None,
                                     link_col: Optional[str] = "CASEID") -> nx.Graph:
    r"""
    Creates donor-recipient network. 

    ARGS: 
        subject: node 1 (feature matrix in pd.dataframe of initial node, node1 x feature)
        predicate: edge label
        object: node 2 (feature matrix in pd.dataframe of target node, node2 x feature)
        link_col: column that is shared between 

    RETURNS:
        nx.Graph of  network
    """ 
    print("------Constructing Donor-Recipient Network------")
    #make all the nodes into an index
    subject_index = [i for i in range(len(subject))]
    object_index = [i + (len(subject) - 1) for i in range(len(object))] #(subject starts at 0, so last index is len(subject) - 1
    subject["NODE_ID"] = subject_index
    object["NODE_ID"] = object_index

    #create edge index
    subject_index = []
    object_index = []

    #filter, so that lookup is < O(n^2)
    shared_links = list(set(subject[link_col].tolist()) & set(object[link_col].tolist()))
    subject_shared = subject[subject[link_col].isin(shared_links)].reset_index(drop=True)
    object_shared = object[object[link_col].isin(shared_links)].reset_index(drop=True)
    
    #match donor to recipient, iterate through donors
    for donor_i in tqdm(range(len(subject_shared)), desc = "Donor-Recipient Matching", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        for recipient_i in range(len(object_shared)):
            caseid = subject_shared[link_col][donor_i]

            if object_shared[link_col][recipient_i] == caseid:
                #add edge to graph
                subject_index.append(subject_shared["NODE_ID"][donor_i])
                object_index.append(object_shared["NODE_ID"][recipient_i])

    edge_index = torch.LongTensor(np.array([subject_index, object_index]))
    edge_type = torch.LongTensor(np.array([0 for i in range(len(subject))]))
    edge_weight = torch.LongTensor(np.array([1 for i in range(len(subject))]))

    #stats
    print("Donor-Recipient Network Statistics:")
    print("\tnodes:", edge_index.shape[1] * 2)
    print("\tedges:", edge_index.shape[1])
    assert edge_index.shape[1] == len(shared_links)
    return edge_index, edge_weight, edge_type, subject, object
            

    