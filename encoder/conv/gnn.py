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
import torch


class GNN(torch.nn.Module):
    def __init__(self, 
                 encoder_type: Optional[Callable] = None, 
                 input_dim: Optional[int] = None, 
                 hidden_dim: Optional[int] = None, 
                 gnn_layers: Optional[int] = None, 
                 classifier_layers: Optional[int] = None,
                 output_dim: Optional[int] = None):
        super(GNN, self).__init__

        self.layer1 = encoder_type(input_dim, hidden_dim)

        self.layers = torch.nn.ModuleList()
        for _ in range(gnn_layers - 1):
            self.layers.append(encoder_type(hidden_dim, hidden_dim))
        
        self.classifier_layers = torch.nn.ModuleList()
        for _ in range(classifier_layers - 1):
            self.classifier_layers.append(torch.nn.Linear(hidden_dim, hidden_dim))

        self.readout = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(p = 0.2)
        self.norm = torch.nn.BatchNorm1D(hidden_dim)
        
    def forward(self, x, edge_index, edge_weight = None):

        #gnn aggregation
        if edge_weight == None:
            x = self.layer1(x, edge_index).relu()
            for layer in self.layers:
                x = layer(x, edge_index).relu()
                x = self.dropout(x)  
        else:
            x = self.layer1(x, edge_index, edge_weight).relu()
            for layer in self.layers:
                x = layer(x, edge_index, edge_weight).relu()
                x = self.dropout(x)   

        #classifier
        x = self.norm(x)
        for layer in self.classifier_layers:
            x = layer(x).relu()
            x = self.dropout(x)  
        
        x = self.readout(x)
        return x

