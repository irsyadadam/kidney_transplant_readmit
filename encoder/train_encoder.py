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
from dotenv import load_dotenv

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

from encoder.conv.gnn import GNN
from encoder.conv.rgnn import RGNN
from encoder.kge_scoring.dismult_decoder import DistMultDecoder
from encoder.kge_scoring.transe_decoder import TransEDecoder
from encoder.enc_models.gae_with_gnnconv_kgescore import gnnconv_and_kgescoring_model

#load config
load_dotenv("MODEL_CONFIG")
NUM_EPOCHS = os.getenv("NUM_EPOCHS")
LR = os.getenv("LR")
GNN_LAYERS = os.getenv("GNN_LAYERS")
LAYERS = os.getenv("LAYERS")
ENCODER = os.getenv("ENCODER")
SCORING = os.getenv("SCORING")
EMBED_DIM = os.getenv("EMBED_DIM")
HIDDEN_DIM = os.getenv("HIDDEN_DIM")
HETERO_HEADS = os.getenv("HETERO_HEADS")


def train_encoder(gnn: Optional[str] = ENCODER, 
                  scoring: Optional[str] = SCORING, 
                  kge: Optional[str] = None,
                  data: Optional[str] = None) -> None:

    gnn_models = {"rgcn": pyg.nn.conv.FastRGCNConv,
                  "rgat": pyg.nn.conv.RGATConv, 
                  "gin": pyg.nn.conv.GINConv,
                  "sage": pyg.nn.conv.SAGEConv,
                  "gatedconv": pyg.nn.conv.GatedGraphConv}

    scoring_func = {"dismult": DistMultDecoder,
                    "transe": TransEDecoder}
    
    kge_enc = {"dismult", "transe"}

    if gnn and scoring:
        
        #standard rgnn
        if gnn is "rgcn" or gnn is "rgat":
            gnn_conv = gnn_models[gnn]
            gnn_model = RGNN(encoder_type = gnn_conv, 
                         input_dim = data.x.shape[1], 
                         hidden_dim = HIDDEN_DIM, 
                         output_dim = EMBED_DIM, 
                         gnn_layers = GNN_LAYERS,
                         classifier_layers = LAYERS)
            
        else:
            #downside is that each aggr is the layer type specified in gnn_conv (maybe different conv w different edges?)
            gnn_model = pyg.nn.to_hetero_with_bases(module = GNN(encoder_type = gnn_conv, 
                                                             input_dim = data.x.shape[1], 
                                                             hidden_dim = HIDDEN_DIM, 
                                                             output_dim = EMBED_DIM, 
                                                             gnn_layers = GNN_LAYERS,
                                                             classifier_layers = LAYERS), 
                                                metadata = data.metadata(), 
                                                num_bases = HETERO_HEADS, 
                                                in_channels = data.x.shape[1])
        
        scoring_func = scoring_func[scoring](num_relations = 2, hidden_channels = EMBED_DIM)

        model = gnnconv_and_kgescoring_model(encoder = gnn_model, decoder = scoring_func)

        #TODO: NOT IMPLEMENTED
        best_model = train_link_pred(model, data)
        extract_embeddings(model, data)

    elif kge:
        raise NotImplementedError
 