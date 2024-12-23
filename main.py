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

from kidney_donor_matching.graph_create.network_creation import create_kr_sim_network
from kidneys_donor_matching.encoder.train_encoder import train_encoder
from kidneys_donor_matching.encoder.extract_embeddings import extract_embeddings



if __name__ == '__main__':

    #load configs
    load_dotenv("DATA_PATH.env")
    DATA_PATH = os.getenv("DATA_PATH")
    data = pd.read_csv(DATA_PATH)
    

    #CLI args
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--GNN', type = str, default = None, help = "(str) GNN Backbone for Link Prediction on KR-Sim Network.")
    parser.add_argument('--SCORING', type = str, default = None, help = "(str) Scoring Function for GNN Backbone on KR-Sim Network.")
    parser.add_argument('--KGE', type = str,  default = None, help = "(str) KGE Embedding Method for Link Pred on KR-Sim Network.")

    args = parser.parse_args()

    gnns = ["rgcn", "rgat", "gin", "sage", "sagn", None]
    scoring_func = ["dismult", "transe", None]
    kge_enc = ["dismult", "transe", None]

    assert args.GNN in gnns
    assert args.SCORING in scoring_func
    assert args.KGE in kge_enc

    #load data
    #TODO: FIX THE DATA LOAD
    data = create_kr_sim_network(graph_type = "hetero")

    if args.GNN is None or args.SCORING is None or args.KGE is None:
        #override with configs file
        train_encoder = train_encoder(data = data)
    else:
        train_encoder = train_encoder(gnn = args.GNN, 
                                      scoring = args.SCORING, 
                                      kge = args.KGE,
                                      data = data)
        