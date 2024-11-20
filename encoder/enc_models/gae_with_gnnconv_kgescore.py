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

def gnnconv_and_kgescoring_model(gnn_encoder, kge_scoring):
    #hidden channels between decoder and encoder need to be the same
    return pyg.nn.GAE(encoder = gnn_encoder, decoder = kge_scoring)
