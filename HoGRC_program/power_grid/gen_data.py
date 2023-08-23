import argparse
import numpy as np
import importlib
import pandas as pd
import torch
import random
import networkx as nx
import matplotlib.pyplot as plt
    
def args(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--device', type=str, default='cpu') 
    parser.add_argument('--model_ind', type=str, default='27') 
    parser.add_argument('--data_ind', type=str, default='HCO') 
    parser.add_argument('--net_nam', type=str, default='edges') 
    parser.add_argument('--direc', type=bool, default=True) 
    parser.add_argument('--nj', type=int, default=0) 
    #Parameters of experimental data 
    parser.add_argument('--N', type=int, default=1) 
    parser.add_argument('--n', type=int, default=120) 
    parser.add_argument('--T', type=int, default=10000)
    parser.add_argument('--V', type=int, default=2)
    parser.add_argument('--dt', type=float, default=0.08)
    parser.add_argument('--ddt', type=float, default=0.02)
    parser.add_argument('--couple_str', type=float, default=0.5)
    parser.add_argument('--sigma', type=float, default=0)
    parser.add_argument('--noise_sigma', type=float, default=0)
    parser.add_argument('--ob_noise', type=float, default=0.0)
    parser.add_argument('--qtr', type=float, default=0.6)
    parser.add_argument('--threshold', type=float, default=0.01)
    #Parameters of RC 
    parser.add_argument('--warm_up', type=int, default=100)
    parser.add_argument('--n_internal_units', type=int, default=500)
    parser.add_argument('--spectral_radius', type=float, default=0.9)
    parser.add_argument('--leak', type=float, default=0.1)
    parser.add_argument('--leak1', type=float, default=0.4)
    parser.add_argument('--leak2', type=float, default=0.3)
    parser.add_argument('--connectivity', type=float, default=0.02)
    parser.add_argument('--input_scaling', type=float, default=0.3) 
    parser.add_argument('--noise_level', type=float, default=0.00) 
    parser.add_argument('--circle', type=bool, default=False)
    parser.add_argument('--alpha', type=float, default=10**(-8))
    #Parameters of other methods 
    parser.add_argument('--epochs', type=int, default=300) 
    parser.add_argument('--batchs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    args = parser.parse_args(args=[])
    return(args) 
    
args = args() 
    
# gen_data 
seed = 5
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
    
dataset = importlib.import_module('dataset.Data_'+args.data_ind)
data = dataset.data
    
def gen_data(data):
    dat = data(args)
    dat.gene()
gen_data(data)


    
    
    