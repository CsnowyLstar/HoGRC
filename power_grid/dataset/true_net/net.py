import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd

edges = pd.read_csv("edges.csv").values[:,1:]

edgesn = edges[:,::-1]
E = np.concatenate((edgesn,edges),axis=0)
    
E = pd.DataFrame(E)
E.to_csv("E.csv")