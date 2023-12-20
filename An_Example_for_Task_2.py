##############################################################################################################
# This code serves as an example for prediction of the CRos system using different methods.
# The prediction results obtained from this code are displayed in Figures 3a and 3c in the main text. 
# Other dynamics prediction experiments can be conducted in the same way by selecting different methods.
##############################################################################################################
import argparse
import numpy as np
import importlib
import pandas as pd
import torch
import random
import networkx as nx
import matplotlib.pyplot as plt

################################################################
### (1) Hyperparameter setting                               ###
################################################################
def args(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--device', type=str, default='cpu') 
    parser.add_argument('--model_ind', type=str, default='18') 
    parser.add_argument('--data_ind', type=str, default='FHN') 
    parser.add_argument('--net_nam', type=str, default='edges') 
    parser.add_argument('--direc', type=bool, default=True) 
    parser.add_argument('--nj', type=int, default=0) 
    #Parameters of experimental data 
    parser.add_argument('--N', type=int, default=1) 
    parser.add_argument('--n', type=int, default=5) 
    parser.add_argument('--T', type=int, default=5000)
    parser.add_argument('--V', type=int, default=2)
    parser.add_argument('--dt', type=float, default=0.25)
    parser.add_argument('--ddt', type=float, default=0.01)
    parser.add_argument('--couple_str', type=float, default=0.5)
    parser.add_argument('--sigma', type=float, default=1.2)
    parser.add_argument('--noise_sigma', type=float, default=0)
    parser.add_argument('--ob_noise', type=float, default=0.001)
    parser.add_argument('--qtr', type=float, default=0.6)
    parser.add_argument('--threshold', type=float, default=0.01)
    #Parameters of RC 
    parser.add_argument('--warm_up', type=int, default=100)
    parser.add_argument('--n_internal_units', type=int, default=500)
    parser.add_argument('--spectral_radius', type=float, default=0.55)
    parser.add_argument('--leak', type=float, default=0.1)
    parser.add_argument('--connectivity', type=float, default=0.05) 
    parser.add_argument('--input_scaling', type=float, default=2) 
    parser.add_argument('--noise_level', type=float, default=0) 
    parser.add_argument('--alpha', type=float, default=10**(-8))
    #Parameters of other methods 
    parser.add_argument('--epochs', type=int, default=100) 
    parser.add_argument('--batchs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    args = parser.parse_args(args=[])
    return(args) 
    
args = args() 
    
################################################################
### (2) Data generation                                      ###
################################################################
seed = 1
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
    
dataset = importlib.import_module('dataset.Data_'+args.data_ind)
data = dataset.data
    
def gen_data(data):
    dat = data(args)
    dat.gene()
gen_data(data)

methods = ['RC', 'PRC', 'HoGRC']
lens3s = np.zeros((len(methods)))

for mi in range(len(methods)):
    model_ind = methods[mi]
    
    ################################################################
    ### (3) The configuration of higher-order structures         ###
    ################################################################
    ifadj = True
    
    if ifadj:
        # edges_in
        def gen_ein(ho):
            edges_in = []
            j = 1
            for i in ho:
                if i=='-':
                    j=2*j 
                else:
                    edges_in.append([j,int(i)])   
            edges_in = pd.DataFrame(np.array(edges_in))
            edges_in.to_csv("./dataset/data/edges_in.csv")
            return(edges_in)
        ho = '12-12'
        edges_in = gen_ein(ho)
        
        # edges_out
        def gen_eout(hp):
            net = nx.DiGraph() 
            j = 0
            for i in hp:
                if i=='-':
                    j=j+1
                else:
                    net.add_edge(j,int(i))    
            edges = net.edges()
            edges = pd.DataFrame(edges())
            edges.to_csv("./dataset/data/edges.csv")
        hp = '1234-023-14-2-'
        #hp = '1234-0234-0134-0124-0123'
        edges_out = gen_eout(hp)
            
        # edges_ex
        edges_ex = pd.DataFrame(np.array([[1],[-1],[1]]))
        edges_ex.to_csv("./dataset/data/edges_ex.csv")
    
    
    ################################################################
    ### (4) Model training                                       ###
    ################################################################
    print("#########"+model_ind+"#########")
    mod = importlib.import_module('models.Model_'+model_ind)
    model = mod.model
    
    ntr = int(args.T*args.qtr)
    experiment = model(args)
    experiment.train()
    
    ################################################################
    ### (5) Model testing                                        ###
    ################################################################
    nj = 1
    Vj = 0
    preds,error = experiment.evalue1()
    print("Total error:", np.mean(np.abs(error)))
    print("Train error:", np.mean(np.abs(error[:,:,:ntr-args.warm_up,:])), np.mean(np.abs(error[:,nj,:ntr-args.warm_up,Vj])))
    print("Test error:", np.mean(np.abs(error[:,:,ntr-args.warm_up:,:])), np.mean(np.abs(error[:,nj,ntr-args.warm_up:,Vj])))
    
    steps = 1500
    num = 5
    start2s = (np.linspace(args.warm_up,ntr-steps,num+2)[1:-1]).astype(int)
    start3s = (np.linspace(ntr,args.T-steps,num+2)[1:-1]).astype(int)
    def multi():
        preds2s,preds3s,error2s,error3s = experiment.evalue2(start2s=start2s, start3s=start3s, steps=steps) 
        print("Error2s:", np.mean(np.abs(error2s)), np.mean(np.abs(error2s[:,:,nj,:,Vj])))
        print("Error3s:", np.mean(np.abs(error3s)), np.mean(np.abs(error3s[:,:,nj,:,Vj])))
        return(preds2s,preds3s,error2s,error3s)
    
    # Saving multi-step prediction results
    def sav_er2(): 
        preds2s,preds3s,error2s,error3s = multi()
        for ni in range(args.n):
            for vj in range(args.V):
                error_2 = np.abs(error2s[:,0,ni,:,vj])
                error_3 = np.abs(error3s[:,0,ni,:,vj])
                error_2pd = pd.DataFrame(error_2)
                error_3pd = pd.DataFrame(error_3)
                error_2pd.to_csv('results/error'+model_ind+str(ni)+'_'+str(vj)+'_2.csv')
                error_3pd.to_csv('results/error'+model_ind+str(ni)+'_'+str(vj)+'_3.csv')
        lens2,lens3 = experiment.tVPT(error2s,error3s,steps,num)
        return(preds2s,preds3s,error2s,error3s,lens2,lens3)
        
    preds2s,preds3s,error2s,error3s,lens2,lens3 = sav_er2()
    lens3s[mi] = np.mean(lens3)
   
print("########### Final results ############")
print("VPS results:")
for mi in range(len(methods)):
    print(methods[mi],'mean VPS:',lens3s[mi])                                    