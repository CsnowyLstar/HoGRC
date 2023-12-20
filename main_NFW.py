import argparse
import numpy as np
import importlib
import pandas as pd
import torch
import random
import networkx as nx
    
################################################################
### (1) Hyperparameter setting                               ###
################################################################
def args(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--device', type=str, default='cpu') 
    parser.add_argument('--model_ind', type=str, default='HoGRC') 
    parser.add_argument('--data_ind', type=str, default='wings') 
    parser.add_argument('--net_nam', type=str, default='er') 
    parser.add_argument('--direc', type=bool, default=True) 
    parser.add_argument('--nj', type=int, default=0) 
    #Parameters of experimental data 
    parser.add_argument('--N', type=int, default=1) 
    parser.add_argument('--n', type=int, default=1) 
    parser.add_argument('--T', type=int, default=10000)
    parser.add_argument('--V', type=int, default=5)
    parser.add_argument('--dt', type=float, default=0.001)
    parser.add_argument('--ddt', type=float, default=0.0001)
    parser.add_argument('--couple_str', type=float, default=1)
    parser.add_argument('--sigma', type=float, default=1)
    parser.add_argument('--noise_sigma', type=float, default=0)
    parser.add_argument('--ob_noise', type=float, default=0.0002)
    parser.add_argument('--qtr', type=float, default=0.6)
    parser.add_argument('--threshold', type=float, default=0.01)
    #Parameters of RC 
    parser.add_argument('--warm_up', type=int, default=50)
    parser.add_argument('--n_internal_units', type=int, default=1000)
    parser.add_argument('--spectral_radius', type=float, default=0.85)
    parser.add_argument('--leak', type=float, default=0.1)
    parser.add_argument('--connectivity', type=float, default=0.02)
    parser.add_argument('--input_scaling', type=float, default=0.03) 
    parser.add_argument('--noise_level', type=float, default=0) 
    parser.add_argument('--alpha', type=float, default=10**(-8))
    #Parameters of other methods 
    parser.add_argument('--epochs', type=int, default=300) 
    parser.add_argument('--batchs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    args = parser.parse_args(args=[])
    return(args) 
    
args = args() 
    
# Select a model from ['RC', 'PRC', 'HoGRC']
model_ind = 'HoGRC'
    

################################################################
### (2) Data generation                                      ###
################################################################
seed = 0
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
    
dataset = importlib.import_module('dataset.Data_'+args.data_ind)
data = dataset.data
    
def gen_data(data):
    dat = data(args)
    dat.gene()
gen_data(data)


################################################################
### (3) The configuration of higher-order structures         ###
################################################################
ifadj = True

if ifadj:
    #edges_in
    def gen_in(i,hon):
        edge_in = []
        g = ''
        u = 2**i 
        for v in hon:
            edge_in.append([u,v])
            g = g+str(v)+'_' 
        return(edge_in,g)
    edges_in = []
    ho = ''
    hons = [[1,2,4,16],[2,13],[4,11],[7,8],[6,16]]
    #hons = [[2,4,16],[31],[31],[31],[31]]
    for i in range(args.V):
        hon = hons[i]
        edge_in,g = gen_in(i,hon)
        edges_in = edges_in+edge_in 
        ho = ho+g
    edges_in = pd.DataFrame(np.array(edges_in))
    edges_in.to_csv("./dataset/data/edges_in.csv")

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
nj = 0
Vj = 0
preds,error = experiment.evalue1()
print("Total error:", np.mean(np.abs(error)))
print("Train error:", np.mean(np.abs(error[:,:,:ntr-args.warm_up,:])), np.mean(np.abs(error[:,nj,:ntr-args.warm_up,Vj])))
print("Test error:", np.mean(np.abs(error[:,:,ntr-args.warm_up:,:])), np.mean(np.abs(error[:,nj,ntr-args.warm_up:,Vj])))
    
steps = 1300
num = 5
start2s = (np.linspace(args.warm_up,ntr-steps,num+2)[1:-1]).astype(int)
start3s = (np.linspace(ntr,args.T-steps,num+2)[1:-1]).astype(int)
def multi():
    preds2s,preds3s,error2s,error3s = experiment.evalue2(start2s=start2s, start3s=start3s, steps=steps) 
    print("Error2s:", np.mean(np.abs(error2s)), np.mean(np.abs(error2s[:,:,nj,:,Vj])))
    print("Error3s:", np.mean(np.abs(error3s)), np.mean(np.abs(error3s[:,:,nj,:,Vj])))
    return(preds2s,preds3s,error2s,error3s)

# Saving one-step prediction results
def sav_er1():
    er = np.zeros((args.n,2*args.V))
    for ni in range(args.n):
        for Vj in range(args.V):
            a = np.mean(np.abs(error[:,ni,:ntr-args.warm_up,Vj]))
            b = np.mean(np.abs(error[:,ni,ntr-args.warm_up:,Vj]))
            er[ni,Vj] = a
            er[ni,Vj+args.V] = b
    er = pd.DataFrame(er)
    er.to_csv('results/er'+model_ind+'_'+ho+'.csv')
    return(er,er,er,er)

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
    return(preds2s,preds3s,error2s,error3s)
    
preds2s,preds3s,error2s,error3s = sav_er2()
    
nj = 0
st = 2
#experiment.draw1(preds,preds2s[st],preds3s[st],start2s[st],start3s[st],steps,nj)
        
    
    
    
    
    