import argparse
import numpy as np
import importlib
import pandas as pd
import torch
import random
import networkx as nx
    
def args(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--device', type=str, default='cpu') 
    parser.add_argument('--model_ind', type=str, default='27') 
    parser.add_argument('--data_ind', type=str, default='Colpitts') 
    parser.add_argument('--net_nam', type=str, default='er') 
    parser.add_argument('--direc', type=bool, default=True) 
    parser.add_argument('--nj', type=int, default=0) 
    #Parameters of experimental data 
    parser.add_argument('--N', type=int, default=1) 
    parser.add_argument('--n', type=int, default=1) 
    parser.add_argument('--T', type=int, default=5000)
    parser.add_argument('--V', type=int, default=3)
    parser.add_argument('--dt', type=float, default=0.2)
    parser.add_argument('--ddt', type=float, default=0.001)
    parser.add_argument('--couple_str', type=float, default=1)
    parser.add_argument('--sigma', type=float, default=1)
    parser.add_argument('--noise_sigma', type=float, default=0)
    parser.add_argument('--ob_noise', type=float, default=0.0)
    parser.add_argument('--qtr', type=float, default=0.6)
    parser.add_argument('--threshold', type=float, default=0.01)
    #Parameters of RC 
    parser.add_argument('--warm_up', type=int, default=50)
    parser.add_argument('--n_internal_units', type=int, default=1000)
    parser.add_argument('--spectral_radius', type=float, default=0.6)
    parser.add_argument('--leak', type=float, default=0.3)
    parser.add_argument('--connectivity', type=float, default=0.02)
    parser.add_argument('--input_scaling', type=float, default=0.2) 
    parser.add_argument('--noise_level', type=float, default=0.00) 
    parser.add_argument('--alpha', type=float, default=10**(-6))
    #Parameters of other methods 
    parser.add_argument('--epochs', type=int, default=300) 
    parser.add_argument('--batchs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    args = parser.parse_args(args=[])
    return(args) 

args = args() 
    
#model_inds = ['22m','27m','88m']
model_inds = ['22m']
    

################################################################
# Generate data
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
#gen_data(data)


################################################################
# Adjust the higher-order structure
################################################################
ifadj = False 

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
    ho = '2-124-12'
    #ho = '7-7-7'
    edges_in = gen_ein(ho)
        
    # edges_ex
    edges_ex = pd.DataFrame(np.array([[1],[-1],[2]]))
    edges_ex.to_csv("./dataset/data/edges_ex.csv")


################################################################
# Train the model and save the test results
################################################################
for model_ind in model_inds:
    print("#########"+model_ind+"#########")
    
    mod = importlib.import_module('models.Model'+model_ind)    
    model = mod.model
    
    # train
    ntr = int(args.T*args.qtr)
    experiment = model(args)
    experiment.train()
    
    # test
    nj = 0
    Vj = 0
    preds,error = experiment.evalue1()
    print("Total error:", np.mean(np.abs(error)))
    print("Train error:", np.mean(np.abs(error[:,:,:ntr-args.warm_up,:])), np.mean(np.abs(error[:,nj,:ntr-args.warm_up,Vj])))
    print("Test error:", np.mean(np.abs(error[:,:,ntr-args.warm_up:,:])), np.mean(np.abs(error[:,nj,ntr-args.warm_up:,Vj])))
    
    
    steps = 800
    num = 50
    start2s = (np.linspace(args.warm_up,ntr-steps,num+2)[1:-1]).astype(int)
    start3s = (np.linspace(ntr,args.T-steps,num+2)[1:-1]).astype(int)
    def multi():
        preds2s,preds3s,error2s,error3s = experiment.evalue2(start2s=start2s, start3s=start3s, steps=steps) 
        print("Error2s:", np.mean(np.abs(error2s)), np.mean(np.abs(error2s[:,:,nj,:,Vj])))
        print("Error3s:", np.mean(np.abs(error3s)), np.mean(np.abs(error3s[:,:,nj,:,Vj])))
        return(preds2s,preds3s,error2s,error3s)
    
    ##Retain prediction results and errors
    def sav_er1(): 
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
    
    #one step prediction error and VPT
    def sav_er2():
        preds2s,preds3s,error2s,error3s = multi()
        er = np.zeros((2,3))
        for Vj in range(args.V):
            a = np.mean(np.abs(error[:,nj,:ntr-args.warm_up,Vj]))
            b = np.mean(np.abs(error[:,nj,ntr-args.warm_up:,Vj]))
            er[0,Vj] = a
            er[1,Vj] = b
        er = pd.DataFrame(er)
        er.to_csv('er'+model_ind+'_'+ho+'.csv')
        error_2 = pd.DataFrame(np.abs(error2s[:,0,nj,:,0]))
        error_3 = pd.DataFrame(np.abs(error3s[:,0,nj,:,0]))
        error_2.to_csv('results/error'+model_ind+'_2x'+'_'+ho+'.csv')
        error_3.to_csv('results/error'+model_ind+'_3x'+'_'+ho+'.csv')
        error_2 = pd.DataFrame(np.abs(error2s[:,0,nj,:,1]))
        error_3 = pd.DataFrame(np.abs(error3s[:,0,nj,:,1]))
        error_2.to_csv('results/error'+model_ind+'_2y'+'_'+ho+'.csv')
        error_3.to_csv('results/error'+model_ind+'_3y'+'_'+ho+'.csv')
        error_2 = pd.DataFrame(np.abs(error2s[:,0,nj,:,2]))
        error_3 = pd.DataFrame(np.abs(error3s[:,0,nj,:,2]))
        error_2.to_csv('results/error'+model_ind+'_2z'+'_'+ho+'.csv')
        error_3.to_csv('results/error'+model_ind+'_3z'+'_'+ho+'.csv')
        
        return(preds2s,preds3s,error2s,error3s)
    
    #one step prediction error
    def sav_er3():
        er = np.zeros((2,3))
        for Vj in range(args.V):
            a = np.mean(np.abs(error[:,nj,:ntr-args.warm_up,Vj]))
            b = np.mean(np.abs(error[:,nj,ntr-args.warm_up:,Vj]))
            er[0,Vj] = a
            er[1,Vj] = b
        er = pd.DataFrame(er)
        er.to_csv('results/er'+model_ind+'_'+ho+'.csv')
        return(er,er,er,er)
        
    preds2s,preds3s,error2s,error3s = sav_er1()
    
nj = 0
st = 0
#experiment.draw1(preds,preds2s[st],preds3s[st],start2s[st],start3s[st],steps,nj)
        
    
    
    
    
    