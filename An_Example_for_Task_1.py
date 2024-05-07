##############################################################################################################
# This code serves as an example for inferring the high-order neighbors of node z in the Loren63 system.
# The inference results obtained from this code are displayed in Figure 2a in the main text. 
# The encoding of the higher-order neighbors is detailed in Section 2.1 of the supplementary material.
# Other structure inference experiments can be conducted using Algorithm 1 in the main text by configuring different structures (Step (3)).
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
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for computation (e.g., "cpu", "cuda").') 
    parser.add_argument('--model_ind', type=str, default='HoGRC', help='Method identifier.') 
    parser.add_argument('--data_ind', type=str, default='CL', help='Data identifier.') 
    parser.add_argument('--net_nam', type=str, default='er', help='Network name.') 
    parser.add_argument('--direc', type=bool, default=True, help='Direction of network (True for directed).') 
    parser.add_argument('--nj', type=int, default=0, help='the nj-th variable.') 
    
    #Parameters of experimental data 
    parser.add_argument('--N', type=int, default=1, help='Number of samples.') 
    parser.add_argument('--n', type=int, default=1, help='Number of subsystem') 
    parser.add_argument('--T', type=int, default=5000, help='Number of data points.')
    parser.add_argument('--V', type=int, default=3, help='Number of dimension in a subsystem.')
    parser.add_argument('--dt', type=float, default=0.02, help='Sampling time step size.')
    parser.add_argument('--ddt', type=float, default=0.001, help='Simulating time step size.')
    parser.add_argument('--couple_str', type=float, default=1, help='Coupling strength.')
    parser.add_argument('--noise_sigma', type=float, default=0.0, help='Noise strength.')
    parser.add_argument('--ob_noise', type=float, default=0.0, help='Observational noise strength.')
    parser.add_argument('--qtr', type=float, default=0.6, help='Training ratio.')
    parser.add_argument('--threshold', type=float, default=0.01, help='Threshold value for VPS index.')
    
    #Parameters of RC 
    parser.add_argument('--warm_up', type=int, default=50, help='Warm-up period for reservoir.')
    parser.add_argument('--n_internal_units', type=int, default=1000, help='Number of internal units in reservoir.')
    parser.add_argument('--spectral_radius', type=float, default=0.85, help='Spectral radius of the reservoir adjacency matrix.')
    parser.add_argument('--leak', type=float, default=0.05, help='Leak rate of the reservoir.')
    parser.add_argument('--sigma', type=float, default=1, help='Dynamical bias for reservoir.')
    parser.add_argument('--connectivity', type=float, default=0.02, help='Connectivity of the reservoir.')
    parser.add_argument('--input_scaling', type=float, default=0.1, help='Input scaling factor for reservoir.')
    parser.add_argument('--noise_level', type=float, default=0, help='Noise level for reservoir.')
    parser.add_argument('--alpha', type=float, default=10**(-8), help='Regularization coefficient (ridge regression).')

    # Parameters of other methods
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs.')
    parser.add_argument('--batchs', type=int, default=300, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.2, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay (L2 penalty).')

    args = parser.parse_args(args=[])
    return(args) 
    
args = args() 


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


hos = ['7-7-7','7-7-356','7-7-35','7-7-3','7-7-5','7-7-34','7-7-124']
errors = np.zeros((len(hos),2,args.V))

for hoi in range(len(hos)):
    ################################################################
    ### (3) The configuration of higher-order structures         ###
    ################################################################
    ifadj = True
    
    if ifadj:
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
            
        #true stucture ho = '12-25-43'
        ho = hos[hoi]
        edges_in = gen_ein(ho)
    
    
    ################################################################
    ### (4) Model training                                       ###
    ################################################################
    print("#########"+model_ind+"-"+ho+"#########")
    mod = importlib.import_module('models.Model_'+model_ind)
    model = mod.model
        
    ntr = int(args.T*args.qtr)
    experiment = model(args)
    experiment.train()
        
    ################################################################
    ### (5) Model testing                                        ###
    ################################################################
    nj = 0
    Vj = 2
    preds,error = experiment.evalue1()
    print("Total error:", np.mean(np.abs(error)))
    print("Train error:", np.mean(np.abs(error[:,:,:ntr-args.warm_up,:])), np.mean(np.abs(error[:,nj,:ntr-args.warm_up,Vj])))
    print("Test error:", np.mean(np.abs(error[:,:,ntr-args.warm_up:,:])), np.mean(np.abs(error[:,nj,ntr-args.warm_up:,Vj])))
    
    # Saving one-step prediction results
    er = np.zeros((2,args.V))
    for Vj in range(args.V):
        a = np.mean(np.abs(error[:,nj,:ntr-args.warm_up,Vj]))
        b = np.mean(np.abs(error[:,nj,ntr-args.warm_up:,Vj]))
        er[0,Vj] = a
        er[1,Vj] = b
    er = pd.DataFrame(er)
    er.to_csv('results/er'+model_ind+'_'+ho+'.csv')
    errors[hoi] = er

fig = plt.figure(figsize=(25,10))
font1 = {'family':'Times New Roman', 'weight':'normal','size':40}
width = 0.8
x=np.arange(len(errors))
l=errors[:,0,2]
labels = np.arange(len(hos))
plt.bar(x, l, width=width, tick_label = labels,fc = 'g',label='x')   
plt.scatter(x[-2], l[-2]+4e-6, marker ='*', color = 'r', s=800)  
plt.text(0-0.1,1e-6,'{{x,y,z}}',size=35,rotation='vertical')
plt.text(1-0.1,1e-6,'{{x,y},{x,z},{y,z}}',size=35,rotation='vertical')
plt.text(2-0.1,1e-6,'{{x,y},{x,z}}',size=35,rotation='vertical')
plt.text(3-0.1,1e-6,'{{x,y}}',size=35,rotation='vertical')
plt.text(4-0.1,1e-6,'{{x,z}}',size=35,rotation='vertical')
plt.text(5-0.1,1e-6,'{{x,y},{z}}',size=35,rotation='vertical')
plt.text(6-0.1,1e-6,'{{x},{y},{z}}',size=35,rotation='vertical')
plt.ylim(ymin = 0, ymax = 0.00001)    
plt.xlabel("Candidate complexes",size=35)
plt.ylabel(r"e(z)",size=35)
plt.tick_params(labelsize=35)
plt.savefig("results/task1.png")

print("###################################################")
print("One-step Prediction Error with different candidate neighbors:")
print(l)
