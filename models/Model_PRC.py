# =================== 导入库 =======================
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import random
import pandas as pd
import joblib 

import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse

from models.reservoir_model_PRC import Reservoir
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

class model():
    def __init__(self, args):
        self.args = args 
        self.Xsn, self.time_point, self.edges_in, self.edges_ex, self.edges_out = self.read_data()
        
        self.ntr = int(self.args.qtr*self.args.T)
        self.loss_f = torch.nn.MSELoss()
        self.epochs = args.epochs
        self.batchs = args.batchs
        self.device = args.device
        self.reservoir = self.RC_param()
    
    def read_data(self):
        args = self.args
        Xs = pd.read_csv("./dataset/data/trajectory.csv").values[:,1:].transpose()
        Xss = np.zeros((args.n, args.N*args.T, args.V))
        Xsn = np.zeros((args.N, args.n, args.T, args.V))
        for i in range(args.V):
            Xss[:,:,i] = Xs[:,i*args.N*args.T:(i+1)*args.N*args.T]
        for i in range(args.N):
            Xsn[i,:,:,:] = Xss[:,i*args.T:(i+1)*args.T,:]
        time_point = pd.read_csv("./dataset/data/time_point.csv").values[:,1:]
        edges_out = pd.read_csv("./dataset/data/edges.csv").values[:,1:]
        edges_in = pd.read_csv("./dataset/data/edges_in.csv").values[:,1:]
        edges_ex = pd.read_csv("./dataset/data/edges_ex.csv").values[:,1:]
        Xsn = torch.tensor(Xsn).float()
        Xsn = Xsn + args.ob_noise * np.random.randn(Xsn.shape[0],Xsn.shape[1],Xsn.shape[2],Xsn.shape[3])
        return(Xsn,time_point,edges_in,edges_ex,edges_out)
    
    def VPT(self,error2s,error3s,steps,num):
        args = self.args
        threshold = args.threshold
        n,V = args.n,args.V
        Xsn = self.Xsn
        X = Xsn[0].numpy()
        sigmas = np.zeros((n,V))
        for ni in range(n):
            for j in range(V):
                sigmas[ni,j] = np.std(X[ni,:,j])
        rmse2 = np.zeros((n,num,steps))
        rmse3 = np.zeros((n,num,steps))
        for ni in range(n):
            for j in range(V):
                error2 = error2s[:,0,ni,:,j]
                error3 = error3s[:,0,ni,:,j]
                sigma = sigmas[ni,j]
                rmse2[ni] += (error2/sigma)**2
                rmse3[ni] += (error3/sigma)**2
            rmse2[ni] = rmse2[ni]/(V)
            rmse3[ni] = rmse3[ni]/(V)
        lens2 = np.zeros((n,num))+steps
        lens3 = np.zeros((n,num))+steps
        for ni in range(n):
            for j in range(num):
                for t in range(steps):
                    if rmse2[ni,j,t]>threshold:
                        lens2[ni,j] = t
                        break 
                for t in range(steps):
                    if rmse3[ni,j,t]>threshold:
                        lens3[ni,j] = t
                        break
        print('lens2:',np.mean(lens2))
        print('lens3:',np.mean(lens3))
        return(lens2,lens3)
    
    def tVPT(self,error2s,error3s,steps,num):
        args = self.args
        threshold = args.threshold
        n,V = args.n,args.V
        Xsn = self.Xsn
        X = Xsn[0].numpy()
        sigmas = np.zeros((n,V))
        for ni in range(n):
            for j in range(V):
                sigmas[ni,j] = np.std(X[ni,:,j])
        rmse2 = np.zeros((num,steps))
        rmse3 = np.zeros((num,steps))
        for ni in range(n):
            for j in range(V):
                error2 = error2s[:,0,ni,:,j]
                error3 = error3s[:,0,ni,:,j]
                sigma = sigmas[ni,j]
                rmse2 += (error2/sigma)**2
                rmse3 += (error3/sigma)**2
        rmse2 = rmse2/(n*V)
        rmse3 = rmse3/(n*V)
        lens2 = np.zeros((num))+steps
        lens3 = np.zeros((num))+steps
        for j in range(num):
            for t in range(steps):
                if rmse2[j,t]>threshold:
                    lens2[j] = t
                    break 
            for t in range(steps):
                if rmse3[j,t]>threshold:
                    lens3[j] = t
                    break
        print('lens2:',np.mean(lens2))
        print('lens3:',np.mean(lens3))
        return(lens2,lens3)
    
    def RC_param(self):
        args = self.args
        edges_in = self.edges_in 
        edges_ex = self.edges_ex 
        edges_out = self.edges_out
        reservoir = Reservoir(n_internal_units = args.n_internal_units,
                              spectral_radius = args.spectral_radius,
                              leak = args.leak,
                              connectivity = args.connectivity,
                              input_scaling = args.input_scaling,
                              noise_level = args.noise_level,
                              edges_in = edges_in,
                              edges_ex = edges_ex,
                              edges_out = edges_out,
                              args = args)
        return(reservoir)
    
    def train(self):
        print("train...")
        Xsn, args, ntr = self.Xsn, self.args, self.ntr
        reservoir = self.reservoir
        res_states = reservoir.get_states(Xsn, n_drop=0, bidir=False)
        # ============ Generate representation of the MTS ============
        Ni = 0
        warm_up = args.warm_up 
        X_train = res_states[Ni,warm_up:ntr,:]
        Y_train = (Xsn[Ni,:,warm_up+1:ntr+1,:].numpy()-Xsn[Ni,:,warm_up:ntr,:].numpy())/args.dt
        for ni in range(args.n):
            for Vi in range(args.V):
                X = X_train[:,(ni*args.V+Vi)*args.n_internal_units:(ni*args.V+Vi+1)*args.n_internal_units]
                Y = Y_train[ni,:,Vi]
                readout = Ridge(alpha=args.alpha)
                readout.fit(X,Y)
                joblib.dump(readout,'./models/model/readout'+str(ni)+'_'+str(Vi)+'.pkl')
        print("Training complete")

    def evalue1(self):
        print("evaluate...")
        reservoir = self.reservoir
        args, Xsn = self.args, self.Xsn
        N, n, T, V = args.N, args.n, args.T, args.V
        warm_up = args.warm_up
        Ni = 0
        res_states = reservoir.get_states(Xsn, n_drop=0, bidir=False)
        
        readout_dict = {}
        for ni in range(n):
            for Vi in range(V):
                readout_dict[(ni, Vi)] = joblib.load('./models/model/readout'+str(ni)+'_'+str(Vi)+'.pkl')
        
        preds = np.zeros((N,n,T-warm_up,V))
        for ni in range(n):
            for Vi in range(V):
                #readout = joblib.load('./models/model/readout'+str(ni)+'_'+str(Vi)+'.pkl')
                readout = readout_dict[(ni, Vi)]
                for i in range(T-warm_up):
                    X = res_states[:,i+warm_up-1,(ni*V+Vi)*args.n_internal_units:(ni*V+Vi+1)*args.n_internal_units]
                    preds[Ni,ni,i,Vi] = readout.predict(X)*args.dt+Xsn[:,ni,i+warm_up-1,Vi].numpy()  
        error = (Xsn[:,:,warm_up:T,:] - preds).numpy()
        return(preds,error)
    
    def evalue2(self, start2s, start3s, steps):
        reservoir = self.reservoir
        args = self.args
        Xsn, N, n, V = self.Xsn, self.args.N, self.args.n, self.args.V    
        Ni = 0
        res_states = reservoir.get_states(Xsn, n_drop=0, bidir=False)
        
        readout_dict = {}
        for ni in range(n):
            for Vi in range(V):
                readout_dict[(ni, Vi)] = joblib.load('./models/model/readout'+str(ni)+'_'+str(Vi)+'.pkl')
 
        preds2s = np.zeros((len(start2s),N,n,steps,V))
        error2s = np.zeros((len(start2s),N,n,steps,V))
        for i in range(len(start2s)):
            start2 = start2s[i]        
            preds2 = np.zeros((N,n,steps,V)) 
            previous_states = res_states[:,start2-1,:]
            current_input = np.zeros((N,n,V))
            current_input[:,:,:] = Xsn[:,:,start2-1,:].numpy()
            for j in range(steps):
                for ni in range(n):
                    for Vi in range(args.V): 
                        readout = readout_dict[(ni, Vi)]
                        #readout = joblib.load('./models/model/readout'+str(ni)+'_'+str(Vi)+'.pkl')
                        X = previous_states[:,(ni*args.V+Vi)*args.n_internal_units:(ni*args.V+Vi+1)*args.n_internal_units]
                        current_input[:,ni,Vi] = readout.predict(X)*args.dt + current_input[:,ni,Vi]
                preds2[Ni,:,j,:] = current_input[0] 
                previous_states = reservoir._compute_netx_state(previous_states, current_input)
            error2 = (Xsn[:,:,start2:(start2+steps),:] - preds2).numpy()
            preds2s[i] = preds2
            error2s[i] = error2  
        
        preds3s = np.zeros((len(start3s),N,n,steps,V))
        error3s = np.zeros((len(start3s),N,n,steps,V))
        for i in range(len(start3s)):
            start3 = start3s[i]        
            preds3 = np.zeros((N,n,steps,V)) 
            previous_states = res_states[:,start3-1,:]
            current_input = np.zeros((N,n,V))
            current_input[:,:,:] = Xsn[:,:,start3-1,:].numpy()
            for j in range(steps):
                for ni in range(n):
                    for Vi in range(args.V): 
                        readout = readout_dict[(ni, Vi)]
                        #readout = joblib.load('./models/model/readout'+str(ni)+'_'+str(Vi)+'.pkl')
                        X = previous_states[:,(ni*args.V+Vi)*args.n_internal_units:(ni*args.V+Vi+1)*args.n_internal_units]
                        current_input[:,ni,Vi] = readout.predict(X)*args.dt + current_input[:,ni,Vi]
                preds3[Ni,:,j,:] = current_input[0] 
                previous_states = reservoir._compute_netx_state(previous_states, current_input)
            error3 = (Xsn[:,:,start3:(start3+steps),:] - preds3).numpy()
            preds3s[i] = preds3
            error3s[i] = error3  
        
        return(preds2s,preds3s,error2s,error3s) 
        
    def draw1(self,preds,preds2,preds3,start2,start3,steps,nj):
        Xsn = self.Xsn
        args= self.args
        warm_up = args.warm_up
        Nj = 0
        ds = 20
        #draw
        X = Xsn.numpy()
        font1 = {'family':'Times New Roman', 'weight':'normal','size':25}
        
        fig = plt.figure(figsize=(20,12))
        for dimension in range(args.V):
            ax11 = fig.add_subplot(args.V,2,1+2*dimension)
            be = start2-ds
            la = start2+steps+ds
            ax11.plot(np.arange(be,la), X[Nj,nj,be:la,dimension],'k-o',markersize=7,label='True')
            ax11.plot(np.arange(be,la), preds[Nj,nj,be-warm_up:la-warm_up,dimension],'y-x',markersize=5,label='One-step prediction')
            ax11.plot(np.arange(start2,start2+steps), preds2[Nj,nj,:,dimension],'r-o',markersize=3,label='Multi-step interpolation prediction')
            ax11.tick_params(labelsize=15)
            ylabel = r'$x_'+str(dimension)+'$'
            ax11.set_ylabel(ylabel,size=25)
            #ax11.legend(prop=font1)
            
            ax12 = fig.add_subplot(args.V,2,2+2*dimension)
            be = start3-ds
            la = start3+steps+ds
            ax12.plot(np.arange(be,la), X[Nj,nj,be:la,dimension],'k-o',markersize=7,label='True')
            ax12.plot(np.arange(be,la), preds[Nj,nj,be-warm_up:la-warm_up,dimension],'y-x',markersize=5,label='One-step prediction')
            ax12.plot(np.arange(start3,start3+steps), preds3[Nj,nj,:,dimension],'g-o',markersize=3,label='Multi-step extrapolation prediction')
            ax12.tick_params(labelsize=15)
            ax12.set_ylabel(ylabel,size=25)
            #ax12.legend(prop=font1)
        
    def draw2(self,preds,preds2,preds3,start2,start3,steps,nj):
        Xsn = self.Xsn
        args= self.args
        warm_up = args.warm_up
        Nj = 0
        #draw
        X = Xsn.numpy()
        font1 = {'family':'Times New Roman', 'weight':'normal','size':20}
        
        fig = plt.figure(figsize=(16,12))
        for dimension in range(args.V):
            ax1 = fig.add_subplot(args.V,1,dimension+1)
            ax1.plot(np.arange(warm_up,args.T), X[Nj,nj,warm_up:args.T,dimension],'k-o',markersize=7,label='True')
            ax1.plot(np.arange(warm_up,args.T), preds[Nj,nj,:,dimension],'y-x',markersize=5,label='One-step prediction')
            ax1.plot(np.arange(start2,start2+steps), preds2[Nj,nj,:,dimension],'r-o',markersize=3,label='Multi-step interpolation prediction')
            ax1.plot(np.arange(start3,start3+steps), preds3[Nj,nj,:,dimension],'g-o',markersize=3,label='Multi-step extrapolation prediction')
            ax1.tick_params(labelsize=15)
            ylabel = r'$x_'+str(dimension)+'$'
            ax1.set_ylabel(ylabel,size=25)   
    
    def draw_FHN(self,preds,preds2,preds3,start2,start3,steps,nj):    
        X = self.Xsn.numpy()
        
        Nj = 0
        itv = 2
        font1 = {'family':'Times New Roman', 'weight':'normal','size':30}
        fig = plt.figure(figsize=(16,12))
        ax1 = fig.add_subplot(1,1,1)
        
        ax1.plot(X[Nj,nj,::itv,0],X[Nj,nj,::itv,1],'k-',markersize=10,label='True')
        #ax1.plot(preds[Nj,nj,::itv,0],preds[Nj,nj,::itv,1],'y-x', markersize=7,label='One-step prediction')
        ax1.plot(preds2[Nj,nj,::itv,0],preds2[Nj,nj,::itv,1],'ro',markersize=10,label='Multi-step interpolation prediction')
        ax1.plot(preds3[Nj,nj,::itv,0],preds3[Nj,nj,::itv,1],'go',markersize=10,label='Multi-step extrapolation prediction') 
        
        ax1.tick_params(labelsize=30)
        ax1.set_xlabel('$x_0$',size=40)
        ax1.set_ylabel('$x_1$',size=40)
        ax1.legend(prop=font1)
        
    def draw_3d(self,preds,preds2,preds3,start2,start3,steps,nj):    
        X = self.Xsn.numpy()  
        X = X[:,:,1000:,:]
        preds3 = preds3[:,:,:steps,:]
        
        Nj = 0
        itv = 1
        font1 = {'family':'Times New Roman', 'weight':'normal','size':30}
        fig = plt.figure(figsize=(20,20))
        ax1 = fig.add_subplot(111, projection='3d')
        
        ax1.grid(False)
        ax1.plot(X[Nj,nj,::itv,0],X[Nj,nj,::itv,1],X[Nj,nj,::itv,2],'k-',markersize=10,label='True')
        #ax1.plot(preds[Nj,nj,::itv,0],preds[Nj,nj,::itv,1],preds[Nj,nj,::itv,2],'y-x', markersize=7,label='One-step prediction')
        ax1.plot(preds2[Nj,nj,::itv,0],preds2[Nj,nj,::itv,1],preds2[Nj,nj,::itv,2],'ro',markersize=10,label='Multi-step interpolation prediction')
        ax1.plot(preds3[Nj,nj,::itv,0],preds3[Nj,nj,::itv,1],preds3[Nj,nj,::itv,2],'go',markersize=10,label='Multi-step extrapolation prediction') 
        
        ax1.tick_params(labelsize=25)
        ax1.set_xlabel('$x_0$',size=30)
        ax1.set_ylabel('$x_1$',size=30)
        ax1.set_zlabel('$x_2$',size=30)
        ax1.legend(prop=font1)
        