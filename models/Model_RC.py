# =================== 导入库 =======================
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import pandas as pd
import joblib 

import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse

from models.reservoir_model_RC import Reservoir
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
    
class model():
    def __init__(self, args):
        self.args = args 
        self.Xsn, self.time_point = self.read_data()
    
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
        Xsn = torch.tensor(Xsn).float()
        Xsn = Xsn + args.ob_noise * np.random.randn(Xsn.shape[0],Xsn.shape[1],Xsn.shape[2],Xsn.shape[3])
        return(Xsn,time_point)
    
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
        reservoir = Reservoir(n_internal_units = args.n_internal_units,
                              spectral_radius = args.spectral_radius,
                              leak = args.leak,
                              connectivity = args.connectivity,
                              input_scaling = args.input_scaling,
                              noise_level = args.noise_level,
                              args = args)
        return(reservoir)
    
    def train(self): 
        print("train...")
        Xsn = self.Xsn 
        reservoir = self.reservoir 
        res_states = reservoir.get_states(Xsn, n_drop=0, bidir=False)
        # ============ Generate representation of the MTS ============
        N, n, V, ntr, dt = self.args.N, self.args.n, self.args.V, self.ntr, self.args.dt
        warm_up = self.args.warm_up 
        readout = Ridge(alpha=self.args.alpha)
        for Ni in range(N): 
            if Ni == 0:
                X = res_states[Ni,warm_up:ntr,:] 
                Yb = Xsn[Ni,:,warm_up+1:ntr+1,:].numpy().swapaxes(0,1).reshape(ntr-warm_up,n*V)
                Yl = Xsn[Ni,:,warm_up:ntr,:].numpy().swapaxes(0,1).reshape(ntr-warm_up,n*V)
                Y = (Yb-Yl)/dt
            else: 
                X = np.concatenate((X,res_states[Ni,warm_up:ntr,:]), axis=0)
                Yb = Xsn[Ni,:,warm_up+1:ntr+1,:].numpy().swapaxes(0,1).reshape(ntr-warm_up,n*V)
                Yl = Xsn[Ni,:,warm_up:ntr,:].numpy().swapaxes(0,1).reshape(ntr-warm_up,n*V)
                Y = np.concatenate((Y,(Yb-Yl)/dt), axis=0)
        readout.fit(X,Y)
        joblib.dump(readout,'./models/model/readout.pkl')
        print("Training complete")

    def evalue1(self):
        print("evaluate...")
        reservoir, args, Xsn = self.reservoir, self.args, self.Xsn
        N, n, T, V, dt = args.N, args.n, args.T, args.V, args.dt
        warm_up = args.warm_up
        preds = np.zeros((N,T-warm_up,n*V))
        res_states = reservoir.get_states(Xsn, n_drop=0, bidir=False)
        
        readout = joblib.load('./models/model/readout.pkl')
        for i in range(T-warm_up):
            Yb = Xsn[:,:,i+warm_up-1,:].numpy().reshape(N,n*V)
            preds[:,i,:] = readout.predict(res_states[:,i+warm_up-1])*dt + Yb
        preds = preds.reshape(N,T-warm_up,n,V).swapaxes(1,2)
        
        error = (Xsn[:,:,warm_up:T,:] - preds).numpy()
        return(preds,error)
    
    def evalue2(self, start2s, start3s, steps):
        reservoir, args = self.reservoir, self.args
        Xsn, N, n, V, dt = self.Xsn, args.N, args.n, args.V, args.dt
        
        res_states = reservoir.get_states(Xsn, n_drop=0, bidir=False)
        readout = joblib.load('./models/model/readout.pkl')
        
        preds2s = np.zeros((len(start2s),N,n,steps,V))
        error2s = np.zeros((len(start2s),N,n,steps,V))
        for i in range(len(start2s)):
            start2 = start2s[i]
            preds2 = np.zeros((N,steps,n*V)) 
            previous_state = res_states[:,start2-1,:] 
            current_input = np.zeros((N,n,V))
            current_input[:,:,:] = Xsn[:,:,start2-1,:].numpy()
            for j in range(steps):
                current_input = readout.predict(previous_state)*dt + current_input.reshape(N,n*V)
                preds2[:,j,:] = current_input 
                previous_state = reservoir._compute_netx_state(previous_state,current_input)
            preds2 = preds2.reshape(N,steps,n,V).swapaxes(1,2)
            error2 = (Xsn[:,:,start2:(start2+steps),:] - preds2).numpy()
            preds2s[i] = preds2 
            error2s[i] = error2
        
        preds3s = np.zeros((len(start3s),N,n,steps,V))
        error3s = np.zeros((len(start3s),N,n,steps,V))
        for i in range(len(start3s)):
            start3 = start3s[i]
            preds3 = np.zeros((N,steps,n*V)) 
            previous_state = res_states[:,start3-1,:] 
            current_input = np.zeros((N,n,V))
            current_input[:,:,:] = Xsn[:,:,start3-1,:].numpy()
            for j in range(steps):
                current_input = readout.predict(previous_state)*dt + current_input.reshape(N,n*V)
                preds3[:,j,:] = current_input 
                previous_state = reservoir._compute_netx_state(previous_state,current_input)
            preds3 = preds3.reshape(N,steps,n,V).swapaxes(1,2)
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