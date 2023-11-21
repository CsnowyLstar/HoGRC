import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
from models.neural_dynamics import NDCN
import sys
import functools
from torch.optim.lr_scheduler import ExponentialLR
from scipy.linalg import fractional_matrix_power

class model():
    def __init__(self, args):
        self.args = args 
        self.Xsn, self.time_point, self.edge_index, self.OM = self.read_data()
        self.ndcn = self.build_model()
        self.ntr = int(args.T*args.qtr)
    
    def read_data(self):
        args = self.args
        Xs = pd.read_csv("./dataset/data/trajectory.csv").values[:,1:].transpose()
        Xss = np.zeros((args.n, args.N*args.T, args.V))
        Xsn = np.zeros((args.N, args.n, args.T, args.V))
        for i in range(args.V):
            Xss[:,:,i] = Xs[:,i*args.N*args.T:(i+1)*args.N*args.T]
        for i in range(args.N):
            Xsn[i,:,:,:] = Xss[:,i*args.T:(i+1)*args.T,:]
        time_point = pd.read_csv("./dataset/data/time_point.csv").values[:,1:][:,0]
        edges = pd.read_csv("./dataset/data/edges.csv").values[:,1:].transpose()
        Xsn = torch.tensor(Xsn).float()
        time_point = torch.tensor(time_point).float()
        if len(edges) != 0:
            de = torch.tensor(edges) 
            if args.direc:
                edge_index = de
            else:
                edge_index = torch.cat((de,de[[1,0]]),axis=1)
        else:
            edge_index = torch.tensor([]).long()
        
        A = np.zeros((args.n,args.n))
        for i in range(edge_index.shape[1]):
            A[edge_index[0,i],edge_index[1,i]] = 1
        '''
        A = torch.tensor(A).float()
        D = torch.diag(A.sum(1))
        L = (D - A)
        OM = L
        '''
        A = A + np.diag(np.ones(args.n))
        D = np.diag(A.sum(1))
        tilD = fractional_matrix_power(D,-0.5)
        OM = np.dot(np.dot(tilD,A),tilD)
        OM = torch.tensor(OM).float()
        return(Xsn,time_point,edge_index,OM)
    
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
    
    def build_model(self):
        args = self.args
        OM = self.OM
        input_size = args.V
        hidden_size = 32
        dropout = 0  # 0 default, not stochastic ODE  
        output_size = args.V  # 1 for regression  
    
        ndcn = NDCN(n=args.n,input_size=input_size, hidden_size=hidden_size, A=OM, output_size=output_size,
                     dropout=dropout, no_embed=False, no_graph=False, no_control=False,
                     rtol=1e-4, atol=1e-6, method='euler')
        return(ndcn)
    
    def train(self):
        way = 0
        args, ntr = self.args, self.ntr
        warm_up, dt = args.warm_up, args.dt
        time_point, ndcn, Xsn = self.time_point, self.ndcn, self.Xsn
        params = ndcn.parameters()
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = ExponentialLR(optimizer, gamma=0.98)
        criterion = F.l1_loss
        #SGD; Predict next step 
        if way ==0: 
            for epoch in range(args.epochs):
                loss = 0
                for Ni in range(args.N):
                    for t in range(warm_up, ntr-1):
                        optimizer.zero_grad()
                        true_y0 = Xsn[Ni,:,t,:]
                        true_y_train = Xsn[Ni,:,t:t+2,:]
                        t_train = torch.tensor([0,dt])
                        pred_y = ndcn(t_train, true_y0).transpose(0,1)
                        loss_train = criterion(pred_y[:,-1,:], true_y_train[:,-1,:])
                        loss_train.backward()
                        optimizer.step()
                        loss += loss_train.detach().numpy()
                print("epoch:",epoch,", loss_train:",loss/(args.N*(ntr-1-warm_up)))
        #BGD; Predict next step 
        elif way == 1: 
            for epoch in range(args.epochs):
                for Ni in range(args.N):
                    Tloss = torch.tensor(0.0)
                    for t in range(warm_up, ntr-1):
                        optimizer.zero_grad()
                        true_y0 = Xsn[Ni,:,t,:]
                        true_y_train = Xsn[Ni,:,t:t+2,:]
                        t_train = torch.tensor([0,dt])
                        pred_y = ndcn(t_train, true_y0).transpose(0,1)
                        loss_train = criterion(pred_y[:,-1,:], (true_y_train[:,-1,:]-true_y_train[:,-2,:])/dt)
                        Tloss += loss_train
                    Tloss.backward()
                    optimizer.step()
                    scheduler.step()
                    print("epoch:",epoch,", loss:",Tloss/(ntr-1-warm_up)) 
        #SGD; Predict all tracks 
        elif way == 2: 
            for epoch in range(args.epochs):
                for Ni in range(args.N): 
                    Tloss = torch.tensor(0.0) 
                    for t in range(ntr-1): 
                        optimizer.zero_grad() 
                        true_y0 = Xsn[Ni,:,t,:] 
                        true_y_train = Xsn[Ni,:,t:ntr,:] 
                        t_train = time_point[t:ntr] 
                        pred_y = ndcn(t_train, true_y0).transpose(0,1) 
                        loss_train = criterion(pred_y[:,-1,:], true_y_train[:,-1,:]) 
                        loss_train.backward() 
                        optimizer.step() 
                        Tloss += loss_train                        
                    print("epoch:",epoch,", loss:",Tloss/(ntr-1)) 
        torch.save(ndcn, './models/model/ndcn.pkl') 
        
    def evalue1(self):
        ndcn = torch.load('./models/model/ndcn.pkl')
        args = self.args
        Xsn = self.Xsn
        n, T, dt, N, V, warm_up = args.n, args.T, args.dt, args.N, args.V, args.warm_up
        
        with torch.no_grad():
            preds = np.zeros((N,n,T-warm_up,V))
            for Ni in range(N):
                for t in range(T-warm_up):
                    true_y0 = Xsn[Ni,:,t+warm_up-1,:]
                    t_test = torch.tensor([0,dt])
                    pred_y = ndcn(t_test, true_y0).transpose(0,1)
                    preds[Ni,:,t,:] = pred_y[:,-1,:]*dt + true_y0
                    
        error = (Xsn[:,:,warm_up:T,:] - preds).numpy()
        
        return(preds,error)

    def evalue2(self, start2s, start3s, steps):
        way = 1
        ndcn = torch.load('./models/model/ndcn.pkl')
        args = self.args
        Xsn = self.Xsn
        n, dt, N, V = args.n, args.dt, args.N, args.V
        with torch.no_grad():
            preds2s = np.zeros((len(start2s),N,n,steps,V))
            error2s = np.zeros((len(start2s),N,n,steps,V))
            for i in range(len(start2s)):
                start2 = start2s[i]
                preds2 = np.zeros((N,n,steps,V)) 
                for Ni in range(N):
                    if way==1:
                        true_y0 = Xsn[Ni,:,start2-1,:]
                        for t in range(steps):
                            t_test = torch.tensor([0,dt])
                            pred_y = ndcn(t_test, true_y0).transpose(0,1)
                            preds2[Ni,:,t,:] = pred_y[:,-1,:]*dt + true_y0
                            true_y0 = torch.tensor(preds2[Ni,:,t,:]).float()
                    else:
                        true_y0 = Xsn[Ni,:,start2-1,:]
                        t_test = torch.tensor(np.arange(steps+1)*dt)
                        pred_y = ndcn(t_test, true_y0).transpose(0,1)
                        preds2[Ni] = pred_y[:,1:,:]*dt + Xsn[Ni,:,start2-1:start2-1+steps,:]    
                error2 = (Xsn[:,:,start2:(start2+steps),:] - preds2).numpy()
                preds2s[i] = preds2
                error2s[i] = error2                
                
            preds3s = np.zeros((len(start3s),N,n,steps,V))
            error3s = np.zeros((len(start3s),N,n,steps,V))
            for i in range(len(start3s)):
                start3 = start3s[i]
                preds3 = np.zeros((N,n,steps,V)) 
                for Ni in range(N):
                    if way==1:
                        true_y0 = Xsn[Ni,:,start3-1,:]
                        for t in range(steps):
                            t_test = torch.tensor([0,dt])
                            pred_y = ndcn(t_test, true_y0).transpose(0,1)
                            preds3[Ni,:,t,:] = pred_y[:,-1,:]*dt + true_y0
                            true_y0 = torch.tensor(preds3[Ni,:,t,:]).float()
                    else:
                        true_y0 = Xsn[Ni,:,start3-1,:]
                        t_test = torch.tensor(np.arange(steps+1)*dt)
                        pred_y = ndcn(t_test, true_y0).transpose(0,1)
                        preds3[Ni] = pred_y[:,1:,:]*dt + Xsn[Ni,:,start3-1:start3-1+steps,:]    
                error3 = (Xsn[:,:,start3:(start3+steps),:] - preds3).numpy()
                preds3s[i] = preds3
                error3s[i] = error3                   
        
        return(preds2s,preds3s,error2s,error3s)

    def draw1(self,preds,preds2,preds3,start2,start3,steps,plot_figure,nj):
        Xsn = self.Xsn
        args= self.args
        ntr = self.ntr
        warm_up = args.warm_up
        Nj = 0
        ds = 400
        #draw
        if plot_figure==1:
            X = Xsn.numpy()
            font1 = {'family':'Times New Roman', 'weight':'normal','size':25}
            
            fig = plt.figure(figsize=(20,12))
            dimension = 0
            ax11 = fig.add_subplot(args.V,2,1)
            be = start2-ds
            la = start2+ds
            ax11.plot(np.arange(be,la), X[Nj,nj,be:la,dimension],'k-o',markersize=7,label='True')
            ax11.plot(np.arange(be,la), preds[Nj,nj,be-warm_up:la-warm_up,dimension],'y-x',markersize=5,label='One-step prediction')
            ax11.plot(np.arange(start2,start2+steps), preds2[Nj,nj,:,dimension],'r-o',markersize=3,label='Multi-step interpolation prediction')
            ax11.tick_params(labelsize=15)
            ax11.set_ylabel(r'$x_{A,0}$',size=25)
            #ax11.legend(prop=font1)
            
            ax12 = fig.add_subplot(args.V,2,2)
            be = start3-ds
            la = start3+ds
            ax12.plot(np.arange(be,la), X[Nj,nj,be:la,dimension],'k-o',markersize=7,label='True')
            ax12.plot(np.arange(be,la), preds[Nj,nj,be-warm_up:la-warm_up,dimension],'y-x',markersize=5,label='One-step prediction')
            ax12.plot(np.arange(start3,start3+steps), preds3[Nj,nj,:,dimension],'g-o',markersize=3,label='Multi-step extrapolation prediction')
            ax12.tick_params(labelsize=15)
            ax12.set_ylabel(r'$x_{A,0}$',size=25)
            #ax12.legend(prop=font1)
            
            if args.V>1:
                dimension = 1
                ax21 = fig.add_subplot(args.V,2,3)
                be = start2-ds
                la = start2+ds
                ax21.plot(np.arange(be,la), X[Nj,nj,be:la,dimension],'k-o',markersize=7,label='True')
                ax21.plot(np.arange(be,la), preds[Nj,nj,be-warm_up:la-warm_up,dimension],'y-x',markersize=5,label='One-step prediction')
                ax21.plot(np.arange(start2,start2+steps), preds2[Nj,nj,:,dimension],'r-o',markersize=3,label='Multi-step interpolation prediction')
                ax21.tick_params(labelsize=15)
                ax21.set_ylabel(r'$x_{A,1}$',size=25)
                
                ax22 = fig.add_subplot(args.V,2,4)
                be = start3-ds
                la = start3+ds
                ax22.plot(np.arange(be,la), X[Nj,nj,be:la,dimension],'k-o',markersize=7,label='True')
                ax22.plot(np.arange(be,la), preds[Nj,nj,be-warm_up:la-warm_up,dimension],'y-x',markersize=5,label='One-step prediction')
                ax22.plot(np.arange(start3,start3+steps), preds3[Nj,nj,:,dimension],'g-o',markersize=3,label='Multi-step extrapolation prediction')
                ax22.tick_params(labelsize=15)
                ax22.set_ylabel(r'$x_{A,1}$',size=25)
                #ax1.legend(prop=font1)
            
            if args.V>2:
                dimension = 2
                ax31 = fig.add_subplot(args.V,2,5)
                be = start2-ds
                la = start2+ds
                ax31.plot(np.arange(be,la), X[Nj,nj,be:la,dimension],'k-o',markersize=7,label='True')
                ax31.plot(np.arange(be,la), preds[Nj,nj,be-warm_up:la-warm_up,dimension],'y-x',markersize=5,label='One-step prediction')
                ax31.plot(np.arange(start2,start2+steps), preds2[Nj,nj,:,dimension],'r-o',markersize=3,label='Multi-step interpolation prediction')
                ax31.tick_params(labelsize=15)
                ax31.set_ylabel(r'$x_{A,2}$',size=25)
                ax31.set_xlabel('t',size=25)
                
                ax32 = fig.add_subplot(args.V,2,6)
                be = start3-ds
                la = start3+ds
                ax32.plot(np.arange(be,la), X[Nj,nj,be:la,dimension],'k-o',markersize=7,label='True')
                ax32.plot(np.arange(be,la), preds[Nj,nj,be-warm_up:la-warm_up,dimension],'y-x',markersize=5,label='One-step prediction')
                ax32.plot(np.arange(start3,start3+steps), preds3[Nj,nj,:,dimension],'g-o',markersize=3,label='Multi-step extrapolation prediction')
                ax32.tick_params(labelsize=15)
                ax32.set_ylabel(r'$x_{A,2}$',size=25)
                ax32.set_xlabel('t',size=25)
                #ax1.legend(prop=font1)      












