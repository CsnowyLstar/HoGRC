import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import networkx as nx
import pandas as pd
import sys
import functools

from models.utils.ElementaryFunctionsPool import *
from models.utils.ElementaryFunctions_Matrix import *
from models.utils.NumericalDerivatives import *
from models.utils.TwoPhaseInference import *

class model():
    def __init__(self, args):
        self.args = args 
        self.Xsn, self.Xn, self.time_point, self.edge_index, self.A, self.Lambda = self.read_data()
        self.ntr = int(args.T*args.qtr)
        self.selfPolyOrder = 3
        self.coupledPolyOrder = 1
        self.Keep = 10
        self.Batchsize = args.n
        self.SampleTimes = 1
        self.plotstart = 0.5
        self.plotend = 0.9
    
    def read_data(self):
        args = self.args
        Xs = pd.read_csv("./dataset/data/trajectory.csv").values[:,1:].transpose()
        Xss = np.zeros((args.n, args.N*args.T, args.V))
        Xsn = np.zeros((args.N, args.n, args.T, args.V))
        for i in range(args.V):
            Xss[:,:,i] = Xs[:,i*args.N*args.T:(i+1)*args.N*args.T]
        for i in range(args.N):
            Xsn[i,:,:,:] = Xss[:,i*args.T:(i+1)*args.T,:]
        Xn = np.zeros((args.N, args.T, args.n*args.V))
        for i in range(args.n):
            Xn[:,:,i*args.V:(i+1)*args.V] = Xsn[:,i,:,:]
            
        time_point = pd.read_csv("./dataset/data/time_point.csv").values[:,1:][:,0]
        
        edges = pd.read_csv("./dataset/data/edges.csv").values[:,1:].transpose()
        if len(edges) != 0:
            de = torch.tensor(edges) 
            edge_index = torch.cat((de,de[[1,0]]),axis=1)
        else:
            edge_index = torch.tensor([]).long()
        
        A = np.zeros((args.n,args.n))
        for i in range(edge_index.shape[1]):
            A[edge_index[0,i],edge_index[1,i]] = 1
        
        Lambda = pd.read_csv('models/utils/Lambda_Kura.csv',encoding='utf-8',header=None)

        return(Xsn,Xn,time_point,edge_index,A,Lambda)
    
    def VPT(self,error2s,error3s,steps,num):
        args = self.args
        threshold = args.threshold
        n,V = args.n,args.V
        Xsn = self.Xsn
        X = Xsn[0]
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
        X = Xsn[0]
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
    
    def train(self):
        args = self.args
        A, Xn, Lambda = self.A, self.Xn, self.Lambda
        selfPolyOrder, coupledPolyOrder = self.selfPolyOrder, self.coupledPolyOrder
        Keep, Batchsize, SampleTimes = self.Keep, self.Batchsize, self.SampleTimes
        plotstart, plotend, ntr = self.plotstart, self.plotend, self.ntr
        V, n, dt = args.V, args.n, args.dt
        Ni = 0
        data = Xn[Ni,:ntr]
        NumDiv = NumericalDeriv(data,V,n,dt)
        data = data[2:-2,:]
        Matrix = ElementaryFunctions_Matrix(data, V, n, A, selfPolyOrder, coupledPolyOrder, PolynomialIndex = True, TrigonometricIndex = True, \
            ExponentialIndex = True, FractionalIndex = False, ActivationIndex = False, RescalingIndex = False, CoupledPolynomialIndex = True, \
                CoupledTrigonometricIndex = True, CoupledExponentialIndex = True, CoupledFractionalIndex = False, \
                    CoupledActivationIndex = True, CoupledRescalingIndex = True)
        Matrix = Matrix.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
        
        for dim in range(V):
            locals()['InferredResults'+str(dim+1)], locals()['phase_one_series'+str(dim+1)], \
                locals()['wAIC'+str(dim+1)], locals()['withConstant'+str(dim+1)] \
                    = TwoPhaseInference(Matrix, NumDiv, n, dim, V, Keep, SampleTimes, Batchsize, Lambda, plotstart, plotend)
            locals()['InferredResults'+str(dim+1)].to_csv('models/results/'+'Inferred_model_of_'+str(dim+1)+'-dimension.csv', index=True, header=True)
            
    def evalue1(self):  
        args = self.args
        A, Xsn, Xn = self.A, self.Xsn, self.Xn
        selfPolyOrder, coupledPolyOrder = self.selfPolyOrder, self.coupledPolyOrder
        n, T, N, V, dt, warm_up = args.n, args.T, args.N, args.V, args.dt, args.warm_up
        
        terms = []
        coefs = []
        for dim in range(V):
            coef = pd.read_csv('./models/results/Inferred_model_of_'+str(dim+1)+'-dimension.csv').values
            a,b = coef.shape
            terms.append(coef[:,0])
            co = np.zeros(a)
            for ai in range(a):
                co[ai] = coef[ai,1:].mean()
            coefs.append(co)
        
        preds = np.zeros((N,n,T-warm_up,V))
        for Ni in range(N):
            data = Xn[Ni]
            Matrix = ElementaryFunctions_Matrix(data, V, n, A, selfPolyOrder, coupledPolyOrder, PolynomialIndex = True, TrigonometricIndex = True, \
                ExponentialIndex = True, FractionalIndex = False, ActivationIndex = False, RescalingIndex = False, CoupledPolynomialIndex = True, \
                    CoupledTrigonometricIndex = True, CoupledExponentialIndex = True, CoupledFractionalIndex = False, \
                        CoupledActivationIndex = True, CoupledRescalingIndex = True)
            #Matrix = Matrix.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
            Matrix = pd.concat([Matrix, pd.DataFrame(data=np.ones((Matrix.shape[0],1)),columns=['constant'])], axis=1)
            Dt = np.zeros((n*T,V))
            for Vi in range(V):
                Dt[:,Vi] = np.sum(Matrix[terms[Vi]].values * coefs[Vi], axis=1)
            Dt = Dt.reshape(n,T,V)
            preds[Ni] = Xsn[Ni,:,(warm_up-1):(T-1),:] + dt*Dt[:,(warm_up-1):(T-1),:]
        
        error = Xsn[:,:,warm_up:T,:] - preds
        return(preds,error)
    
    def evalue2(self, start2s, start3s, steps):
        args = self.args
        A, Xsn, Xn = self.A, self.Xsn, self.Xn
        selfPolyOrder, coupledPolyOrder = self.selfPolyOrder, self.coupledPolyOrder
        ntr = self.ntr
        n, T, N, V, dt, warm_up = args.n, args.T, args.N, args.V, args.dt, args.warm_up
                
        terms = []
        coefs = []
        for dim in range(V):
            coef = pd.read_csv('./models/results/Inferred_model_of_'+str(dim+1)+'-dimension.csv').values
            a,b = coef.shape
            terms.append(coef[:,0])
            co = np.zeros(a)
            for ai in range(a):
                co[ai] = coef[ai,1:].mean()
            coefs.append(co)        
        
        preds2s = np.zeros((len(start2s),N,n,steps,V))
        error2s = np.ones((len(start2s),N,n,steps,V))*100
        preds3s = np.zeros((len(start3s),N,n,steps,V))
        error3s = np.ones((len(start3s),N,n,steps,V))*100            
        
        for j in range(len(start2s)):
            print('2-',j)
            start2 = start2s[j]
            preds2 = np.zeros((N,steps,n*V)) 
            for Ni in range(N):
                data = Xn[Ni,start2-1,:][None,:]   
                k = -1
                for i in range(steps):
                    Matrix = ElementaryFunctions_Matrix(data, V, n, A, selfPolyOrder, coupledPolyOrder, PolynomialIndex = True, TrigonometricIndex = True, \
                        ExponentialIndex = True, FractionalIndex = False, ActivationIndex = False, RescalingIndex = False, CoupledPolynomialIndex = True, \
                            CoupledTrigonometricIndex = True, CoupledExponentialIndex = True, CoupledFractionalIndex = False, \
                                CoupledActivationIndex = True, CoupledRescalingIndex = True)
                    Matrix = Matrix.replace([np.inf, -np.inf], np.nan).dropna(axis=1)   
                    Matrix = pd.concat([Matrix, pd.DataFrame(data=np.ones((Matrix.shape[0],1)),columns=['constant'])], axis=1)
                    Dt = np.zeros((n,V))
                    for tei in range(len(terms)):
                        termi = terms[tei]
                        for tej in range(len(termi)):
                            termij = termi[tej]
                            if termij not in Matrix:
                                k = i
                    if k!= -1:
                        break
                    for Vi in range(V):
                        Dt[:,Vi] = np.sum(Matrix[terms[Vi]].values * coefs[Vi], axis=1)
                    preds2[Ni,i,:] = data.reshape(n*V) + dt*Dt.reshape(n*V)
                    data = preds2[Ni,i,:][None,:]
                    
            preds2 = preds2.reshape(N,steps,n,V).swapaxes(1,2)
            error2 = Xsn[:,:,start2:(start2+steps),:] - preds2
            preds2s[j,:,:,:k,:] = preds2[:,:,:k,:] 
            error2s[j,:,:,:k,:] = error2[:,:,:k,:] 
                    
        for j in range(len(start3s)):
            print('3-',j)
            start3 = start3s[j]            
            preds3 = np.zeros((N,steps,n*V)) 
            for Ni in range(N):
                data = Xn[Ni,start3-1,:][None,:]  
                k=-1
                for i in range(steps):
                    Matrix = ElementaryFunctions_Matrix(data, V, n, A, selfPolyOrder, coupledPolyOrder, PolynomialIndex = True, TrigonometricIndex = True, \
                        ExponentialIndex = True, FractionalIndex = False, ActivationIndex = False, RescalingIndex = False, CoupledPolynomialIndex = True, \
                            CoupledTrigonometricIndex = True, CoupledExponentialIndex = True, CoupledFractionalIndex = False, \
                                CoupledActivationIndex = True, CoupledRescalingIndex = True)
                    Matrix = Matrix.replace([np.inf, -np.inf], np.nan).dropna(axis=1)      
                    Matrix = pd.concat([Matrix, pd.DataFrame(data=np.ones((Matrix.shape[0],1)),columns=['constant'])], axis=1)
                    Dt = np.zeros((n,V))
                    for tei in range(len(terms)):
                        termi = terms[tei]
                        for tej in range(len(termi)):
                            termij = termi[tej]
                            if termij not in Matrix:
                                k = i
                    if k!= -1:
                        break
                    for Vi in range(V):
                        Dt[:,Vi] = np.sum(Matrix[terms[Vi]].values * coefs[Vi], axis=1)
                    preds3[Ni,i,:] = data.reshape(n*V) + dt*Dt.reshape(n*V)
                    data = preds3[Ni,i,:][None,:]        
            
            preds3 = preds3.reshape(N,steps,n,V).swapaxes(1,2)
            error3 = Xsn[:,:,start3:(start3+steps),:] - preds3             
            preds3s[j,:,:,:k,:] = preds3[:,:,:k,:]
            error3s[j,:,:,:k,:] = error3[:,:,:k,:]           
        
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






