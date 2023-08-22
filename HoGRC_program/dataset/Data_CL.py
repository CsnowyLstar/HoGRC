import numpy as np
import random as random
import networkx as nx
import pandas as pd
import torch.nn as nn 
import torch

class data():
    def __init__(self, args):
        self.noise_sigma = args.noise_sigma
        self.args = args
        self.N = args.N 
        self.n = args.n #network size
        self.T = args.T #total iteration steps
        self.V = args.V #dimension
        self.dt = args.dt
        self.ddt = args.ddt
        
        self.c = args.couple_str
        self.h = 2*(np.random.rand(self.n)-0.5)*0.5
        
        self.net = self.gen_net()
    
    def gene(self):
        Xss = self.gen_data() 
        self.sav_data(Xss)
    
    def gen_net(self):
        args = self.args
        if args.direc:
            net = nx.DiGraph() 
        else:
            net = nx.Graph() 
        net.add_nodes_from(np.arange(self.n)) 
        if args.net_nam == 'er': 
            for u, v in nx.erdos_renyi_graph(self.n, 0.6).edges():
                net.add_edge(u,v,weight=random.uniform(0,1))
        elif args.net_nam == 'ba':
            for u, v in nx.barabasi_albert_graph(self.n, 4).edges():
                net.add_edge(u,v,weight=random.uniform(1,1)) 
        elif args.net_nam == 'rg':
            for u, v in nx.random_regular_graph(4,self.n).edges():
                net.add_edge(u,v,weight=random.uniform(1,1))
        else:
            edges = pd.read_csv("./dataset/data/edges.csv").values[:,1:]
            weights = pd.read_csv("./dataset/data/weights.csv").values[:,1:]
            for i in range(edges.shape[0]):
                net.add_edge(edges[i,0],edges[i,1],weight=weights[i])
            print(args.net_nam)
            
        for i in range(args.n):
            swei = 0
            for j in net.neighbors(i):
                swei = swei+net.get_edge_data(i,j)['weight'] 
            for j in net.neighbors(i):
                net.edges[i,j]['weight'] = net.edges[i,j]['weight']/swei
        return(net)
    
    # self-dynamics
    def F(self, x, k):
        f = np.zeros(x.shape)
        f[0] = -10*(x[0]-x[1])
        f[1] = 28*(1+self.h[k])*x[0] - x[1] - x[0]*x[2]
        f[2] = -8/3*x[2] + x[0]*x[1]
        return(f)
    
    # coupling-dynamics
    def G(self, x, y):
        g = np.zeros(x.shape)
        g[0] = -10*(y[1]-x[1])
        return(g)
    
    def gen_data(self):
        T, N, n, V, dt, ddt = self.T, self.N, self.n, self.V, self.dt, self.ddt
        net, F, G = self.net, self.F, self.G
        c = self.c
        Xs = np.zeros((T*N,n,V))
        for Ni in range(N):
            # initial condition 
            x = np.zeros((T,n,V))
            x_cur = np.ones((n,V)) + 5*1e-1*np.random.rand(n,V) 
            x[0,:,:] = x_cur
            deln = int(dt/ddt)
            # dynamical equation
            for it in range(T*deln-1):
                for i in range(n):
                    f = F(x_cur[i,:],i)
                    g = 0
                    for j in net.neighbors(i):
                        g += G(x_cur[i,:],x_cur[j,:])*net.get_edge_data(i,j)['weight']
                    dx = (f + c*g)*ddt
                    x_cur[i,:] = x_cur[i,:] + dx
                if (it+1)%deln == 0:
                    x[int((it+1)/deln),:,:] = x_cur
            Xs[Ni*T:(Ni+1)*T,:,:] = x 
        Xss = np.zeros((T*N*V,n))
        for Vi in range(V):
            Xss[Vi*T*N:(Vi+1)*T*N,:] = Xs[:,:,Vi]
        noise = self.noise_sigma * np.random.randn(Xss.shape[0],Xss.shape[1])
        Xss += noise
        return(Xss)
    
    def sav_data(self, Xss):
        time_point = np.arange(self.T)*self.dt
        Xss = pd.DataFrame(Xss)
        time_point = pd.DataFrame(time_point)
        Xss.to_csv("./dataset/data/trajectory.csv")
        time_point.to_csv("./dataset/data/time_point.csv")
        edges = self.net.edges()
        edges = pd.DataFrame(edges())
        edges.to_csv("./dataset/data/edges.csv")
        
        net = self.net
        weights = []
        for u,v in net.edges():
            weights.append(net.edges[u,v]['weight'])
        weights = pd.DataFrame(weights)
        weights.to_csv('./dataset/data/weights.csv')
        
        #edges_in = pd.DataFrame(np.array([[1,1],[1,2],[2,2],[2,5],[4,4],[4,3]]))
        edges_in = pd.DataFrame(np.array([[1,1],[1,2],[2,2],[2,5],[4,4],[4,3]]))
        edges_in.to_csv("./dataset/data/edges_in.csv")
        edges_ex = pd.DataFrame(np.array([[1],[2],[2]]))
        edges_ex.to_csv("./dataset/data/edges_ex.csv")
        
        print("Data generated successfully!")
    
    











