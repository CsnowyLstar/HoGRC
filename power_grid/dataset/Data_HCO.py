import numpy as np
import random as random
import networkx as nx
import pandas as pd
    
class data():
    def __init__(self, args):
        self.noise_sigma = args.noise_sigma
        self.N = args.N 
        self.n = args.n #network size
        self.T = args.T #total iteration steps
        self.V = args.V #dimension
        self.dt = args.dt
        self.ddt = args.ddt
        self.args = args
        phase_lag = True
        if phase_lag:
            self.alpha = 0.02
            self.beta = 0.06
        else:
            self.alpha = 0
            self.beta = 0
        
        symb = np.random.rand(args.n)
        for i in range(args.n):
            if symb[i]>0.5:
                symb[i]=1
            else:
                symb[i]=-1
        self.w = (np.random.rand(args.n)+0.3)*symb
        #self.w = np.ones(args.n)*(0.5)
        self.epsilon1 = 0.4
        self.epsilon2 = 0.4
        self.net, self.Sd = self.gen_net()
        #self.edge_d = self.delet(self.Sd)
    
    '''
    def delet(self,Sd):
        edge_d = []
        for i in range(Sd.shape[0]):
            edge_d.append([Sd[i,0],Sd[i,1]])
            edge_d.append([Sd[i,0],Sd[i,2]])
        return(edge_d)
    '''
    
    def gene(self):
        Xss,theta = self.gen_data() 
        self.sav_data(Xss,theta)
    
    def gen_net(self):
        args = self.args
        net = nx.DiGraph() 
        net.add_nodes_from(np.arange(self.n)) 
        if args.net_nam == 'er': 
            for u, v in nx.erdos_renyi_graph(self.n, 0.4).edges():
                net.add_edge(u,v,weight=random.uniform(0.5,1))
                if not args.direc:
                    net.add_edge(v,u,weight=random.uniform(0.5,1))
        elif args.net_nam == 'ba':
            for u, v in nx.barabasi_albert_graph(self.n, 2).edges():
                net.add_edge(u,v,weight=random.uniform(1,1)) 
                if not args.direc:
                    net.add_edge(v,u,weight=random.uniform(1,1))
        elif args.net_nam == 'rg':
            for u, v in nx.random_regular_graph(3,self.n).edges():
                net.add_edge(u,v,weight=random.uniform(1,1))
                if not args.direc:
                    net.add_edge(v,u,weight=random.uniform(1,1))
        else:
            #edges = pd.read_csv("./dataset/data/edges.csv").values[:,1:]
            edges = pd.read_csv("./dataset/true_net/E.csv").values[:,1:]
            #weights = pd.read_csv("./dataset/data/weights.csv").values[:,1:]
            for i in range(edges.shape[0]):
                net.add_edge(edges[i,0],edges[i,1],weight=1)
            print(args.net_nam)
        
        if args.n >= 5:
            #Sd = np.array([[3,6,7],[5,2,47],[14,15,17],[21,36,1],[37,8,38]])
            #Sd = np.array([[25,27,28],[27,25,28],[28,25,27]])
            '''
            Sd = np.array([[14,111,112],[111,14,112],[112,14,111],
                           [19,21,22],[21,19,22],[22,19,21],
                           [19,22,112],[22,19,112],[112,19,22],
                           [20,21,23],[21,20,23],[23,20,21],
                           [30,32,33],[32,30,33],[33,30,32],
                           [32,34,35],[34,32,35],[35,32,34],
                           [46,47,55],[47,46,55],[55,46,47],
                           [49,57,114],[57,49,114],[114,49,57],
                           [81,82,117],[82,81,117],[117,81,82],
                           [84,87,88],[87,84,88],[88,84,87],
                           [95,100,109],[100,95,109],[109,95,100]])
            '''
            Sd = np.array([[14,111,112],
                           [33,30,32],
                           [49,57,114],
                           [81,82,117],
                           [84,87,88],
                           [100,95,109]])
        return(net,Sd)
        
    # self-dynamics
    def F(self, x, i):
        f = self.w[i]
        return(f)
    
    # pairwise interaction
    def G(self, x, y):
        #g = np.sin(y+np.pi/4)**2
        g = np.sin(y-x-self.alpha) + np.sin(self.alpha)
        return(g)
    
    # higher-order interaction
    def SG(self, x, y, z):
        sg = np.sin(y+z-2*x-self.beta) + np.sin(self.beta)
        return(sg)
    
    def gen_data(self):
        T, N, n, V, dt, ddt = self.T, self.N, self.n, self.V, self.dt, self.ddt
        net, F, G, SG, Sd = self.net, self.F, self.G, self.SG, self.Sd
        #edge_d = self.edge_d
        epsilon1, epsilon2 = self.epsilon1, self.epsilon2
        Xs = np.zeros((T*N,n,V))
        theta = np.zeros((T*N,n))
        vv = 1
        for Ni in range(N):
            # initial condition 
            x = np.zeros((T,n,vv))
            x_cur = np.random.rand(n,vv)*2*np.pi
            x[0,:,:] = x_cur
            deln = int(dt/ddt)
            # dynamical equation
            for it in range(T*deln-1):
                for i in range(n):
                    f = F(x_cur[i,:],i)
                    g = 0
                    for j in net.neighbors(i):
                        #if [i,j] not in edge_d:
                        g += G(x_cur[i,:],x_cur[j,:])*net.get_edge_data(i,j)['weight']
                    sg = 0
                    for j in range(Sd.shape[0]):
                        if i == Sd[j,0]:
                            xj = x_cur[int(Sd[j,1]),:]
                            xk = x_cur[int(Sd[j,2]),:]
                            sg += SG(x_cur[i,:], xj, xk)
                    dx = (f + epsilon1*g + epsilon2*sg)*ddt 
                    x_cur[i,:] = x_cur[i,:]+dx
                if (it+1)%deln == 0:
                    x[int((it+1)/deln),:,:] = x_cur
            Xs[Ni*T:(Ni+1)*T,:,0] = np.sin(x)[:,:,0]
            Xs[Ni*T:(Ni+1)*T,:,1] = np.cos(x)[:,:,0]
            theta[Ni*T:(Ni+1)*T,:] = x[:,:,0]
        Xss = np.zeros((T*N*V,n))
        for Vi in range(V):
            Xss[Vi*T*N:(Vi+1)*T*N,:] = Xs[:,:,Vi]
        return(Xss,theta)
    
    def sav_data(self, Xss, theta):
        time_point = np.arange(self.T)*self.dt
        Xss = pd.DataFrame(Xss)
        time_point = pd.DataFrame(time_point)
        Xss.to_csv("./dataset/data/trajectory.csv")
        time_point.to_csv("./dataset/data/time_point.csv")
        edges = self.net.edges()
        edges = pd.DataFrame(edges())
        edges.to_csv("./dataset/data/edges.csv")
        
        theta = pd.DataFrame(theta)
        theta.to_csv("./dataset/data/theta.csv")
        w = pd.DataFrame(self.w)
        w.to_csv("./dataset/data/w.csv")
        Sd = pd.DataFrame(self.Sd)
        Sd.to_csv("./dataset/data/Sd.csv")
        
        net = self.net
        weights = []
        for u,v in net.edges():
            weights.append(net.edges[u,v]['weight'])
        weights = pd.DataFrame(weights)
        weights.to_csv('./dataset/data/weights.csv')
        
        edges_in = pd.DataFrame(np.array([]))
        #edges_in = pd.DataFrame(np.array([[1,1]]))
        edges_in.to_csv("./dataset/data/edges_in.csv")
        #edges_ex = pd.DataFrame(np.array([[3],[3],[3]]))
        edges_ex = pd.DataFrame(np.array([[1,1,2],[1,2,1],[2,1,2],[2,2,1]]))
        edges_ex.to_csv("./dataset/data/edges_ex.csv")
        
        print("Data generated successfully!")













