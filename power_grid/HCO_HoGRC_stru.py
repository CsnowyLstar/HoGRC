import argparse
import numpy as np
import importlib
import networkx as nx
import pandas as pd
from scipy import sparse
from scipy.sparse import csc_matrix,find
import torch
from sklearn.linear_model import Ridge
import joblib 
import matplotlib.pyplot as plt

def args(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--device', type=str, default='cpu') 
    parser.add_argument('--model_ind', type=str, default='HoGRC') 
    parser.add_argument('--data_ind', type=str, default='HCO') 
    parser.add_argument('--net_nam', type=str, default='edges') 
    parser.add_argument('--direc', type=bool, default=True) 
    parser.add_argument('--nj', type=int, default=33) 
    #Parameters of experimental data 
    parser.add_argument('--N', type=int, default=1) 
    parser.add_argument('--n', type=int, default=120) 
    parser.add_argument('--T', type=int, default=10000)
    parser.add_argument('--V', type=int, default=2)
    parser.add_argument('--dt', type=float, default=0.08)
    parser.add_argument('--ddt', type=float, default=0.01)
    parser.add_argument('--couple_str', type=float, default=0.4)
    parser.add_argument('--sigma', type=float, default=1)
    parser.add_argument('--noise_sigma', type=float, default=0)
    parser.add_argument('--ob_noise', type=float, default=0)
    parser.add_argument('--qtr', type=float, default=0.8)
    parser.add_argument('--threshold', type=float, default=0.1)
    #Parameters of RC 
    parser.add_argument('--warm_up', type=int, default=100)
    parser.add_argument('--n_internal_units', type=int, default=500)
    parser.add_argument('--spectral_radius', type=float, default=0.8)
    parser.add_argument('--leak', type=float, default=0.1)
    parser.add_argument('--leak1', type=float, default=0.4)
    parser.add_argument('--leak2', type=float, default=0.3)
    parser.add_argument('--connectivity', type=float, default=0.02)
    parser.add_argument('--input_scaling', type=float, default=0.5) 
    parser.add_argument('--noise_level', type=float, default=0.00) 
    parser.add_argument('--circle', type=bool, default=False)
    parser.add_argument('--alpha', type=float, default=10**(-4))
    #Parameters of other methods 
    parser.add_argument('--epochs', type=int, default=300) 
    parser.add_argument('--batchs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    args = parser.parse_args(args=[])
    return(args) 

def conv(x):
    r=0
    rs=[]
    while(x!=0):
        r=x%2
        x=x//2
        rs=[r]+rs
    rs = rs[::-1]
    loc = []
    for i in range(len(rs)):
        if rs[i]==1:
            loc.append(i)
    return(np.array(loc))

##### read data #####
def read_data():
    the = pd.read_csv("./dataset/data/theta.csv").values[:,1:].transpose()
    theta = np.zeros((args.N, args.n, args.T, 1))
    theta[0,:,:,0] = the
    Sd = pd.read_csv("./dataset/data/Sd.csv").values[:,1:]
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
    
    X = Xsn[0,:,:,0].numpy()
    #plt.plot(X[3].T,'-o')
    return(Xsn,theta,time_point,edges_in,edges_ex,edges_out,Sd)

##### reservoir #####
def _initialize_internal_weights(args,input_weights,rec):
    n_internal_units = args.n_internal_units
    spectral_radius = args.spectral_radius
    connectivity = args.connectivity
    V = args.V
    print("initialize_internal_weights")
    
    row = []
    col = []
    data = []
    val = 0
    for i in range(len(rec)-1):
        sta = rec[i]
        end = rec[i+1]
        if end<sta:
            val = val+V*n_internal_units 
        else:
            e_max = 0
            while(e_max==0):
                weight = sparse.rand(end-sta,end-sta,density=connectivity).todense() 
                weight[np.where(weight > 0)] -= 0.5
                # Adjust the spectral radius.
                E, _ = np.linalg.eig(weight)
                e_max = np.max(np.abs(E))
                weight /= np.abs(e_max)/spectral_radius      
            for u in range(sta+val,end+val):
                for v in range(sta+val,end+val):
                    wei = weight[u-sta-val,v-sta-val]
                    if wei != 0:
                        row.append(u)
                        col.append(v)
                        data.append(wei)
    gn = args.n*V*n_internal_units
    internal_weights = csc_matrix((data, (row, col)), shape=(gn, gn))
    return internal_weights 

def _initialize_input_weights(args,edges_in,edges_out,edges_ex):
    rec = []
    records = []
    input_weights = np.zeros((args.n*args.V*args.n_internal_units, args.n*args.V))
    
    for ni in range(args.n):
        sta = 0
        rec.append(sta)
        records.append(sta)
        input_weight = np.zeros((args.V*args.n_internal_units, args.n*args.V))
        us = []
        vs = []
        
        for edge in edges_out:
            if edge[0] == ni and edge[0] != args.nj:
                for ex in edges_ex:
                    us.append(conv(ex[0])+ni*args.V) 
                    inout = conv(ex[2])+edge[1]*args.V 
                    if ex[1]>0:
                        inout = np.concatenate((inout,conv(ex[1])+ni*args.V),axis=0)
                    vs.append(inout)
        
        if ni == args.nj:
            for edge1 in o3:
                for ex in edges_ex:
                    us.append(conv(ex[0])+ni*args.V) 
                    inout = conv(ex[2])+edge1*args.V 
                    if ex[1]>0:
                        inout = np.concatenate((inout,conv(ex[1])+ni*args.V),axis=0)
                    vs.append(inout)                        
        
        for edge in Sd:
            if edge[0] == ni and edge[0] != args.nj:
                for ex in [1,2]:
                    us.append(conv(ex)+ni*args.V)
                    inout = conv(3)+edge[1]*args.V 
                    inout = np.concatenate((inout,conv(3)+edge[2]*args.V))
                    inout = np.concatenate((inout,conv(3)+ni*args.V))
                    vs.append(inout)
        
        if ni == args.nj:
            if len(ho3)>0:
                for ex in [1,2]:
                    us.append(conv(ex)+ni*args.V)
                    inout = conv(3)+ni*args.V 
                    for hoi in range(len(ho3)):
                        inout = np.concatenate((inout,conv(3)+ho3[hoi]*args.V))
                    vs.append(inout)                        
                    
        lov = ni*args.V
        vsi = []
        for i in range(len(us)):
            if us[i]==lov:
                vsi.append(vs[i])
        num = len(vsi) 
        locs = []
        k = 0
        for i in range(num):
            loc = []
            for j in vsi[i]:
                loc.append(j)
            if len(loc)>0:
                locs.append(loc)
                k = k + len(loc)
        le = int(2*args.n_internal_units/num)
        for loc in locs: 
            weight = (2.0*np.random.random(size=(le,len(loc))) - 1.0)*args.input_scaling
            end = sta + le
            input_weight[sta:end,loc] = weight
            sta = end 
            if loc == locs[-1]:
                sta = 2*args.n_internal_units
            rec.append(sta)
        records.append(sta)
        input_weights[ni*args.V*args.n_internal_units:(ni+1)*args.V*args.n_internal_units,:] = input_weight
                
    row = []
    col = []
    data = []
    for i in range(input_weights.shape[0]):
        for j in range(input_weights.shape[1]):
            if input_weights[i,j] != 0:
                row.append(i)
                col.append(j)
                data.append(input_weights[i,j])
    input_weights = csc_matrix((data, (row, col)), shape=(input_weights.shape[0], input_weights.shape[1]))
    return input_weights,rec,records

def _compute_netx_state(internal_weights, input_weights, previous_state, current_input):
    n = args.n
    V = args.V
    n_internal_units = args.n_internal_units 
    noise_level = args.noise_level
    leak = args.leak
    N, _ = previous_state.shape 
    
    state1 = previous_state 
    state2 = np.zeros((N, n*V*n_internal_units)) 
    state2 = internal_weights.dot(previous_state.T).T
    state2 += input_weights.dot(current_input.reshape(N,n*V).T).T
    state2 += np.random.rand(n*V*n_internal_units, N).T*noise_level + args.sigma
    state2 = np.tanh(state2)
    state_total = leak*state1 + (1-leak)*state2 
    return(state_total)

def _compute_state_matrix(internal_weights, input_weights, Xsn, n_drop=0):
    n_internal_units = args.n_internal_units 
    noise_level = args.noise_level
    leak = args.leak
    
    N, n, T, V = Xsn.shape 
    previous_state = np.zeros((N, n*V*n_internal_units)) 
    # Storage
    state_matrix = np.empty((N, T - n_drop, n*V*n_internal_units), dtype=float)
    for t in range(T):
        state1 = previous_state 
        state2 = np.zeros((N, n*V*n_internal_units)) 
        current_input = Xsn[:,:,t,:].reshape(N,n*V)
        state2 = internal_weights.dot(previous_state.T).T
        state2 += input_weights.dot(current_input.T).T
        state2 += np.random.rand(n*V*n_internal_units, N).T*noise_level + args.sigma
        state2 = np.tanh(state2)
        previous_state = leak*state1 + (1-leak)*state2
        # Store everything after the dropout period
        if (t > n_drop - 1):
            state_matrix[:, t - n_drop, :] = previous_state            
    return state_matrix

def get_states(internal_weights, input_weights, Xsn, n_drop=0, bidir=True):
    N, n, T, V = Xsn.shape
    
    # compute sequence of reservoir states
    states = _compute_state_matrix(internal_weights, input_weights, Xsn, n_drop)
    
    # reservoir states on time reversed input
    if bidir is True:
        X_r = Xsn[:, :, ::-1, :]
        states_r = _compute_state_matrix(X_r, n_drop)
        states = np.concatenate((states, states_r), axis=3)

    return states

def draw1(args,preds,preds2,preds3,start2,start3,steps,nj):
    warm_up = args.warm_up
    ds = steps
    #draw
    X = Xsn.numpy()
    font1 = {'family':'Times New Roman', 'weight':'normal','size':25}
    
    fig = plt.figure(figsize=(20,12))
    dimension = 0
    ax11 = fig.add_subplot(args.V,2,1)
    be = start2-ds
    la = start2+ds
    ax11.plot(np.arange(be,la), X[0,nj,be:la,dimension],'k-o',markersize=7,label='True')
    ax11.plot(np.arange(be,la), preds[nj,be-warm_up:la-warm_up,dimension],'y-x',markersize=5,label='One-step prediction')
    ax11.plot(np.arange(start2,start2+steps), preds2[nj,:,dimension],'r-o',markersize=3,label='Multi-step interpolation prediction')
    ax11.tick_params(labelsize=15)
    ax11.set_ylabel(r'$x_A$',size=25)
    #ax11.legend(prop=font1)
    
    ax12 = fig.add_subplot(args.V,2,2)
    be = start3-ds
    la = start3+ds
    ax12.plot(np.arange(be,la), X[0,nj,be:la,dimension],'k-o',markersize=7,label='True')
    ax12.plot(np.arange(be,la), preds[nj,be-warm_up:la-warm_up,dimension],'y-x',markersize=5,label='One-step prediction')
    ax12.plot(np.arange(start3,start3+steps), preds3[nj,:,dimension],'g-o',markersize=3,label='Multi-step extrapolation prediction')
    ax12.tick_params(labelsize=15)
    ax12.set_ylabel(r'$x_A$',size=25)
    #ax12.legend(prop=font1)
    
    if args.V>1:
        dimension = 1
        ax21 = fig.add_subplot(args.V,2,3)
        be = start2-ds
        la = start2+ds
        ax21.plot(np.arange(be,la), X[0,nj,be:la,dimension],'k-o',markersize=7,label='True')
        ax21.plot(np.arange(be,la), preds[nj,be-warm_up:la-warm_up,dimension],'y-x',markersize=5,label='One-step prediction')
        ax21.plot(np.arange(start2,start2+steps), preds2[nj,:,dimension],'r-o',markersize=3,label='Multi-step interpolation prediction')
        ax21.tick_params(labelsize=15)
        ax21.set_ylabel(r'$y_A$',size=25)
        
        ax22 = fig.add_subplot(args.V,2,4)
        be = start3-ds
        la = start3+ds
        ax22.plot(np.arange(be,la), X[0,nj,be:la,dimension],'k-o',markersize=7,label='True')
        ax22.plot(np.arange(be,la), preds[nj,be-warm_up:la-warm_up,dimension],'y-x',markersize=5,label='One-step prediction')
        ax22.plot(np.arange(start3,start3+steps), preds3[nj,:,dimension],'g-o',markersize=3,label='Multi-step extrapolation prediction')
        ax22.tick_params(labelsize=15)
        ax22.set_ylabel(r'$y_A$',size=25)
        #ax1.legend(prop=font1)
    
    if args.V>2:
        dimension = 2
        ax31 = fig.add_subplot(args.V,2,5)
        be = start2-ds
        la = start2+ds
        ax31.plot(np.arange(be,la), X[0,nj,be:la,dimension],'k-o',markersize=7,label='True')
        ax31.plot(np.arange(be,la), preds[nj,be-warm_up:la-warm_up,dimension],'y-x',markersize=5,label='One-step prediction')
        ax31.plot(np.arange(start2,start2+steps), preds2[nj,:,dimension],'r-o',markersize=3,label='Multi-step interpolation prediction')
        ax31.tick_params(labelsize=15)
        ax31.set_ylabel(r'$z_A$',size=25)
        ax31.set_xlabel('t',size=25)
        
        ax32 = fig.add_subplot(args.V,2,6)
        be = start3-ds
        la = start3+ds
        ax32.plot(np.arange(be,la), X[0,nj,be:la,dimension],'k-o',markersize=7,label='True')
        ax32.plot(np.arange(be,la), preds[nj,be-warm_up:la-warm_up,dimension],'y-x',markersize=5,label='One-step prediction')
        ax32.plot(np.arange(start3,start3+steps), preds3[nj,:,dimension],'g-o',markersize=3,label='Multi-step extrapolation prediction')
        ax32.tick_params(labelsize=15)
        ax32.set_ylabel(r'$z_A$',size=25)
        ax32.set_xlabel('t',size=25)
        #ax1.legend(prop=font1)  

def test1(args):
    ntr = int(args.T*args.qtr)

    input_weights,rec,records = _initialize_input_weights(args, edges_in, edges_out, edges_ex)
    internal_weights = _initialize_internal_weights(args,input_weights,rec)
    
    res_states = get_states(internal_weights, input_weights, Xsn, n_drop=0, bidir=False)
    Ni = 0
    # train
    warm_up = args.warm_up 
    X_train = res_states[Ni,warm_up:ntr,:]
    Y_train = (theta[Ni,:,warm_up+1:ntr+1,:]-theta[Ni,:,warm_up:ntr,:])/args.dt
    for ni in range(args.n):
        X = X_train[:,(ni*args.V)*args.n_internal_units:((ni+1)*args.V)*args.n_internal_units]
        Y = Y_train[ni,:,:]
        readout = Ridge(alpha=args.alpha)
        readout.fit(X,Y)
        joblib.dump(readout,'./models/model/readout'+str(ni)+'.pkl')
    print("Training complete")
    
    # test
    preds = np.zeros((args.n, args.T-args.warm_up, args.V))
    pre_theta = np.zeros((args.n, args.T-args.warm_up, 1))
    for ni in range(args.n):
        readout = joblib.load('./models/model/readout'+str(ni)+'.pkl')
        for i in range(args.T-args.warm_up):
            X = res_states[:,i+warm_up-1,(ni*args.V)*args.n_internal_units:((ni+1)*args.V)*args.n_internal_units]
            pre_theta[ni,i,0] = readout.predict(X)[0][0]*args.dt+theta[0,ni,i+warm_up-1,0]
            preds[ni,i,0] = np.sin(pre_theta[ni,i,0])
            preds[ni,i,1] = np.cos(pre_theta[ni,i,0])
    error = (Xsn[:,:,args.warm_up:args.T,:] - preds).numpy()
    
    nj = args.nj
    print("Total error:", np.mean(np.abs(error)))
    print("Train error:", np.mean(np.abs(error[:,:,:ntr-args.warm_up,:])), np.mean(np.abs(error[:,nj,:ntr-args.warm_up,:])))
    print("Test error:", np.mean(np.abs(error[:,:,ntr-args.warm_up:,:])), np.mean(np.abs(error[:,nj,ntr-args.warm_up:,:])))
    return(internal_weights, input_weights, preds, error)

def test2(args, internal_weights, input_weights, start2s, start3s, steps):
    N,n,V = args.N,args.n,args.V
    
    res_states = get_states(internal_weights, input_weights, Xsn, n_drop=0, bidir=False) 

    preds2s = np.zeros((len(start2s),n,steps,V))
    error2s = np.zeros((len(start2s),n,steps,V))
    for i in range(len(start2s)):
        start2 = start2s[i]        
        preds2 = np.zeros((n,steps,V)) 
        pre_theta2 = np.zeros((n,steps,1))
        previous_states = res_states[:,start2-1,:]
        current_input = np.zeros((N,n,V))
        current_input[:,:,:] = Xsn[:,:,start2-1,:].numpy()
        current_theta = theta[0,:,start2-1,0]
        for j in range(steps):
            for ni in range(n):
                readout = joblib.load('./models/model/readout'+str(ni)+'.pkl')
                X = previous_states[:,(ni*args.V)*args.n_internal_units:((ni+1)*args.V)*args.n_internal_units]
                current_theta[ni] = readout.predict(X)[0][0]*args.dt + current_theta[ni]
                current_input[:,ni,0] = np.sin(current_theta[ni])
                current_input[:,ni,1] = np.cos(current_theta[ni])
            preds2[:,j,:] = current_input[0] 
            pre_theta2[:,j,0] = current_theta
            previous_states = _compute_netx_state(internal_weights, input_weights, 
                                                      previous_states, current_input)
        error2 = (Xsn[0,:,start2:(start2+steps),:] - preds2).numpy()
        preds2s[i] = preds2
        error2s[i] = error2  
     
    preds3s = np.zeros((len(start3s),n,steps,V))
    error3s = np.zeros((len(start3s),n,steps,V))
    for i in range(len(start3s)):
        start3 = start3s[i]        
        preds3 = np.zeros((n,steps,V)) 
        pre_theta3 = np.zeros((n,steps,1))
        previous_states = res_states[:,start3-1,:]
        current_input = np.zeros((N,n,V))
        current_input[:,:,:] = Xsn[:,:,start3-1,:].numpy()
        current_theta = theta[0,:,start3-1,0]
        for j in range(steps):
            for ni in range(n):
                readout = joblib.load('./models/model/readout'+str(ni)+'.pkl')
                X = previous_states[:,(ni*args.V)*args.n_internal_units:((ni+1)*args.V)*args.n_internal_units]
                current_theta[ni] = readout.predict(X)[0][0]*args.dt + current_theta[ni]
                current_input[:,ni,0] = np.sin(current_theta[ni])
                current_input[:,ni,1] = np.cos(current_theta[ni])
            preds3[:,j,:] = current_input[0] 
            pre_theta3[:,j,0] = current_theta
            previous_states = _compute_netx_state(internal_weights, input_weights, 
                                                      previous_states, current_input)
        error3 = (Xsn[0,:,start3:(start3+steps),:] - preds3).numpy()
        preds3s[i] = preds3
        error3s[i] = error3 
    
    nj = args.nj
    print("Error2s:", np.mean(np.abs(error2s)), np.mean(np.abs(error2s[:,nj,:,:])))
    print("Error3s:", np.mean(np.abs(error3s)), np.mean(np.abs(error3s[:,nj,:,:])))
    return(preds2s,preds3s,error2s,error3s)

def VPT(error2s,error3s,steps,num):
    threshold = args.threshold
    n,V,dt = args.n,args.V,args.dt
    X = Xsn[0].numpy()
    sigmas = np.zeros((n,V))
    for ni in range(n):
        for j in range(V):
            sigmas[ni,j] = np.std(X[ni,:,j])
    rmse2 = np.zeros((n,num,steps))
    rmse3 = np.zeros((n,num,steps))
    for ni in range(n):
        for j in range(V):
            error2 = error2s[:,ni,:,j]
            error3 = error3s[:,ni,:,j]
            sigma = sigmas[ni,j]
            rmse2[ni] += (error2/sigma)**2
            rmse3[ni] += (error3/sigma)**2
    rmse2 = rmse2/(V)
    rmse3 = rmse3/(V)
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
    print('lens2:',np.mean(lens2,axis=1))
    print('lens3:',np.mean(lens3,axis=1))
    return(lens2,lens3)

def tVPT(error2s,error3s,steps,num):
    threshold = args.threshold
    n,V = args.n,args.V
    X = Xsn[0].numpy()
    sigmas = np.zeros((n,V))
    for ni in range(n):
        for j in range(V):
            sigmas[ni,j] = np.std(X[ni,:,j])
    rmse2 = np.zeros((num,steps))
    rmse3 = np.zeros((num,steps))
    for ni in range(n):
        for j in range(V):
            error2 = error2s[:,ni,:,j]
            error3 = error3s[:,ni,:,j]
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

def sav(error,nam):
    ntr = int(args.T*args.qtr)
    nj = args.nj
    er = np.array([np.mean(np.abs(error[:,nj,:ntr-args.warm_up,:])),np.mean(np.abs(error[:,nj,ntr-args.warm_up:,:]))])
    er = pd.DataFrame(er)
    er.to_csv('stru'+nam+'.csv')
    return(er)

if __name__ == '__main__':
    
    seed = 0
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    args = args()   
    
    Xsn, theta, time_point, edges_in, edges_ex, edges_out, Sd = read_data()

    # o3 is used to configure pairwise neighbors, ho3 is used to configure higher-order neighbors.
    # Here is the optimal structural configuration: o3 = np.array([36,50]), ho3 = np.array([30,32])
    o3 = np.array([36,50])
    ho3 = np.array([30,32])

    nam = ''
    for i in o3:
        nam += str(i)
        
    # train and test 
    internal_weights,input_weights,preds,error = test1(args)
    er = sav(error,nam)
    
    '''
    # multi steps prediction
    ntr = int(args.T*args.qtr)
    steps = 1000
    num = 5
    start2s = (np.linspace(args.warm_up+steps,ntr-steps,num+2)[1:-1]).astype(int)
    start3s = (np.linspace(ntr,args.T-steps,num+2)[1:-1]).astype(int)
    preds2s,preds3s,error2s,error3s = test2(args, internal_weights, input_weights, start2s, start3s, steps)
    # VPT
    lens2,lens3 = VPT(error2s,error3s,steps,num)
    lenf2,lenf3 = tVPT(error2s,error3s,steps,num)
    '''
    
    
    
    
