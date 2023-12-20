import numpy as np
from scipy import sparse
import torch

class Reservoir(object):
    """
    Build a reservoir and evaluate internal states
    """
    
    def __init__(self, n_internal_units=100, spectral_radius=0.99, leak=0.4,
                 connectivity=0.3, input_scaling=0.2, noise_level=0.01, args=None):
        
        # Initialize attributes
        self._n_internal_units = n_internal_units
        self._input_scaling = input_scaling
        self._noise_level = noise_level
        self.leak = leak
        self.args = args
        
        # Input weights depend on input size: they are set when data is provided
        self._input_weights = None

        # Generate internal weights
        self._internal_weights = self._initialize_internal_weights(
            n_internal_units,
            connectivity,
            spectral_radius)
        
    def _initialize_internal_weights(self, n_internal_units,
                                     connectivity, spectral_radius):
        
        print("initialize_internal_weights")
        internal_weights = np.zeros((n_internal_units,n_internal_units))
        # Generate sparse, uniformly distributed weights.
        internal_weights[:,:] = sparse.rand(n_internal_units,
                                       n_internal_units,
                                       density=connectivity).todense()

        # Ensure that the nonzero values are uniformly distributed in [-0.5, 0.5]
        internal_weights[np.where(internal_weights > 0)] -= 0.5
        
        # Adjust the spectral radius.
        E, _ = np.linalg.eig(internal_weights)
        e_max = np.max(np.abs(E))
        internal_weights /= np.abs(e_max)/spectral_radius       
        
        return internal_weights
    
    def _compute_netx_state(self, previous_state, current_input):
        _n_internal_units = self._n_internal_units 
        _internal_weights = self._internal_weights
        _input_weights = self._input_weights 
        _noise_level = self._noise_level
        leak = self.leak
        N, _ = previous_state.shape 
        
        state1 = previous_state 
        state2 = np.zeros((N, _n_internal_units)) 
        state2 = _internal_weights.dot(previous_state.T).T
        state2 += _input_weights.dot(current_input.T).T
        state2 += np.random.rand(_n_internal_units, N).T*_noise_level + self.args.sigma
        state2 = np.tanh(state2)
        state_total = leak*state1 + (1-leak)*state2 
        
        return(state_total)
        

    def _compute_state_matrix(self, Xs, n_drop=0):
        _n_internal_units = self._n_internal_units 
        _internal_weights = self._internal_weights
        _input_weights = self._input_weights 
        _noise_level = self._noise_level
        leak = self.leak
        
        N, n, T, V = Xs.shape 
        previous_state = np.zeros((N, _n_internal_units)) 
        # Storage
        state_matrix = np.zeros((N, T - n_drop, _n_internal_units), dtype=float)
        for t in range(T):
            state1 = previous_state 
            state2 = np.zeros((N, _n_internal_units)) 
            current_input = Xs[:,:,t,:].reshape(N,n*V)
            state2 = _internal_weights.dot(previous_state.T).T
            state2 += _input_weights.dot(current_input.T).T
            state2 += np.random.rand(_n_internal_units, N).T*_noise_level + self.args.sigma
            state2 = np.tanh(state2)
            previous_state = leak*state1 + (1-leak)*state2
            # Store everything after the dropout period
            if (t > n_drop - 1):
                state_matrix[:, t - n_drop, :] = previous_state            
                
        return state_matrix
        
    
    def get_states(self, Xs, n_drop=0, bidir=True):
        
        N, n, T, V = Xs.shape
        if self._input_weights is None:
            print("initialize_input_weights")
            #self._input_weights = (2.0*np.random.binomial(1, 0.5 , [self._n_internal_units, n*V]) - 1.0)*self._input_scaling
            self._input_weights = (2.0*np.random.random(size=(self._n_internal_units, n*V)) - 1.0)*self._input_scaling
        
        # compute sequence of reservoir states
        states = self._compute_state_matrix(Xs, n_drop)
        
        # reservoir states on time reversed input
        if bidir is True:
            X_r = Xs[:, :, ::-1, :]
            states_r = self._compute_state_matrix(X_r, n_drop)
            states = np.concatenate((states, states_r), axis=3)

        return states