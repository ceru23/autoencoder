import theano
import theano.tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
from theano import sandbox 

class rbm(object):
    #code
    def __init__(self,W_init,b_vis_init,b_hid_init,k):
        
        
      
        assert W_init is not None, 'Error! W must be passed as an argument'
        assert not (W_init.shape[0] != b_vis_init.shape[0]) | (W_init.shape[1] != b_hid_init.shape[0]), 'Error! Dimensions of W, b_hid, and b_vis have to be consistent'       #CHECK DIM!!!!
        self.W = theano.shared(value=W_init,name='W',borrow=True)
        self.b_vis = theano.shared(value=b_vis_init,name='b_vis',borrow=True)
        self.b_hid = theano.shared(value=b_hid_init,name='b_hid',borrow=True)
        self.k = k
        
       
        
        self.params = [self.W,self.b_vis,self.b_hid]
        
        np.random.seed(0xbeef)
        rng = RandomStreams(seed=np.random.randint(1 << 30))
        
        self.rng = rng
        self.updates = []
        
    def gibbs_step(self,v):
        mean_h = T.nnet.sigmoid(T.dot(v,self.W)+self.b_hid)
        h = self.rng.binomial(size=mean_h.shape, n=1, p=mean_h, dtype=theano.config.floatX)
        mean_v = T.nnet.sigmoid(T.dot(h,self.W.T)+self.b_vis)
        v = self.rng.binomial(size=mean_v.shape, n=1, p=mean_v, dtype=theano.config.floatX)
        
        return mean_v,v,mean_h,h
   
    def free_energy(self,v):
        return -(v * self.b_vis).sum() - T.log(1 + T.exp(T.dot(self.v, self.W) + self.b_hid)).sum()
    
    
    def run_rbm(self,v):
        chain, updates = theano.scan(lambda v: self.gibbs_step(v)[1], outputs_info=[v],
                                 n_steps=self.k)
        v_sample = chain[-1]

        mean_v = self.gibbs_step(v_sample)[0]
        h = self.gibbs_step(v_sample)[3]
        mean_h = self.gibbs_step(v_sample)[2]
        monitor = T.xlogx.xlogy0(v, mean_v) + T.xlogx.xlogy0(1 - v, 1 - mean_v)
        monitor = monitor.sum() / v.shape[0]
        self.updates = updates
        
        cost = (self.free_energy(v) - self.free_energy(v_sample)) / v.shape[0]
        
        gparams = T.grad(cost,self.params,consider_constant=[v_sample])
        
        for gparam,param in zip(gparams,self.params): self.updates[param] = param-gparam*0.5
        
        return v_sample,monitor,cost,updates,h,mean_h

    

    def free_energy(self,v):
        return -(v * self.b_vis).sum() - T.log(1 + T.exp(T.dot(v, self.W) + self.b_hid)).sum()
    
    
    def train(self,data):
        x = T.matrix('x')
        #sandbox.cuda.basic_ops.gpu_from_host(x)
        v,monitor,cost,updates,h,mean_h = self.run_rbm(x)

    
        train = theano.function(
        inputs = [x],
        outputs = [cost],
        updates = updates)
    
        train(data)
        
        
    
    

'''
import timeit

if __name__ == '__main__':
    
    n_vis = 6
    n_hid = 2
    n_iter = 100
    k = 50
    
    #W = theano.shared(value=np.random.randn(6,2).astype(theano.config.floatX),borrow=True)
    #b_vis = theano.shared(value = np.ones((n_vis,),dtype=theano.config.floatX),borrow = True)
    #b_hid = theano.shared(value = np.ones((n_hid,),dtype=theano.config.floatX),borrow = True)
    W = np.random.randn(6,2).astype(theano.config.floatX)
    b_vis = np.ones((n_vis,),dtype=theano.config.floatX)
    b_hid = np.ones((n_hid,),dtype=theano.config.floatX)
    
    r = rbm(W,b_vis,b_hid,k)
 
    print 'W_init: ',r.W.get_value()
    
    data = np.array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,0],[0,0,1,1,1,0]]).astype(theano.config.floatX)


    t1 = timeit.default_timer()
    for i in range(n_iter):
        r.train(data)
    t2 = timeit.default_timer()
    
    print 'W_final: ', r.W.get_value(), 'in ', t2-t1
    '''
     
        
