import theano
import theano.tensor as T
import numpy as np
from theano import sandbox

class layer(object):
    #code
    def __init__(self,n_in,n_out,W_init,b_init):
        
        self.n_in = n_in
        self.n_out = n_out
        
        scale = 0.0001
        
        if W_init is None:
            self.W = theano.shared(
        			value=(scale*np.random.randn(self.n_in,self.n_out)).astype(theano.config.floatX),
        			name = 'W',
        			borrow = True)
        else:
            self.W = theano.shared(
        			value=W_init,
        			name = 'W',
        			borrow = True)
        
        if b_init is None:
            self.b = theano.shared(
            		value = np.zeros((self.n_out,),dtype=theano.config.floatX),
            		name = 'b',
            		borrow = True)
        else:
            self.b = theano.shared(
            		value = b_init.astype(theano.config.floatX),
            		name = 'b',
            		borrow = True)
            
        self.param = [self.W,self.b]
				
    def output(self,x):
        return T.nnet.sigmoid(T.dot(x,self.W)+self.b)
    
    def current_output(self,data):
        
        x = T.matrix('X')
        y = T.matrix('y')
        
        y = self.output(x)
        #outcome = theano.function([x], sandbox.cuda.basic_ops.gpu_from_host(y))
        outcome = theano.function([x], y)
        return outcome(data)    
    
        
        
