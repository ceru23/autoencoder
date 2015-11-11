import theano
import theano.tensor as T
import numpy as np
from theano import sandbox

class layer(object):
    #code
    def __init__(self,n_in,n_out,W_init,b_init,activation='sigmoid'):
        
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        
        #scale = 0.0001
        scale = 1
        
        if W_init is None:
            self.W = theano.shared(
        			value=(scale*np.random.randn(self.n_in,self.n_out)).astype(theano.config.floatX),
        			name = 'W',
        			borrow = True)
        else:
            self.W = theano.shared(
        			value=W_init*scale,
        			name = 'W',
        			borrow = True)
        
        if b_init is None:
            self.b = theano.shared(
            		value = np.zeros((self.n_out,),dtype=theano.config.floatX),
            		name = 'b',
            		borrow = True)
        else:
            self.b = theano.shared(
            		value = (b_init*scale).astype(theano.config.floatX),
            		name = 'b',
            		borrow = True)
            
        self.param = [self.W,self.b]
				
    def output(self,x):
        if(self.activation=='sigmoid'):
            return T.nnet.sigmoid(T.dot(x,self.W)+self.b)
        elif (self.activation=='relu'):
            return T.switch((T.dot(x,self.W)+self.b)<0, 0, T.dot(x,self.W)+self.b)
        elif(self.activation=='softplus'):
            return T.nnet.softplus(T.dot(x,self.W)+self.b)
        else:
            return T.nnet.sigmoid(T.dot(x,self.W)+self.b)
        #return T.nnet.softplus(T.dot(x,self.W)+self.b)
    def current_output(self,data):
        
        x = T.matrix('X')
        y = T.matrix('y')
        
        y = self.output(x)
        #outcome = theano.function([x], sandbox.cuda.basic_ops.gpu_from_host(y))
        outcome = theano.function([x], y)
        return outcome(data)    
    
        
        
