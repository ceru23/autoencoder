
import theano
import theano.tensor as T
import numpy as np
import timeit
import math
import rbm
import layer


class autoencoder(object):
    
    def __init__(self,units_for_level):
        
        
        
        self.layers = self.generate_layers(units_for_level)
        self.param = []
        self.n_layers = len(units_for_level)-1
        
       
       
            
        
        
        
            
    def pre_train_layers(self,lev,data,k=10,iters=10):
           
            
            if(lev==0):
                b_zero = np.ones(self.layers[lev].n_in,).astype(theano.config.floatX)
                r_layer = rbm.rbm(self.layers[lev].W.get_value(),b_zero,self.layers[lev].b.get_value(),k)
                for i in range(iters):
                    r_layer.train(data)
                self.layers[lev].W.set_value(r_layer.W.get_value())               
                if(lev<self.n_layers-1):
                    out = self.layers[lev].current_output(data)
                    self.pre_train_layers(lev+1,out,k,iters)
            else:
                r_layer = rbm.rbm(self.layers[lev].W.get_value(),self.layers[lev-1].b.get_value(),self.layers[lev].b.get_value(),k)
                
                for i in range(iters):
                    r_layer.train(data)
                self.layers[lev].W.set_value(r_layer.W.get_value())
                if(lev<self.n_layers-1):
                     out = self.layers[lev].current_output(data)   
                     self.pre_train_layers(lev+1,out,k,iters)
            
                
       
    
        
            
    def generate_layers(self,units_for_layers):
        layers = []
        
        for i in range(len(units_for_layers)-1):
            
            layers.append(layer.layer(units_for_layers[i],units_for_layers[i+1],None,None))
            
            
        return layers
            
        
    def generate_decoder(self):
        decoder = []
        for l in reversed(range(self.n_layers)):
            b = np.zeros((self.layers[l].n_in,),dtype=theano.config.floatX)
            l_t = layer.layer(self.layers[l].n_out,self.layers[l].n_in,(self.layers[l].W.get_value()).T,b)
            decoder.append(l_t)
        self.layers.extend(decoder)
        
        
        
        for la in self.layers:
            self.param += la.param        
      
      
        
    def output(self,x,lev=0):
        if(lev==len(self.layers)-1):
            return self.layers[-1].output(x)
        else:
            return self.output(self.layers[lev].output(x),lev+1)
        
        
    def get_reduced_data(self,x,lev=0):
        if(lev==self.n_layers-1):
            return self.layers[lev].output(x)
        else:
            return self.get_reduced_data(self.layers[lev].output(x),lev+1)


    def train_auto(self,data,learning_rate=0.1):
        
        x = T.matrix('x')
        x_hat = T.matrix('x_hat')
      
        
        x_hat = self.output(x)
       
        
        cost = T.mean(-T.sum(x * T.log(x_hat) + (1 - x) * T.log(1 - x_hat)))
        gparams = [T.grad(cost,param) for param in auto.param] 
    
        updates = [(par, par-learning_rate*gpar) for par,gpar in zip(auto.param,gparams)]
        
        train = theano.function(
            inputs = [x],
            outputs = cost,
            updates = updates
        )
        
        error = T.mean(T.sqrt((x-x_hat)**2))

        
        mse = theano.function(
            inputs = [x],
            outputs = error
        )
        return train(data),mse(data)
        
        
    def get_hidden_data(self,data):
        x = T.matrix('x')
        y = T.matrix('y')
        
        y = self.get_reduced_data(x)
        
        reduced = theano.function([x],y)
        
        return reduced(data)
        
    def run_auto(self,data):
        x = T.matrix('x')
        x_hat = T.matrix('x_hat')
      
        
        x_hat = self.output(x)
    
        
        output = theano.function([x],x_hat) 
    
        return output(data)
    
if __name__ == '__main__':
    
    
   # with open('swiss.dat') as f:
    #    data = [[float(x) for x in line.split()] for line in f]
    #data = np.asarray(data).astype(theano.config.floatX)
    data = np.loadtxt("swiss.dat",dtype=theano.config.floatX)
    units = [3,5,2]
    #data = np.random.randn(100,6).astype(theano.config.floatX)
    
    learning_rate = 0.1
    iters = 1000
    int_dim = 2
    
    
    
    #units = [data.shape[1],int(math.ceil(data.shape[1]*1.2))+5,int(max(math.ceil(data.shape[1]/4),int_dim+2)+3),int(max(math.ceil(data.shape[1]/10),int_dim+1)),int_dim]
    
      
    auto = autoencoder(units)

  #  tw1 = timeit.default_timer()
   # auto.pre_train_layers(0,data,100,10)
    #tw2 = timeit.default_timer()
    
    auto.generate_decoder()
    
    
   
    
    
        
    t1 = timeit.default_timer()
    
    init_error = auto.train_auto(data)[1]
    
    for i in range(iters):
        print auto.train_auto(data)[1],
    
    print 'Init error: ', init_error
    print 'Final error: ',auto.train_auto(data)[1]
    
    t2 = timeit.default_timer()
    
    #print 'Elapsed time for pretraining: ',tw2-tw1
    print 'Elapsed time for training: ',t2-t1
    
    np.savetxt("reduced.dat",auto.get_hidden_data(data))
    p = [(param.name,param.get_value()) for param in auto.param]    
    np.savetxt("model.dat",p)
    
    #np.savetxt("model.dat",for param in auto.params)
    
    #units = [6,3,2]
    
    #auto = autoencoder(units)
    
  
   
    #auto.pre_train_layers(0,data)
    
    
    #print auto.layers[0].current_output(data)
    
    #auto.generate_decoder()
    
   # for i in auto.layers:
    #    print i.W.get_value().shape
    
    
    #print 'init_error: ',auto.train_auto(data)
    
    #for i in range(iters):
     #   auto.train_auto(data)
    
    #print 'final_error: ',auto.train_auto(data)
    
    #print 'data: ',data
    #print 'rec data: ',auto.run_auto(data)
    

  
    
    #print 'data: ',data
    #print 'rec data with training: ',auto.run_auto(data)
    
    
    
    