
import theano
import theano.tensor as T
import numpy as np
import timeit
import math
import rbm
import layer
import pickle
import matplotlib.pyplot as plt 


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
            
                
    def layers_from_file(self,param_list):
        pass
    
        
            
    def generate_layers(self,units_for_layers):
        layers = []
        
        for i in range(len(units_for_layers)-1):
            
            layers.append(layer.layer(units_for_layers[i],units_for_layers[i+1],None,None,'sigmoid'))
            
            
        return layers
            
        
    def generate_decoder(self):
        decoder = []
        for l in reversed(range(self.n_layers)):
            b = np.zeros((self.layers[l].n_in,),dtype=theano.config.floatX)
            l_t = layer.layer(self.layers[l].n_out,self.layers[l].n_in,(self.layers[l].W.get_value()).T,b,'sigmoid')
            #l_t = layer.layer(self.layers[l].n_out,self.layers[l].n_in,(self.layers[l].W.get_value()).T,b,'relu')
            ###activation for decodor not squashing as in Bengio et al. 2010
            decoder.append(l_t)
        self.layers.extend(decoder)
        
        
        
        for la in self.layers:
            self.param += la.param        
      
      
        
    def output(self,x,lev=0):
        if(lev==len(self.layers)-1):
            return self.layers[-1].output(x)
        else:
            return self.output(self.layers[lev].output(x),lev+1)
        
        
   # def get_weight(self):
    #    return T.sum([T.sum(T.sqrt(param)**2) for param in self.param])
        
        
    
    def get_reduced_data(self,x,lev=0):
        if(lev==self.n_layers-1):
            return self.layers[lev].output(x)
        else:
            return self.get_reduced_data(self.layers[lev].output(x),lev+1)

    def add_noise(self,data,corruption_level=0.25):
        ####aggiungo rumore mettendo a zero certe coordinate
        return (np.random.normal(0.0,corruption_level,data.shape) + data).astype(theano.config.floatX)

    def train_auto(self,data,learning_rate=1,reg_weight=0.25):
        
        x = T.matrix('x')
        x_hat = T.matrix('x_hat')
    
        
        x_hat = self.output(x)
       
        
      
        #reg_weight = T.sum([T.sqrt(T.sum(param.get_value())**2) for param in self.param])
        
        
        log_cost = T.mean(T.sqrt(T.abs_(-T.sum(x * T.log(x_hat) + (1 - x) * T.log(1 - x_hat)))))
        weight = reg_weight*T.sqrt(T.sum([T.sum(param**2) for param in self.param]))
        lin_cost = T.mean(T.sqrt((x-x_hat)**2))
        
        #cost = lin_cost+log_cost+weight
        
        cost = lin_cost
        
        #cost = T.sum(T.sqrt((x-x_hat)**2))+weight
        gparams = [T.grad(cost,param) for param in self.param] 
    
        updates = [(par, par-learning_rate*gpar) for par,gpar in zip(self.param,gparams)]
        
        train = theano.function(
            inputs = [x],
            #outputs = [cost,lin_cost,log_cost,weight],
            outputs = [cost],
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
    
    def save_model(self,f):
        #p = [(par.name,par.get_value()) for par in self.param]
        
       
        pickle.dump(self, f)
        
        
        
        pass
        
    
if __name__ == '__main__':
    
    
   # with open('swiss.dat') as f:
    #    data = [[float(x) for x in line.split()] for line in f]
    #data = np.asarray(data).astype(theano.config.floatX)
    data = np.loadtxt("swiss.dat",dtype=theano.config.floatX)
    #units = [3,9,7,2]
   #units = [3,6,2]
    #data = np.random.randn(100,6).astype(theano.config.floatX)
    
    learning_rate = 20
    iters = 10000
    int_dim = 2
    f = open("model.dat","w")
    
    
    units = [data.shape[1],int(math.ceil(data.shape[1]*1.2))+5,int(max(math.ceil(data.shape[1]/4),int_dim+2)+3),
             int(max(math.ceil(data.shape[1]/10),int_dim+1)),int_dim]
    
      
    auto = autoencoder(units)

    tw1 = timeit.default_timer()
    auto.pre_train_layers(0,data,5,20)
    tw2 = timeit.default_timer()
    
    auto.generate_decoder()
    
  #  p = [(param.name,param.get_value()) for param in auto.param]
   # print p,
   
    
    
        
    t1 = timeit.default_timer()
   
    init_error = auto.train_auto(data)[1]
    
    best = init_error
    auto.save_model(f)
    
    for i in range(iters):
        #if (i%2==0):
         #   noise_data = auto.add_noise(data)
          #  err = auto.train_auto(noise_data)[1]
        #else:
       #if(i%10==0):
        #    plt.cla()
         #   red = auto.get_hidden_data(data)
          #  plt.plot(red[:,0], red[:,1],'o')
           # plt.show()
        
        
        cost,err = auto.train_auto(data)
        p = [np.sum(np.sqrt(param.get_value()**2)) for param in auto.param]
        print 'Weights: ',sum(p)
        print 'Error: ',err
        print 'Cost: ',cost
        if(err<best):
            best = err
            auto.save_model(f)
            print "Best model so far found at iter: ",i
       
    print 'Init error: ', init_error
    
    r = open("model.dat","r")
    
    best_auto = pickle.Unpickler(r).load()
    
    
       
    
    print 'Final error: ',best_auto.train_auto(data)[1]
    
    t2 = timeit.default_timer()
    
    print 'Elapsed time for pretraining: ',tw2-tw1
    print 'Elapsed time for training: ',t2-t1
    p = [np.sum(np.sqrt(param.get_value()**2)) for param in best_auto.param]
    print 'Weights: ',sum(p)
    np.savetxt("reduced.dat",best_auto.get_hidden_data(data))
    np.savetxt("out.dat",best_auto.run_auto(data))
    

    
    
    
    