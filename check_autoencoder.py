from autoencoder import autoencoder
import theano
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt

assert len(sys.argv)>2, "Must provide autoencoder saved in pickle format and datafile!"

f = open(sys.argv[1],"r")
  
data = np.loadtxt(sys.argv[2]).astype(theano.config.floatX)
    
auto = pickle.Unpickler(f).load()

reduced_data = auto.get_hidden_data(data)

output_data = auto.run_auto(data)
    
np.savetxt("reduced_data.dat",reduced_data)
np.savetxt("output_data.dat",output_data)

print 'MSE: ',np.mean(np.sqrt((data-output_data)**2))
if(reduced_data.shape[1]==2):
    plt.plot(reduced_data[:,0], reduced_data[:,1],'o')
    plt.show()
  



