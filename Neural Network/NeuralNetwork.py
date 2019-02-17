import numpy as np 
import matplotlib.pyplot as plt 
from math import e
from scipy.io import loadmat



def sigmoid(x):
    return 1/(1+e**-x)


class Layer:
    
    def __init__(self,num_of_nodes):
        self.num_of_nodes = num_of_nodes #Number of neurons in a layer 
        self.values = np.zeros(num_of_nodes) #Value of each node in layer initially. Will become np.dot(THETA , PREVIOUS_LAYER VALUES)
        
    def __repr__(self):
        return f"Layer : Number of Nodes = {self.num_of_nodes}"
        


class NN:
    
    
    
    def __init__(self,matlab_file):
        data = loadmat(matlab_file)
        self.X = data['X']
        self.y = data['y'].ravel()
        self.y[self.y==10] = 0
        self.layers = []
        
    def __repr__(self):
        return f"Neural Network With {len(self.layers)} Layers"
        
    def first_layer(self,num_of_nodes):
        layer = Layer(num_of_nodes)
        layer.values = self.X
        self.layers.append(layer)
    
        
    def add_layer(self,num_of_nodes):
        layer = Layer(num_of_nodes)
        self.layers.append(layer)
        
    def prediction(self,theta):
        j = 0
        p = np.zeros(self.X.shape[0])

        for i in range(1,len(self.layers)):
            self.layers[i-1].values = np.insert(self.layers[i-1].values,0,np.ones(self.layers[i-1].values.shape[0]),axis=1)
            z = np.dot(self.layers[i-1].values , theta[j].T)
            self.layers[i].values = sigmoid(z)
            j = j+1
            
            
        return np.argmax(self.layers[-1].values,axis=1)








if __name__ == "__main__":

    Network = NN('ex3data1.mat')
    weights = loadmat('ex3weights.mat')
    theta1 , theta2 = weights['Theta1'],weights['Theta2']
    theta2 = np.roll(theta2, 1,axis=0)
    Network.first_layer(400)
    Network.add_layer(25)
    Network.add_layer(10)
    print(Network.layers)
    p = Network.prediction([theta1,theta2])
    print('Training Set Accuracy: {:.1f}%'.format(np.mean(p == Network.y) * 100)) #Expected is 97.5%
