import numpy as np
import matplotlib.pyplot as plt 
from utils import load_matlab,sigmoid
from scipy.io import loadmat

alpha = 0.001
weights = loadmat('ex3weights.mat')
theta1,theta2 = weights['Theta1'],weights['Theta2']
theta2 = np.roll(theta2 , 1, axis=0)




class Layer:

    def __init__(self,num_nodes,activation,layer_type):
        
        self.num_nodes = num_nodes 
        self.activation = activation 
        self.x = None 
        self.z = None 
        self.a = None 
        self.weights = None
        self.layer_type = layer_type

        self.dz = None 
        self.dw = None 



    
     

    def __repr__(self):
        return f"NUMBER OF NODES : {self.num_nodes} , ACTIVATION : {self.activation} , INPUT_LAYER = {self.input_layer}"

    def init_x(self,input_values,insert=False):
        if(insert==True):
            input_values = np.insert(input_values,0,1)
        self.x = input_values

    def init_weights(self,num_nodes_prev,weights):
        if(type(weights) == "<class 'int'>"):
            self.weights = np.random.randn(num_nodes_prev,self.num_nodes)

        else:
            self.weights = weights.T


    def compute_z(self):
        self.z = np.dot(self.x , self.weights)
    
    def compute_a(self):
        self.a = sigmoid(self.z)




X,y = load_matlab('ex3data1.mat')



k=0
for i in range(X.shape[0]):
    L1 = Layer(400,"sigmoid","INPUT")
    L1.init_x(X[i,:])

    L2 = Layer(25 , "sigmoid", "HIDDEN")
    L2.init_x(L1.x,insert=True)
    L2.init_weights(401,theta1)

    L2.compute_z()
    L2.compute_a()
    L3 = Layer(10, "sigmoid", "OUTPUT")
    L3.init_x(L2.a,insert=True)
    L3.init_weights(26,theta2)
    L3.compute_z()
    L3.compute_a()
    p =  np.argmax(L3.a)
    x = np.mean(p==y[i])
    k = k + x
   

print(f'Training Set Accuracy: {(k/X.shape[0])*100}')



"""alpha = 0.1
x=0
J = []

for epoch in range(1000):
    print(f"EPOCH IS {epoch+1}")
    
    for i in range(X.shape[0]):
        L1 = Layer(400,"sigmoid","INPUT")
        L1.init_x(X[i,:])
    
        L2 = Layer(25 , "sigmoid", "HIDDEN")
        L2.init_x(L1.x,insert=True)
        L2.init_weights(401,0)
    
        L2.compute_z()
        L2.compute_a()
        L3 = Layer(10, "sigmoid", "OUTPUT")
        L3.init_x(L2.a,insert=True)
        L3.init_weights(26,0)
    
        L3.compute_z()
        L3.compute_a()
        
        p =  np.argmax(L3.a)
        actual = y[i]
        x = x + np.mean(p == y[i])
        
        
    print("TRAINING SET ACCURACY {:.1f}%".format(np.mean(p == y[i])))
    
    
#if(i%100 == 0):
         #   J.append( y[i] * np.log(L3.a) + (1-y[i]) * (1-np.log(L3.a)))
    
        L3.dz = L3.a - y[i]
        L3.db = np.sum(L3.dz)
    
        L3.dz = L3.dz.reshape(L3.dz.shape[0],-1)
        L2.a  = L2.a.reshape(L2.a.shape[0],-1)
        L3.dw = np.dot(L3.dz, L2.a.T)
    
        g_prime = (sigmoid(L2.z) * (1-sigmoid(L2.z))).sum()
        L2.dz = np.dot(L3.weights, L3.dz) * g_prime
        L1.x = L1.x.reshape(L1.x.shape[0],-1)     
        L2.dw = np.dot(L2.dz,L1.x.T)
    
        L2.db = np.sum(L3.dz)
    
    
        L3.dw = np.insert(L3.dw,1,np.array(L3.db),axis = 1)
        L2.dw = np.insert(L2.dw,1,np.array(L2.db),axis = 1)
    
        L3.weights = L3.weights - alpha * L3.dw.T
        L2.weights = L2.weights - alpha * L2.dw[:-1,:].T
        if(i==X.shape[0]-1):
            print(f"ACCURACY IS {(x/X.shape[0])*100}")"""


