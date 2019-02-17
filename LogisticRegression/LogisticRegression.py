import numpy as np 
import matplotlib.pyplot as plt 
from math import e 
from scipy import optimize


'''
For Logistic Regression our simplified cost function looks something like this :-
J(Theta) = 1/m multiplied by sum from 1 to m of y*log(HTheta_X) + (1-y)*log(1-HTHeta_X)

For the purpose of this repo we will be using scipy's optimize function 

Due to the nature of the optimize function we will have to explicitly pass the training features and targets to our Cost Function 
however the code for implicitly running the cost function has still been provided which can be utilised by manually providing our own
optimisation function for the parameters.

'''




class LogisticRegression:

    def __init__(self,csv_file , feature = False):
        data = np.loadtxt(csv_file , delimiter = ',')
        X = data[:,:-1]
        if feature == False:
            self.X = np.insert(X,0,values = np.ones(X.shape[0]) , axis = 1)
        else:
            self.X = X
        self.y = data[:,-1]
        
        
    def mapFeature(self,X1, X2, degree=6):
        X1 = np.array(X1)
        X2 = np.array(X2)
        if X1.ndim > 0:
            out = [np.ones(X1.shape[0])]
        else:
            out = [np.ones(1)]

        for i in range(1, degree + 1):
            for j in range(i + 1):
                out.append((X1 ** (i - j)) * (X2 ** j))

        if X1.ndim > 0:
            return np.stack(out, axis=1)
        else:
            return np.array(out)
        

    def _sigmoid(self,z):
        return (1)/(1+e**(-z))

    def computeCost(self,theta):
        m = self.y.size
        grad = np.zeros(self.X.shape[1])
        HTheta_X = self._sigmoid(np.dot(self.X,theta))
        J = -(1/m)* np.sum(np.dot(self.y,np.log(HTheta_X)) + np.dot((1-self.y) , np.log(1-HTheta_X)))
        grad = (1/m)*((np.dot(HTheta_X - self.y , self.X)))
            
        return J,grad

    def computeCostReg(self,theta):
        m = self.y.size 
        grad = np.zeros(self.X.shape[1])
        HTheta_X =self._sigmoid(np.dot(self.X,theta))
        J = -(1/m)*(np.sum(  np.dot(self.y,np.log(HTheta_X)) + np.dot(1-self.y,np.log(1-HTheta_X)) ) ) + (lambda_/(2*m))*(np.sum(theta**2))
        grad = (1/m)*((np.dot(HTheta_X - self.y , self.X))) + (lambda_/m)*theta    
        return J , grad 
        
    
    """
    Explicitly passing the feature and target set for the following functions.

    """


    def CostFunction(self,theta,X,y):
        m = y.size
        grad = np.zeros(X.shape[1])
        HTheta_X = self._sigmoid(np.dot(X,theta))
        J = -(1/m)* np.sum(np.dot(y,np.log(HTheta_X)) + np.dot((1-y) , np.log(1-HTheta_X)))
        grad = (1/m)*((np.dot(HTheta_X - y , X))) 
        return J, grad
    
    
    def CostFunctionReg(self,theta,X,y,lambda_):
        m = y.size 
        grad = np.zeros(X.shape[1])
        HTheta_X = self._sigmoid(np.dot(X,theta))
        J = -(1/m)*(np.sum(  np.dot(y,np.log(HTheta_X)) + np.dot(1-y,np.log(1-HTheta_X)) ) ) + (lambda_/(2*m))*(np.sum(theta[1:]**2))      
        grad = (1/m)*((np.dot(HTheta_X - y , X))) + (lambda_/m)*theta    
        return J , grad 
    
        
    def optimise(self , theta,reg=False, lambda_ = None):
        
        
        if reg == False:
            res = optimize.minimize(self.CostFunction,theta,(self.X,self.y),jac=True,method='TNC',options = {'maxiter':400})
        else:
            res = optimize.minimize(self.CostFunctionReg ,theta,(self.X,self.y,lambda_),jac=True,method='TNC',options = {'maxiter':100})
        cost = res.fun
        theta = res.x
        return cost,theta
    

    """Gives the training accuracy of our model"""

    def train(self,theta , threshold):
        HTheta_X = self._sigmoid(np.dot(self.X,theta))
        p = (HTheta_X >= threshold).astype(int)
        return p 

    """Simply pass features to be predicted and will return probability of a truth value"""
    
    def predict(self,values,theta):
        values = np.array([1,*values])
        return self._sigmoid(np.dot(values,theta))
        





if __name__ == '__main__':

    LG = LogisticRegression('ex2data1.txt')
    reduced_cost , reduced_theta = LG.optimise(theta = np.zeros(LG.X.shape[1]))
    prediction = LG.predict([45,85] , reduced_theta)
    print(prediction.round(3)) #Expected 0.775 +/- 0.002


    LG2 = LogisticRegression('ex2data2.txt' , feature = True)
    LG2.X = LG2.mapFeature(LG2.X[:,0] , LG2.X[:,1])
    
    initial_theta = np.zeros(LG2.X.shape[1])
    cost , grad = LG2.CostFunctionReg(initial_theta , LG2.X,LG2.y,1)
    print(cost.round(4)) #Expected 0.693
    print(*grad[:5].round(4)) #Expected gradients (approx) - first five values only: [0.0085, 0.0188, 0.0001, 0.0503, 0.0115]
    

    test_theta = np.ones(LG2.X.shape[1])
    cost, grad = LG2.CostFunctionReg(test_theta, LG2.X, LG2.y, 10)
    print(cost)#Expected 3.16
    print(*grad[:5].round(4)) #Expected gradients (approx) - first five values only: [0.3460, 0.1614, 0.1948, 0.2269, 0.0922]

        