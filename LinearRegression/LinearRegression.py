import numpy as np 
import matplotlib.pyplot as plt 




class LinearRegression:
    
    def __init__(self,csv_file):
        data = np.loadtxt(csv_file , delimiter = ',')
        X = data[: , :-1]
        self.y = data[: , -1]
        self.X = np.insert(X,0,values = np.ones(X.shape[0]) , axis =1) 
        self.mu = np.mean(self.X[:,1:] , axis = 0)
        self.sigma = np.std(self.X[:,1:] , axis = 0)
        
    
    def head(self,n):
        print("Feature Data Is")
        print(self.X[:n,:])
        print("Target Data Is")
        print(np.array(self.y[:n]))
        
        
    def normalise(self):
        X_norm = (self.X[:,1:]-self.mu)/self.sigma
        X_norm = np.insert(X_norm,0,values = np.ones(self.X.shape[0]) , axis =1) 
        return X_norm
        
        
    def computeCost(self,theta):
        """m represents number of training examples
           X is feature set with n features and is matrix of shape m*(n+1)
           theta is parameter set of shape (n+1)*1
           y is target data set of shape m*1
           X*theta represents our predicted value 
           (X*theta-y)^2 is the squared error value
           J is sum of all the squared errors over 2*m
        """
        m = self.y.shape[0]
        HTheta_X = np.dot(self.X,theta)
        Error = HTheta_X-self.y
        J = (np.sum(Error**2))/(2*m)
        return J


    
    def gradientDescent(self,theta,alpha,iterations):
        theta = theta.copy()
        m = self.y.shape[0] 
        J_values = []

        for num in range(iterations):
            for j in range(theta.size):
                Error = np.dot(self.X,theta)-self.y
                Derivative = np.dot(Error, self.X[:,j])
                theta[j] = theta[j] - (np.sum(Derivative))*(alpha/m)
                
            J_values.append(self.computeCost(theta))

        return theta,J_values
    
    def predict_value(self,values,theta , norm=False):
        if(norm == True):
            values = (values-self.mu)/self.sigma
        values = np.insert(values , 0 , np.ones(1) , axis = 0)
        predict = np.dot(values,theta)
        return predict


if __name__ == '__main__':


    LR = LinearRegression('ex1data2.txt')
    LR.X = LR.normalise()
    theta, J_history = LR.gradientDescent(np.zeros(LR.X.shape[1]), 0.15, 1000)
    print(LR.predict_value([1650,3],theta , norm = True)) #Prediction for 1650 sq. feet and 3 bedrooms

    '''For ex1data1 file , no need for normalisation'''
    LR2 = LinearRegression('ex1data1.txt')
    theta , J_history = LR2.gradientDescent(np.zeros(2) , 0.01 ,1500)
    print(LR2.predict_value([7] , theta))










