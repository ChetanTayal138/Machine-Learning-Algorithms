import numpy as np 
import matplotlib.pyplot as plt

"""Some initial data for testing. Uncomment this docstring and uncomment print
statements to check working of the optmizers

X = 2* np.random.rand(100,1) #draws from a uniform distribution
y = 4 + 3*X + np.random.randn(100,1) #draws from a normal distribution
ones = np.ones(X.shape[0])
X_b = np.insert(X,0,ones,axis=1) """


#Normal Equation 
def normal_equation(X,y,show_data = False,bias = False):
	
	if(show_data == True):
		plt.scatter(X,y)
		plt.show()

	ones = np.ones(X.shape[0])
	if(bias == True):
		X_b = np.insert(X,0,ones,axis=1)
	else:
		X_b = X
	XX = X_b.T.dot(X_b)
	XY = X_b.T.dot(y)
	Theta = np.linalg.inv(XX).dot(XY)
	return Theta

#print(normal_equation(X,y,bias=True))


#Batch Gradient Descent 
def BGD(X,y,eta,iterations,bias=False):
	m = X.shape[0]
	if(bias == True):
		X_b = np.insert(X,0,np.ones(m),axis=1)
	else:
		X_b = X

	theta = np.random.randn(X_b.shape[1] , 1)

	for iteration in range(iterations):
		err = X_b.dot(theta)-y
		derivatives = (2/m) * X_b.T.dot(err)
		theta = theta - eta*derivatives

	return theta

#print(BGD(X_b,y,0.1,1000,100))


#Stochastic Gradient Descent 
def SGD(X,y,epochs,t0,t1,bias = False):

	m = X.shape[0]

	if(bias == True):
		X_b = np.insert(X,0,np.ones(X.shape[0]),axis=1)
	else:
		X_b = X

	
	theta = np.random.randn(X_b.shape[1],1)
	for epoch in range(epochs):
		for i in range(m):
			random_index = np.random.randint(m)
			xi = X_b[random_index:random_index+1]
			yi = y[random_index:random_index+1]
			err = xi.dot(theta)-yi
			gradients = 2* xi.T.dot(err)
			eta = t0/((epoch*m+i)+t1)
			theta = theta - eta*gradients


	return theta 

#print(SGD(X,y,50,5,50,bias=True))
