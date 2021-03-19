import numpy as np
import matplotlib.pyplot as plt
#import scipy
#from PIL import Image
#from scipy import ndimage
#from lr_utils import load_dataset

def initialize_with_zeros(dim):
	"""
	This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
	
	@type dim:		number
	@param dim:		size of the w vector we want (or number of parameters in this case)
					should be equal to number of x values
					
	@rtype:			tuple
	@return:		w -- initialized vector of shape (dim, 1)
					b -- initialized scalar (corresponds to the bias)
	"""

	w = np.zeros([dim, 1])
	b = 0

	assert(w.shape == (dim, 1))
	assert(isinstance(b, float) or isinstance(b, int))
	
	return w, b
	
def sigmoid(z):
	"""
	Compute the sigmoid of z

	@type z:		number or numpy array
	@param z:		[transpose of W] * X + b
	@rtype:			number or numpy array
	@return:		sigmoid function of z as defined
	"""
	#z = np.where(z > 709, 709, z)
	#z = np.where(z < -709, -709, z)
	return 1 / (1 + np.exp(-z))
	
def propagate(w, b, X, Y):
	"""
	Implement the cost function and its gradient

	@type w:		numpy array
	@param w:		weights, a numpy array of size (nx, 1) img example: (num_px * num_px * 3, 1)
	@type b:		number
	@param b:		bias, a scalar
	@type X:		numpy array
	@param X:		data of size (nx, 1)
	@type Y:		numpy array
	@param Y:		true "label" vector (containing 0 if false, 1 if true) of size (1, number of examples)

	@rtype:			tuple
	@return:		cost -- negative log-likelihood cost for logistic regression
					dw -- gradient of the loss with respect to w, thus same shape as w
					db -- gradient of the loss with respect to b, thus same shape as b
	"""
	
	#X should be shape (nx, m) where nx is number of x values while m is number of samples
	m = X.shape[1]
	
	# FORWARD PROPAGATION (FROM X TO COST)
	# compute activation
	A = sigmoid(np.dot(w.T, X) + b)                                    
							   
	# compute cost
	cost = (-1/m)*np.sum(Y* np.log(A) + (1-Y)*np.log(1-A))
	
	# BACKWARD PROPAGATION (TO FIND GRAD)
	dw = (1/m)*np.dot(X, (A-Y).T)
	db = (1/m)*np.sum(A - Y)

	assert(dw.shape == w.shape)
	assert(db.dtype == float)
	cost = np.squeeze(cost)
	assert(cost.shape == ())
	
	grads = {"dw": dw,
			 "db": db}
	
	return grads, cost
	
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False, meetTol=True, tol=.00001):
	"""
	This function optimizes w and b by running a gradient descent algorithm
	
	@type w:				numpy array
	@param w:				weights, a numpy array of size (nx, 1) img example: (num_px * num_px * 3, 1)
	@type b:				number
	@param b:				bias, a scalar
	@type X:				numpy array
	@param X:				data of size (nx, 1)
	@type Y:				numpy array
	@param Y:				true "label" vector (containing 0 if false, 1 if true) of size (1, number of examples)
	@type num_iterations:	number
	@param num_iterations:	number of iterations of the optimization loop
	@type learning_rate:	number
	@param learning_rate:	learning rate of the gradient descent update rule
	@type print_cost:		bool
	@param print_cost:		True to print the loss every 100 steps
	
	@rtype:					tuple
	@return:				params -- dictionary containing the weights w and bias b
							grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
							costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
	"""
	
	costs = []
	
	for i in range(num_iterations):
		
		
		# Cost and gradient calculation
		grads, cost = propagate(w, b, X, Y)
		
		# Retrieve derivatives from grads
		dw = grads["dw"]
		db = grads["db"]
		
		# update rule
		w = w - learning_rate*dw
		b = b - learning_rate*db
		
		# Record the costs
		costs.append(cost)
		
		# Print the cost every 10 training iterations
		if print_cost and i % 100 == 0:
			print ("Cost after iteration %i: %f" %(i, cost))
			
		if meetTol and len(costs) > 1:
			if abs(cost - costs[-2]) < tol:
				print("Cost tolerance met on iteration: ", str(i))
				break
	
	params = {"w": w,
			  "b": b}
	
	grads = {"dw": dw,
			 "db": db}
	
	return params, grads, costs
	
def predict(w, b, X):
	'''
	Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
	
	@type w:				numpy array
	@param w:				weights, a numpy array of size (nx, 1) img example: (num_px * num_px * 3, 1)
	@type b:				number
	@param b:				bias, a scalar
	@type X:				numpy array
	@param X:				data of size (nx, 1)
	
	@rtype:					numpy array
	@return:				Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
	'''
	
	#X should be shape (nx, m) where nx is number of x values while m is number of samples
	m = X.shape[1]
	Y_prediction = np.zeros((1,m))
	w = w.reshape(X.shape[0], 1)
	
	# Compute vector "A" predicting the probabilities
	A = sigmoid(np.dot(w.T, X) + b)
	
	for i in range(A.shape[1]):
		
		# Convert probabilities A[0,i] to actual predictions p[0,i]
		if A[0, i] <= .5:
			Y_prediction[0, i] = 0
		else:
			Y_prediction[0, i] = 1
		Y_prediction[0, i] = A[0, i]
	
	assert(Y_prediction.shape == (1, m))
	
	return Y_prediction
	
def model(X_train, Y_train, X_test, Y_test, num_iterations = 200, learning_rate = .5, print_cost = False):
	"""
	Builds the logistic regression model by calling the function you've implemented previously
	
	Arguments:
	X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
	Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
	X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
	Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
	num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
	learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
	print_cost -- Set to true to print the cost every 100 iterations
	
	Returns:
	d -- dictionary containing information about the model.
	"""
	
	# initialize parameters with zeros 
	w, b = initialize_with_zeros(X_train.shape[0])

	# Gradient descent 
	parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
	
	# Retrieve parameters w and b from dictionary "parameters"
	w = parameters["w"]
	b = parameters["b"]
	
	# Predict test/train set examples
	Y_prediction_test = predict(w, b, X_test)
	Y_prediction_train = predict(w, b, X_train)

	# Print train/test Errors
	print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
	print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

	
	d = {"costs": costs,
		 "Y_prediction_test": Y_prediction_test, 
		 "Y_prediction_train" : Y_prediction_train, 
		 "w" : w, 
		 "b" : b,
		 "learning_rate" : learning_rate,
		 "num_iterations": num_iterations}
	
	return d