########################################Import Libraries##############################################################
# used for manipulating directory paths
import os

# # Import Graders
# import sys
# sys.path.append(sys.path.append(r'C:\Users\Orphanides\Documents\Statistics Learning\Machine Learning Coursea (Andrew Ng)\Assignment Grader Modules'))
# sys.path
# import utils
# grader = utils.Grader()

# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D  # needed to plot 3-D surfaces

########################################Multiple Linear Regression##############################################################

########Set Working Directory and load data############
os.chdir("C:/Users/Orphanides/Documents/Statistics Learning/Machine Learning Coursea (Andrew Ng)/Assignment 1/Exercise1/Data")
os.listdir()
data = np.loadtxt('ex1data2.txt', delimiter=',') #import csv file

######################Slicing Data: Defining Variables y and X#####################
X = data[:, :2]  #all columns up to and EXCLUDING 2. #a[:end] - items from the beginning through end-1
#https://github.com/mstampfer/Coursera-Stanford-ML-Python/blob/master/Coursera%20Stanford%20ML%20Python%20wiki.ipynb

#previous comment
#all rows, and all columns up to and including 2 [remember, python indexes starting from 0 so this is column 3 of the dataframe technically]
#ie. [0,1,2] #wtf why is it not [:,:1]

y = data[:, 2] #all rows of column 2, which is technically just column 2
m = y.size


#n is number of features
#so m x n matrix is a matrix with the number of rows equal to observations, and each column corresponding to the feature.
# eg. x_0 x_1 x_2
#i=1  1    4   2       (2x3); 2 observations and 3 features
#i=2  1    8   2

#Printing out some rows
print('{:>8s}{:>8s}{:>10s}'.format('X[:,0]', 'X[:, 1]', 'y'))
print('-'*26)
for i in range(10):
    print('{:8.0f}{:8.0f}{:10.0f}'.format(X[i, 0], X[i, 1], y[i]))


#############################Feature Normalisation##################################


####Explanation###
#If mu[i,:] only works if mu is a TWO DIMENSIONAL array.
#If mu is a ONE DIMENSIONAL array, mu[i] only works.

def  featureNormalize(X):
   
    X_norm = X.copy() #create a copy of the existing dataset
    mu = np.zeros(X.shape[1]) #creates a vector of zeros corresponding to the number of columns or features - ie. a mean for each feature
    sigma = np.zeros(X.shape[1]) #creates a vector of zeros corresponding to the number of columns or features - ie. a variance for each feature
    #two columns
    
    for i in range(2): #problem with using the range function, refer to notes at very bottom of script
        mu[i] = np.mean(X_norm[:,i]) 
        sigma[i] = np.std(X_norm[:,i])
    
    for i in range (2):
        X_norm[:,i] = (X_norm[:,i] - mu[i])/sigma[i]
    
    return X_norm, mu, sigma

X_norm, mu, sigma = featureNormalize(X) #execute the function and get the outputs X_norm, mu and sigma back

#Check this feature normalisation code and a way to make it more efficient. Doesn't seem to be working right. Submit to
#grader?

print('Computed mean:', mu)
print('Computed standard deviation:', sigma)

# grader[4] = featureNormalize
# grader.grade()

X = np.concatenate([np.ones((m, 1)), X_norm], axis=1) #add the intercept feature x_0 values in [x_0 for all i = 1]
#np.ones(m,1) creates a 2 dimensional (mx1) vector of ones.
#np.contacenate(np,ones, X_norm) saves the purpose of 'merging' the two vectors/matrices into one big matrix/dataset
#axis = 1 tells numpy to merge by COLUMN, not by ROW(which would be axis=0) #see here: https://www.google.com/search?q=numpy+axis&sxsrf=APq-WBu5vjA34355a449pyoE2zUquuhsvw:1643772414132&tbm=isch&source=iu&ictx=1&vet=1&sa=X&ved=2ahUKEwjdtte2ieD1AhWyUGwGHQSEA3QQ_h16BAgLEAc&biw=1115&bih=611&dpr=2.03#imgrc=gPcqaRZ_oIIdYM

##########################Define Cost Function###################################

def computeCostMulti(X, y, theta):
    J = 0 #why is this here in the instructions again? #to initialise the variable J presumably, it will be replaced by the following line
    J = (1/2*m) * sum( np.square( (X@theta - y) ) ) #Cost Function is really just average sum of squared residuals (with 2m in denominator for some reason instead of m)
    return J

# #X@theta. X is a (mxn) array. So it would treat theta as a (nx1) vector so the inner shape dimensions match (ie. n = n) are compatible. So it spits out a
# #(mx1) 1d array.
# #test this with @
# theta = np.full( (3,), 0)
# a= X@theta #yes it works like exactly that!
# b= X@theta - y
# # c = np.reshape(b, (m,1))

# sum(np.square(X@theta - y))
# c = np.square(X@theta - y)
# #try to replace the np.dots here with @ instead. https://mkang32.github.io/python/2020/08/30/numpy-matmul.html #np.dot VS @ operator

# # theta = np.full( (3,), 0)
# # J = computeCostMulti(X,y,theta)


############################Gradient Descent###########################################


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    
    theta = theta.copy() # make a copy of theta, which will be updated by gradient descent
    
    J_history = [] #calls ComputeMultiCost and saves the cost on every iteration to a python list
    
    for i in range(num_iters):
        temp0 = theta[0] - alpha/m * np.sum(X@theta - y)
        temp1 = theta[1] - alpha/m * np.sum( (X@theta - y) * X[:,1])
        temp2 = theta[2] - alpha/m * np.sum( (X@theta - y) * X[:,2] )
                                              
        theta[0] = temp0
        theta[1] = temp1
        theta[2] = temp2      
        J_history.append(computeCostMulti(X, y, theta)) # save the cost J in every iteration
        
        #no need to put an i here, loop will run automatically
    
    return theta, J_history


##############Check Gradient Descent Results##################


# some gradient descent settings
alpha = 0.10
num_iters = 400

# initialize fitting parameters
theta = np.zeros(3) 
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)

# Plot the convergence graph
pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost J')

# Display the gradient descent's result
print('theta computed from gradient descent: {:s}'.format(str(theta)))



########################################Estimate the price of a 1650 sq-ft, 3 br house###########################################################

########################################Gradient Descent Method###################################
# ======================= YOUR CODE HERE ===========================
# Recall that the first column of X is all-ones. 
# Thus, it does not need to be normalized.

price = 0   # You should change this

# ===================================================================

print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): ${:.0f}'.format(price))














######################################Normal Equation Solution########################################
data = np.loadtxt('ex1data2.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = y.size
X = np.concatenate([np.ones((m, 1)), X], axis=1)

def normalEqn(X, y):
    """
    Computes the closed-form solution to linear regression using the normal equations.
    
    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).
    
    y : array_like
        The value at each data point. A vector of shape (m, ).
    
    Returns
    -------
    theta : array_like
        Estimated linear regression parameters. A vector of shape (n+1, ).
    
    Instructions
    ------------
    Complete the code to compute the closed form solution to linear
    regression and put the result in theta.
    
    Hint
    ----
    Look up the function `np.linalg.pinv` for computing matrix inverse.
    """
    theta = np.zeros(X.shape[1])
    
    # ===================== YOUR CODE HERE ============================
    theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)
    
    # =================================================================
    return theta

theta = normalEqn(X, y);

# Display normal equation's result
print('Theta computed from the normal equations: {:s}'.format(str(theta)));

# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================



from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X, y)
print('intercept:', model.intercept_)
print('slope:', model.coef_)