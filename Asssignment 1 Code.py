########################################Import Libraries##############################################################
# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D  # needed to plot 3-D surfaces

# Import Graders


########################################Multiple Linear Regression##############################################################

########Set Working Directory#######
os.chdir("C:/Users/Orphanides/Documents/Statistics Learning/Machine Learning Coursea (Andrew Ng)/Assignment 1/Exercise1/Data")
os.listdir()
data = np.loadtxt('ex1data2.txt', delimiter=',') #import csv file

##########Slicing Data: Defining Variables y and X########
X = data[:, :2]  #all rows up to and EXCLUDING 2. #a[:end] - items from the beginning through end-1
#https://github.com/mstampfer/Coursera-Stanford-ML-Python/blob/master/Coursera%20Stanford%20ML%20Python%20wiki.ipynb

#previous comment
#all rows, and all columns up to and including 2 [remember, python indexes starting from 0 so this is column 3 of the dataframe technically]
#ie. [0,1,2] #wtf why is it not [:,:1]

y = data[:, 2] #all rows of column 2, which is technically just column 2
m = y.size
y = np.reshape( y, (m,1))
#make sure to make this into a 2d array


#n is number of features
#so m x n matrix is a matrix with the number of rows equal to observations, and each column corresponding to the feature.
# eg. x_1 x_2 x_3
#i=1  1    1   2       (2x3); 2 observations and 3 features
#i=2  0    1   2

# #Printing out some rows
# print('{:>8s}{:>8s}{:>10s}'.format('X[:,0]', 'X[:, 1]', 'y'))
# print('-'*26)
# for i in range(10):
#     print('{:8.0f}{:8.0f}{:10.0f}'.format(X[i, 0], X[i, 1], y[i]))

#doesn't work with the two dimensional array we just created unfortunately.

##########Feature Normalisation##########
def  featureNormalize(X):
   
    X_norm = X.copy() #create a copy of the existing dataset
    mu = np.zeros(X.shape[1]) #creates a vector of zeros corresponding to the number of columns or features - ie. a mean for each feature
    mu = np.reshape(mu, (1,2)) #to reshape the 1 dimensional array into two dimensions #https://www.kite.com/python/answers/how-to-reshape-a-1d-numpy-array-to-a-2d-numpy-array-in-python
    #two columns
    sigma = np.zeros(X.shape[1]) #creates a vector of zeros corresponding to the number of columns or features - ie. a variance for each feature
    #two columns
    sigma = np.reshape(sigma, (1,2))
    
    for i in range(2): #problem with using the range function, refer to notes at very bottom of script
        mu[:,i] = np.mean(X_norm[:,i]) 
        sigma[:,i] = np.std(X_norm[:,i])
    
    for i in range (2):
        X_norm[:,i] = (X_norm[:,i] - mu[:,i])/sigma[:,i]
    
    return X_norm, mu, sigma

X_norm, mu, sigma = featureNormalize(X)

print('Computed mean:', mu)
print('Computed standard deviation:', sigma)

X = np.concatenate([np.ones((m, 1)), X_norm], axis=1) #add the intercept feature x_0 values in [x_0 for all i = 1]
#review this later

##########Define Cost Function##########

# #test out this formula first
# theta = np.array([[0,0,0]]).T #make sure to make this a two dimensional array
# theta.shape

# J = (1/2*m) *( np.dot( (np.dot(X,theta) - y).T , (np.dot(X,theta) - y)) )

#   #x_0, x_1, x_2 all exist in the model, so n=2 (theta_1 and theta_2), then theta_0 so n+1 = 3
 
def computeCostMulti(X, y, theta):
    J = 0 #why is this here in the instructions again?
    J = (1/2*m) *( np.dot( (np.dot(X,theta) - y).T , (np.dot(X,theta) - y)) ) #translating the matrix formula for J into code
    return J

#try to replace the np.dots here with @ instead. https://mkang32.github.io/python/2020/08/30/numpy-matmul.html #np.dot VS @ operator

theta = np.array([ [0,0,0] ]).T
J = computeCostMulti(X,y,theta)

##########Gradient Descent##########

theta = np.array([[0,0,0]]).T #default theta is a column vector
# theta = np.array([0,0,0]) #default theta is a column vector #one dimensional version for reference,
# #no transpose here as it won't work on that.

# a = np.reshape(X[:,1], (m,1) )
# b = X[:,1]

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    
    # make a copy of theta, which will be updated by gradient descent
    theta = theta.copy()
    
    J_history = [] #calls ComputeMultiCost and saves the cost on every iteration to a python list
    
    for i in range(num_iters):
        temp0 = theta[0,:] - alpha/m * np.sum(X@theta - y)
        temp1 = theta[1,:] - alpha/m * np.sum( (X@theta - y) * np.reshape(X[:,1], (m,1)) )
        temp2 = theta[2,:] - alpha/m * np.sum(( (X@theta - y) * np.reshape(X[:,2], (m,1)) ) )
                                              
        theta[0,:] = temp0
        theta[1,:] = temp1
        theta[2,:] = temp2      
        J_history.append(computeCostMulti(X, y, theta)) # save the cost J in every iteration
        
        #just have to figure out how to integrate the i in here. Or do we even have to put an i in here
    
    return theta, J_history


######Check Gradient Descent Results

# initialize fitting parameters
theta = np.zeros(2) #shouldn't be these, should be 2D for it to work in my code

# some gradient descent settings
alpha = 0.1
num_iters = 400
theta = np.array([[0,0,0]]).T
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)
pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost J')



#Try to change the code so it can work with 1D arrays. The print command might not work here in this case. 





################################General Python Function Notes#############################################
##Python Range Function
range(0,2) #does NOT get numbers from 0 to 2 but instead from 0 UP TO AND EXCLUDING 2

for i in range(0,2):
    print(i) 
#as you can see, only prints out 0 and 1

#to get numbers up to and INCLUDING 2, has to be range (0,3) #https://pynative.com/python-range-function/


#gradient descent

#########Difference between numpy.dot and a.dot(b)########
#they're the same thing: https://stackoverflow.com/questions/42517281/difference-between-numpy-dot-and-a-dotb

#########Difference between np.dot and @############
 # Use np.dot for dot product. For matrix multiplication, use @ for Python 3.5 or above https://mkang32.github.io/python/2020/08/30/numpy-matmul.html

###Checking Element Wise Multiplication###
# z = np.full((m,1), 2)
# a = np.reshape(X[:,1], (m,1))
# c = a*z


