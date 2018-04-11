#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 01:19:18 2018

@author: julien nyambal
"""

import numpy as np

#number to learn
num_to_learn = 50

#parameters of the network
theta = np.array([0.01])
#data
x = np.array([num_to_learn])
#expected output
y = np.array([num_to_learn])
#hypothesis, predicted output
h = np.dot(theta.T,x)
#learning rate
alpha = 0.0001

#J = np.sum(0.5*(y - h)**2)

#error, and theta lists
error, paramaters = [], []

i = 0
m = x.shape[0]

while (True):
    #Recomputation of the expected output
    h = np.dot(theta.T,x)
    #Computation of the loss
    loss = h - y
    #Computation of the gradient. Given the Mean Sqaure Error Loss      
    gradient = (x * loss) /m
    #m is optional as we have only one data point
    theta = theta - alpha * gradient
    #Computation of the error function, again, in this case m is useless
    J = np.sum(0.5 * (1./m) * loss**2)
    print "Error: ", J
    paramaters.append(theta)
    #threshold the error to stop at it when less or equal to 10^-4
    if (J <= 10e-4):
        break
    i = i + 1
#print "Error: ", error
#print "Parameters: ", paramaters
