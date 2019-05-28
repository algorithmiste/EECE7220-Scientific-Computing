	
''' Generating an nth order polynomial
	Casey Carr
	EECE 7220
	HW1
	08/31/2018 '''

import pandas as pd
import numpy as np
from numpy.linalg import *
import scipy as sci
import matplotlib.pyplot as plt
from matplotlib import *
interactive(True)


# Function to generate an nth order polynomial

def genNthOrderPolynomial(domainOfXVec, rootOfPxVec, xStepSize):
	
	orderN = len(rootOfPxVec)  # Find the number of roots or the order of the polynomial p(x)

	xVec = np.arange(domainOfXVec[0], domainOfXVec[1], xStepSize) # Identify all x for evaluating p(x)

	pVec = np.ones_like(xVec) #(np.size(xVec))

	for i in range(orderN):
		pVec = pVec * (xVec - rootOfPxVec[i]) #np.multiply(pVec,(np.subtract(xVec,rootOfPxVec[i])))
	return pVec

#print(genNthOrderPolynomial((-10,10), [1], 1))

def plotNthOrderPolynomial(domainOfXVec, rootOfPxVec, xStepSize):
	xVec = np.arange(domainOfXVec[0], domainOfXVec[1], xStepSize)
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	plt.scatter(xVec, genNthOrderPolynomial(domainOfXVec, rootOfPxVec, xStepSize), facecolors='none', edgecolors='g')
	props = {
	'title': 'Generating an Nth Order Polynomial',
	'xlabel': 'x',
	'ylabel': 'P(x)'
	}
	ax.set(**props)
	plt.savefig("nthOrderPolynomial.png")
	
	return

plotNthOrderPolynomial((-5,5),[-1, 0, 2],0.1)


