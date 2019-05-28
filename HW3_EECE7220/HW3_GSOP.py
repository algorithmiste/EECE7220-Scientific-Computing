s# Question 1: Write a program to implement the following Gram-Schmidt Orthogonalization Procedure

import scipy as sci
import numpy as np
import pandas as pd
import math

def gramSchmidt(xSet):
	r, q = [], []

	x0 = xSet[0]
	r00 = math.sqrt(np.inner(x0, x0))
	r += [r00]
	# print(r[0])
	if r00 != 0:
		q0 = divideVector(xSet[0],r00)
		q += [q0]
	else:
		return None #break?

	rij = []
	for j in range(1, len(xSet)):
		for i in range(0, len(xSet) -2):
			rij += [(np.inner(xSet[j], q[i]))]
			# print(rij[i])
			r_q_sum = 0
			result = 0
			for k in range(j):
				result = np.multiply(rij[k], q[k])
				np.add(r_q_sum, result)
				# r_q_sum += result
			q_hat = np.subtract(xSet[j], r_q_sum)

			rjj = math.sqrt(np.inner(q_hat, q_hat))
			r += [rjj]
			if rjj != 0:
				q += [divideVector(q_hat, rjj)]
			else: break
	s = []
	for element in q:
		s += [math.sqrt(np.inner(element, element))]
	return q

def divideVectorOfVectors(vector, value):
	vectorToReturn = []
	for subVectors in vector:
		vector_i = []
		for element in subVectors:
			vector_i += [element/value]
		vectorToReturn += [vector_i]
	return vectorToReturn

def divideVector(vector, value):
	vectorToReturn = []
	for element in vector:
		vectorToReturn += [element/value]
	return vectorToReturn

xSet = [[2, 7, 3], [11, 19, 1], [15, 0, 0]]
print(gramSchmidt(xSet))

def modGramSchmidt(xSet):
	r, q = [], []

	x0 = xSet[0]
	r00 = math.sqrt(np.inner(x0, x0))
	r += [r00]
	# print(r[0])
	if r00 != 0:
		q0 = divideVector(xSet[0],r00)
		q += [q0]
	else:
		return None #break?


	rij = []
	for j in range(1, len(xSet)):
		q_hat = xSet[j]

		for i in range(0, len(xSet) -2):

			# rij += [(np.inner(xSet[j], q[i]))]
			# print(rij[i])

			r_q_sum = 0
			rij += [np.inner(q_hat,q[i])]
			# print(rij)
			result = 0
			for k in range(j):
				result = np.multiply(rij[k], q[k])
				np.add(r_q_sum, result)  # r_q_sum += result

			q_hat = np.subtract(q_hat, r_q_sum)

			rjj = math.sqrt(np.inner(q_hat, q_hat))
			r += [rjj]
			if rjj != 0:
				q += [divideVector(q_hat, rjj)]
			else: break
	s = []
	for element in q:
		s += [math.sqrt(np.inner(element, element))]
	return q

xSet = [[2, 7, 3], [11, 19, 1], [15, 0, 0]]
print(modGramSchmidt(xSet))
