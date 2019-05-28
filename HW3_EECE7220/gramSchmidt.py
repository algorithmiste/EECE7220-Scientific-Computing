import scipy as sci
import numpy as np
import pandas as pd
import math
# Data Generation:

B0 = 1.25
B1 = -0.2
B2 = 5.3

x_vec = np.array([np.random.randint(-50, 50) for i in range(100) ])
ones = np.ones(100)
Xmat = np.array([ones, x_vec, x_vec**2])
X = pd.DataFrame(np.transpose(Xmat))

print(len(X.columns))

B = np.array([B0, B1, B2])
# y_vec2 = np.array([(B0 + B1 * x_vec[j] + B2 * x_vec[j]**2) for j in range(100)])
y_vec = np.matmul(X, B)


'''Now the generated data points (xi, yi) for i = 1 to 100 lie exactly on a quadratic polynomial.
Add random variations to each of the data points as follows: yi = f(xi) + error(i)
where the random variable error(i) follows a normal distribution with 0 mean and sigma standard
deviation. i.e. error(i) ~ Normal(0, sigma^2 ). '''
sigma = 50
error_vec = np.array([sigma * np.random.randn() for i in range(100)])

random_y_vec = np.array(np.add(y_vec, error_vec)).reshape(100,1)

def gramSchmidt(xSet):
	r, q = [], []

	x0 = xSet[0]
	r00 = math.sqrt(np.inner(x0, x0))
	r += [r00]
	if r00 != 0:
		q0 = divideVector(xSet[0],r00) 
		q += [q0]
	# else:
	# 	break

	rij = []
	for j in range(1, len(xSet)):
		for i in range(0, len(xSet) -2):
			rij += [(np.inner(xSet[j], q[i]))]
			r_q_sum = 0
			result = 0
			for k in range(j):
				result = np.multiply(rij[k], q[k])
				np.add(r_q_sum, result)
			q_hat = np.subtract(xSet[j], r_q_sum)

			rjj = math.sqrt(np.inner(q_hat, q_hat))
			r += [rjj]
			if rjj != 0:
				q += [divideVector(q_hat, rjj)]
			# else: break
	s = []
	for element in q:
		s += [math.sqrt(np.inner(element, element))]
	return q, r

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
print(gramSchmidt(X))