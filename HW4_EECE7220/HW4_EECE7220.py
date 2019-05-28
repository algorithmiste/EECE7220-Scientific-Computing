import numpy as np
import pandas as pd
import scipy as sci
import scipy.optimize as opt
import math


# Data Generation:

B0 = 1.25
B1 = -0.2
B2 = 5.3

x_vec = np.array([np.random.randint(-50, 50) for i in range(100) ])
ones = np.ones(100)
Xmat = np.array([ones, x_vec, x_vec**2])

X = np.transpose(Xmat)

B = np.array([B0, B1, B2])
y_vec = np.matmul(X, B)

'''Now the generated data points (xi, yi) for i = 1 to 100 lie exactly on a quadratic polynomial.
Add random variations to each of the data points as follows: yi = f(xi) + error(i)
where the random variable error(i) follows a normal distribution with 0 mean and sigma standard
deviation. i.e. error(i) ~ Normal(0, sigma^2 ). '''
sigma = 50
error_vec = np.array([sigma * np.random.randn() for i in range(100)])

random_y_vec = np.array(np.add(y_vec, error_vec)).reshape(100,1)

# Question 1 

print("Q1")

def modGramSchmidt(X):
	m, n = np.shape(X)
	q = np.zeros_like(X)
	r = np.zeros((n, n))

	r[0][0] = np.sqrt(np.inner(X[:,0], X[:,0]))
	if r[0,0] == 0: return 
	else: 
		q[:,0] = X[:,0] / r[0,0]

	for j in range(1, n):
		q_hat = X[:,j]
		for i in range(0, j):
			r[i,j] = np.inner(q_hat, q[:, i])
			q_hat = q_hat - r[i,j]*q[:,i]
		r[j,j] = np.sqrt(np.inner(q_hat, q_hat))
		if r[j,j] == 0: return 
		else: 
			q[:,j] = q_hat / r[j,j]

	return q, r

Q, r = modGramSchmidt(X)
# print(np.matmul(Q,r))
# Q, r = np.linalg.qr(X)
# print(np.matmul(Q, r))

#Back Substitution RB = y_star

y_star = np.matmul(np.transpose(Q), random_y_vec)
n = len(y_star)
B_hat1 = np.zeros((3,1))

for j in range(2, -1, -1):
	summ = 0
	for m in range(j+1, n):
		summ += B_hat1[m] * r[j, m]
	B_hat1[j] += (y_star[j] - summ) / r[j,j]
	
print(B_hat1)
fitted_y_vec1 = np.matmul(X, B_hat1)
print(fitted_y_vec1[:5])

# Question 2
''' Since we have an overconstrained data set i.e. with more data that model parameters: '''
print("Q2")
XT = np.transpose(X)
XTX = np.matmul(XT, X)
XTX_inv = np.linalg.inv(XTX)
B_hat2 = np.matmul(np.matmul(XTX_inv, XT), random_y_vec)
print(B_hat2)
fitted_y_vec2 = np.matmul(X, B_hat2)
print(fitted_y_vec2[:5])

# Question 3
print("Q3")

# U, SIGMA, V = sci.linalg.svd(X)

def SVD(X):
    C = np.matmul(np.transpose(X), X)
    eigenvalues, eigenvectors = sci.linalg.eig(C)
    SIGMA = np.sqrt(np.real(eigenvalues))
    SIGMA = np.diag(SIGMA)
    V = eigenvectors
    n = len(X[0])
    m = len(X)
    # print(V)
    # print(SIGMA)
    U = np.zeros_like(X)

    for i in range(0,n):
        U[:,i] = np.matmul(X, V[:,i]) / SIGMA[i,i]
    return U, SIGMA, V

U, SIGMA, V = SVD(X)

# print(np.matmul(np.matmul(U, SIGMA), np.transpose(V))) 
SIGMA_Vtranspose_inv = sci.linalg.inv(np.matmul(SIGMA, np.transpose(V)))
Utranspose_Y = np.matmul(np.transpose(U), random_y_vec) # u transpose
B_hat3 = np.matmul(SIGMA_Vtranspose_inv, Utranspose_Y)
print(B_hat3)
fitted_y_vec3 = np.matmul(X, B_hat3)
print(fitted_y_vec3[:5])

# Question 4
print("Q4")
x0 = np.array([50, -25, 100]).reshape(3,1)
def fun(B_hat, X, y):
	XB = np.matmul(X, B_hat)
	
	res = np.subtract(y, XB)
	result = sci.linalg.norm(res, ord = 2)
	return result

B_hat4 = sci.optimize.fmin(fun, x0, args = (X, random_y_vec))#, method = 'Nelder-Mead')
print(B_hat4)
fitted_y_vec4 = np.matmul(X, B_hat4)
print(fitted_y_vec4[:5])

# PLOT TRUE MODEL, PREDICTED, AND ESTIMATED MODEL


