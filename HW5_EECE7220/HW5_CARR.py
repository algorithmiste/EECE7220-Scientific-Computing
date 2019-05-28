import numpy as np
import pandas as pd
import scipy as sci
import scipy.optimize as opt
import math

def modGS(h, ei, y):
	m, n = np.shape(h)
	
	q = np.zeros_like(h)
	q = pd.DataFrame(q)
	r = np.zeros((n, n))
	r = pd.DataFrame(r)
	r.loc[0,0] = np.sqrt(np.inner(h.loc[:,0], h.loc[:,0]))
	if r.loc[0,0]  == 0: return 
	else: 
		q.loc[:,0] = h.loc[:,0]/r.loc[0,0] 
	for j in range(1, n):
		q_hat = h.loc[:,j]
		for i in range(0, j):
			r.loc[i,j] = np.inner(q_hat, q.loc[:, i])
			q_hat = q_hat - r.loc[i,j]*q.loc[:,i]
		r.loc[j,j] = np.sqrt(np.inner(q_hat, q_hat))
		if r.loc[j,j] == 0: return 
		else: 
			q.loc[:,j] = q_hat / r.loc[j,j]
	y_star = np.matmul(q, y) 

	# print(np.shape(r))
	n = len(y_star)
	B_hat = np.zeros((11,1))  
	for j in range(10, -1, -1):
		summ = 0
		for u in range(j+1, n-1):
			
			summ += B_hat[u] * r[j].iloc[u]

		B_hat[j] += (y_star[j] - summ) / r.loc[j,j]
	return B_hat

# Data Generation: To evaluate GMRES, generate test data {(xi, yi)}, i in (0, 10) that has a decic relationship
# as follows: 

beta = np.array([1,-567/1562500,64161/6250000,-11727/100000,4523/6250,-10773/4000,63273/10000,-189/20,8.7,-4.5,1])
x_vec = np.array([-0.9000,-0.7200,-0.5400,-0.3600,-0.1800,0,0.1800,0.3600,0.5400,0.7200,0.9000])
print(beta)
X = np.zeros((11,11))
cur_index = 0
for x in x_vec:
	for i in range(len(x_vec)):
		X[cur_index,i] = x**i
	cur_index += 1

X = pd.DataFrame(X)
y = np.zeros((11,1))
for j in range(len(y)):
	y[j] = np.matmul(X.iloc[j], beta)

# Generalized Mean Residual (GMRES)
x0 = np.zeros((11,1)) 
k = 11 
def GMRES(X, y, x0, k):
	x_vec = np.zeros((11,1))
	h = np.zeros((k+1, k)); h = pd.DataFrame(h)
	v = np.zeros((11,11))
	v = pd.DataFrame(v)
	r = []
	res = np.matmul(X,x0)
	r += [y - res] 
	v.loc[:,0] = np.divide(r[0],sci.linalg.norm(r[0]))

	w = []
	for i in range(k):
		w += [np.matmul(X, v[i])]
		k = -1
		for j in range(i):
			h.loc[j,i] = np.inner(np.transpose(w[i]), np.transpose(v[j])) 
			w[i] = w[i] - h[i].iloc[j]*v[j]
			k = j
		k+=1
		h[i].iloc[k] = sci.linalg.norm(w[i])
		v.loc[:,i+1] = np.divide(w[i], h[i].iloc[k])

	#Find yi which minimizes the current residual
	if i == 0:
		ei = np.ones(1)
	else:
		a = np.ones(1)
		b = np.zeros(i)
		ei = np.hstack((a,b))
	normr_ei = sci.linalg.norm(r[0])*ei
	yi = modGS(h, normr_ei, y)
	v = v.loc[:10,:10]
	x_vec = np.matmul(v, yi) 
	return x_vec
print(GMRES(X, y, x0, k))



