import numpy as np
import pandas as pd
import scipy as sci
import scipy.optimize as opt
import math
# import seaborn as sns
import matplotlib.pyplot as plt

# Data Generation:

B0 = 1.25
B1 = -0.2
B2 = 5.3

x_vec = np.array([np.random.randint(-50, 50) for i in range(100) ])
ones = np.ones((100))
X = np.array([ones, x_vec, x_vec**2])
X = np.transpose(X)

B = np.array([B0, B1, B2])
# y_vec2 = np.array([(B0 + B1 * x_vec[j] + B2 * x_vec[j]**2) for j in range(100)])
y_vec = np.matmul(X, B)

'''Now the generated data points (xi, yi) for i = 1 to 100 lie exactly on a quadratic polynomial.
Add random variations to each of the data points as follows: yi = f(xi) + error(i)
where the random variable error(i) follows a normal distribution with 0 mean and sigma standard
deviation. i.e. error(i) ~ Normal(0, sigma^2 ). '''
sigma = 10000
np.random.seed(1)
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

#Back Substitution RB = y_star

y_star = np.matmul(np.transpose(Q), random_y_vec)
n = len(y_star)
B_hat1 = np.zeros((3,1))

for j in range(2, -1, -1):
	summ = 0
	for m in range(j+1, n):
		summ += B_hat1[m] * r[j, m]
	B_hat1[j] += (y_star[j] - summ) / r[j,j]
	
fitted_y_vec1 = np.matmul(X, B_hat1)
print(B_hat1)
random_y_vec = random_y_vec.reshape((100))
fitted_y_vec1 = fitted_y_vec1.reshape((100))

df = np.transpose(pd.DataFrame(np.array([x_vec, y_vec, random_y_vec, fitted_y_vec1])))
df.columns = ['x_vec', 'y_vec', 'random_y_vec', 'fitted_y_vec1']
df = pd.DataFrame.sort_values(df, by ='x_vec', axis = 0)

fig = plt.figure()
ax = plt.axes()
ax.margins()
ax.plot(df['x_vec'], df['y_vec'], 'r--', label = 'y' )
ax.plot(df['x_vec'], df['random_y_vec'], 'b--', label = 'random_y')
ax.plot(df['x_vec'], df['fitted_y_vec1'], 'g+', label = 'fitted_y')
ax.legend(loc = 'best')
props = {
    'title': 'Fitting Y-values using Modified Gram Schmidt',
    'xlabel': 'X',
    'ylabel': 'Y'
    }
ax.set(**props)
plt.show()