import numpy as np
import pandas as pd
import scipy as sci
import scipy.optimize as opt
import math
import seaborn as sns
import matplotlib.pyplot as plt

# Data Generation:

B0 = 1.25
B1 = -0.2
B2 = 5.3

x_vec = np.array([np.random.randint(-50, 50) for i in range(100) ])
ones = np.ones(100)
Xmat = np.array([ones, x_vec, x_vec**2])
X = np.transpose(Xmat)

B = np.array([B0, B1, B2])
# y_vec2 = np.array([(B0 + B1 * x_vec[j] + B2 * x_vec[j]**2) for j in range(100)])
y_vec = np.matmul(X, B)
np.set_printoptions(precision = 3, suppress = True)

'''Now the generated data points (xi, yi) for i = 1 to 100 lie exactly on a quadratic polynomial.
Add random variations to each of the data points as follows: yi = f(xi) + error(i)
where the random variable error(i) follows a normal distribution with 0 mean and sigma standard
deviation. i.e. error(i) ~ Normal(0, sigma^2 ). '''
sigma = 10000
np.random.seed(1)
error_vec = np.array([sigma * np.random.randn() for i in range(100)])
random_y_vec = np.array(np.add(y_vec, error_vec)).reshape(100,1)

# Question 4
print("Q4")
x0 = np.array([50, -25, 100]).reshape(3,1)
def fun(B_hat, X, y):
	XB = np.matmul(X, B_hat)
	
	res = np.subtract(y, XB)
	result = sci.linalg.norm(res, ord = 2)
	return result

B_hat4 = sci.optimize.fmin(fun, x0, args = (X, random_y_vec))
fitted_y_vec4 = np.matmul(X, B_hat4)

print(B_hat4)
random_y_vec = random_y_vec.reshape((100))
fitted_y_vec4 = fitted_y_vec4.reshape((100))

df = np.transpose(pd.DataFrame(np.array([x_vec, y_vec, random_y_vec, fitted_y_vec4])))
df.columns = ['x_vec', 'y_vec', 'random_y_vec', 'fitted_y_vec4']
df = pd.DataFrame.sort_values(df, by ='x_vec', axis = 0)

fig = plt.figure()
ax = plt.axes()
ax.margins()
ax.plot(df['x_vec'], df['y_vec'], 'r--', label = 'y' )
ax.plot(df['x_vec'], df['random_y_vec'], 'b--', label = 'random_y')
ax.plot(df['x_vec'], df['fitted_y_vec4'], 'g+', label = 'fitted_y')
ax.legend(loc = 'best')
props = {
    'title': 'Fitting Y-values using Optimization',
    'xlabel': 'X',
    'ylabel': 'Y'
    }
ax.set(**props)
plt.show()


