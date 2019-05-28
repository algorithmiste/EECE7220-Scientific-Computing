import numpy as np
import pandas as pd
import scipy as sci
import scipy.optimize as opt
import math
import matplotlib.pyplot as plt
# import seaborn as sns
plt.style.use('seaborn-whitegrid')


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
    
    U = np.zeros_like(X)
    for i in range(0,n):
        U[:,i] = np.matmul(X, V[:,i]) / SIGMA[i,i]
    return U, SIGMA, V

U, SIGMA, V = SVD(X)
 
SIGMA_Vtranspose_inv = sci.linalg.inv(np.matmul(SIGMA, np.transpose(V)))
Utranspose_Y = np.matmul(np.transpose(U), random_y_vec) # u transpose
B_hat3 = np.matmul(SIGMA_Vtranspose_inv, Utranspose_Y)
fitted_y_vec3 = np.matmul(X, B_hat3)

print(B_hat3)

random_y_vec = random_y_vec.reshape((100))
fitted_y_vec3 = fitted_y_vec3.reshape((100))

df = np.transpose(pd.DataFrame(np.array([x_vec, y_vec, random_y_vec, fitted_y_vec3])))
df.columns = ['x_vec', 'y_vec', 'random_y_vec', 'fitted_y_vec3']
# print(df)
df = pd.DataFrame.sort_values(df, by ='x_vec', axis = 0)

fig = plt.figure()
ax = plt.axes()
ax.margins()
ax.plot(df['x_vec'], df['y_vec'], 'r--', label = 'y' )
ax.plot(df['x_vec'], df['random_y_vec'], 'b--', label = 'random_y')
ax.plot(df['x_vec'], df['fitted_y_vec3'], 'g+', label = 'fitted_y')
ax.legend(loc = 'best')
props = {
    'title': 'Fitting Y-values using Singular Value Decomposition',
    'xlabel': 'X',
    'ylabel': 'Y'
    }
ax.set(**props)
plt.show()


