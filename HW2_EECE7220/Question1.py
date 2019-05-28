import scipy
import numpy as np
from scipy import linalg
from scipy.linalg import eigh
import pandas as pd



# Question1: (a) Write a computer program to implement the spectral method for solving Ax = b
#			 (b) In your program, test whether the eigenvectors of A are orthogonal
# Input: For any m, an MxM matrix A and an Mx1 vector b
# Output: an Mx1 vector x
# inner product: np.inner(a,b)
# eigenvalues: scipy.linalg.eigvals(A)
# eigenvectors: scipy.linalg.eig(A)

#A = np.diag(d) + np.diag(e, k=1) + np.diag(e, k=-1)

def spectral(A,b):

	eigenvalues_Of_A = scipy.linalg.eigvalsh(A).astype(float); 
	eigenvalues_Of_A, eigenvectors_Of_A = scipy.linalg.eigh(A); 
	# eigDF = pd.DataFrame(eigenvectors_Of_A)  #eigenvectors are the columns from eigh() so take transpose

	eigvectors_Transposed = eigenvectors_Of_A.transpose()
	eigenvectors_Of_A = eigvectors_Transposed

	
	x_vector = np.zeros_like(b)
	for i in range(len(b)):
		constantResult = np.inner(b, eigenvectors_Of_A[i]) / ((eigenvalues_Of_A[i]) * np.inner(eigenvectors_Of_A[i], eigenvectors_Of_A[i]))
		x_vectorI = constantResult * eigenvectors_Of_A[i]
		x_vector += x_vectorI

	print(x_vector)
	return 


#A = np.matrix([[-2,1,0,0,0,0,0,0,0],[1,-2,1,0,0,0,0,0,0],[0,1,-2,1,0,0,0,0,0],[0,0,1,-2,1,0,0,0,0],[0,0,0,1,-2,1,0,0,0],[0,0,0,0,1,-2,1,0,0],[0,0,0,0,0,1,-2,1,0],[0,0,0,0,0,0,1,-2,1],[0,0,0,0,0,0,0,1,-2]])
b = np.repeat(-0.01,9)
A = np.diag(np.repeat(-2,9)) + np.diag(np.repeat(1,8), k=1) + np.diag(np.repeat(1,8), k=-1)

print(spectral(A,b))

