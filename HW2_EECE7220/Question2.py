# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 12:01:41 2018

@author: casey
"""

#   a. Write a general computer program to implement a reduced singular value decomposition (SVD) method of an 
#       MxN matrix A with n << m

'''Input: Use images of a scene under three di
erent illumination to form the data matrix
    A as follows. Recall that images can be read using MATLAB function imread() and
    that a grayscale image is in a matrix form. Let I1; I2; I3 be the three image matrices.
    Vectorize the image matrices (using the colon operator, e.g. vector a1 = I1(:)). Form A with
    a1; a2; a3 as column entries of A. '''
    
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
import pandas as pd


def reducedSVD(A):
    
    C = np.matmul(np.transpose(A), A)
    eigenvalues_Of_C, eigenvectors_Of_C = sci.linalg.eig(C)
#    print(eigenvalues_Of_C)
    eigenvalues_Of_C = eigenvalues_Of_C
    eigenvalues_Of_C = eigenvalues_Of_C.astype(float)
    
    #dF = pd.DataFrame(eigenvectors_Of_C)
    eigenvectors_Of_C = np.transpose(eigenvectors_Of_C)
    
    #leftEig = np.zeros_like([[np.repeat(0, 1228800)],[np.repeat(0,1228800)],[np.repeat(0,1228800)]]) # 3 vectors of 10^6
    #for i in range(len(eigenvectors_Of_C)):
    leftEig1 = np.matmul(A, eigenvectors_Of_C[0]) / eigenvalues_Of_C[0]
    leftEig2 = np.matmul(A, eigenvectors_Of_C[1]) / eigenvalues_Of_C[1]
    leftEig3 = np.matmul(A, eigenvectors_Of_C[2]) / eigenvalues_Of_C[2]
    
    leftEig = np.transpose(np.matrix([leftEig1,leftEig2,leftEig3]))
    
    #   D. Show that the left singular vectors in the U matrix form an orthogonal set.
    innerU1withU2 = np.dot(leftEig1, leftEig2)
    innerU1withU3 = np.dot(leftEig1, leftEig3)
    innerU2withU3 = np.dot(leftEig2,leftEig3)
    
    inner = np.matrix([innerU1withU2, innerU1withU3, innerU2withU3])
    
    #   E. Show that the left singular vectors in U span the images in the data matrix A.
    
    alpha1 = np.inner(A[:,0], leftEig1) / np.inner(leftEig1,leftEig1)
    alpha2 = np.inner(A[:,0], leftEig2) / np.inner(leftEig2,leftEig2)
    alpha3 = np.inner(A[:,0], leftEig3) / np.inner(leftEig3,leftEig3)
    
    # Recover 1st Image
    image1_recovered = alpha1*leftEig1 
    image1_recovered = np.reshape(image1_recovered, (960, 1280))
    plt.imsave("C://Users//casey//Desktop//Scientific Computing//EECE7220_ProgAssignments//HW2_EECE7220//image1_recovered.jpg",image1_recovered)
    # Recover 2nd Image
    image2_recovered = alpha2*leftEig2 
    image2_recovered = np.reshape(image2_recovered, (960, 1280))
    plt.imsave("C://Users//casey//Desktop//Scientific Computing//EECE7220_ProgAssignments//HW2_EECE7220//image2_recovered.jpg",image2_recovered)
    # Recover 3rd Image
    image3_recovered = alpha3*leftEig3 
    image3_recovered = np.reshape(image3_recovered, (960, 1280))
    plt.imsave("C://Users//casey//Desktop//Scientific Computing//EECE7220_ProgAssignments//HW2_EECE7220//image3_recovered.jpg",image3_recovered)
    
    #   F. Create at least six new images using the left singular vectors in U that are not in the
    #      original data matrix A. Save your images using imwrite() as
    #      Image4.jpg, Image5.jpg, ..., Image9.jpg.
    #      Hint: Use the concepts of basis, span and vector space.
    
    #Image1 :
    Image1 = -16.25*alpha1*leftEig1
    Image1 = np.reshape(Image1, (960,1280))
    plt.imsave("C://Users//casey//Desktop//Scientific Computing//EECE7220_ProgAssignments//HW2_EECE7220//I1.jpg",Image1)
    #Image2 :
    Image2 = -982*alpha2*leftEig1
    Image2 = np.reshape(Image2, (960,1280))
    plt.imsave("C://Users//casey//Desktop//Scientific Computing//EECE7220_ProgAssignments//HW2_EECE7220//I2.jpg",Image2)
    #Image3 :
    Image3 = 23.24*alpha3*leftEig1
    Image3 = np.reshape(Image3, (960,1280))
    plt.imsave("C://Users//casey//Desktop//Scientific Computing//EECE7220_ProgAssignments//HW2_EECE7220//I3.jpg",Image3)
    #Image4:
    Image4 = 200.25*alpha3*leftEig1
    Image4 = np.reshape(Image4, (960,1280))
    plt.imsave("C://Users//casey//Desktop//Scientific Computing//EECE7220_ProgAssignments//HW2_EECE7220//I4.jpg",Image4)
    #Image5 :
    Image5 = 1000*alpha1*leftEig1
    Image5 = np.reshape(Image5, (960,1280))
    plt.imsave("C://Users//casey//Desktop//Scientific Computing//EECE7220_ProgAssignments//HW2_EECE7220//I5.jpg",Image5)
    #Image6 :
    Image6 = -45*alpha2*leftEig1
    Image6 = np.reshape(Image6, (960,1280))
    plt.imsave("C://Users//casey//Desktop//Scientific Computing//EECE7220_ProgAssignments//HW2_EECE7220//I6.jpg",Image6)

    
    return leftEig

I1 = plt.imread("C://Users//casey//Desktop//Scientific Computing//EECE7220_ProgAssignments//HW2_EECE7220//Image1.jpg")
I2 = plt.imread("C://Users//casey//Desktop//Scientific Computing//EECE7220_ProgAssignments//HW2_EECE7220//Image2.jpg")
I3 = plt.imread("C://Users//casey//Desktop//Scientific Computing//EECE7220_ProgAssignments//HW2_EECE7220//Image3.jpg")

I1 = np.reshape(I1, (1228800,1))
I2 = np.reshape(I2, (1228800,1))
I3 = np.reshape(I3, (1228800,1))

A = np.column_stack((I1,I2,I3))

print(reducedSVD(A))




    


