#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
What? PCA Principal Component Analysis

References:
"""


# In[8]:


# Import python modules
from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig
from sklearn.decomposition import PCA


# In[7]:


# [1] IMPLEMENTATION USING NUMPY

# define matrix
A = array([
  [1, 2],
  [3, 4],
[5, 6]])

print("Original matrix \n", A)

# column means
M = mean(A.T, axis=1)
# center columns by subtracting column means 
C=A-M
# calculate covariance matrix of centered matrix 
V = cov(C.T)
# factorize covariance matrix
values, vectors = eig(V)
print(vectors)
print(values)
# project data
P = vectors.T.dot(C.T)
print(P.T)

"""
Interestingly, we can see that only the first eigenvector is required, 
suggesting that we could project our 3 × 2 matrix onto a 3 × 1 matrix with little loss.
"""


# In[12]:


# [2] IMPLEMENTATION IN SKLERN

# create the transform
pca = PCA(2)
# fit transform
pca.fit(A)
# access values and vectors
print(pca.components_)
print(pca.explained_variance_)
# transform data
B = pca.transform(A)
print(B)


# In[ ]:




