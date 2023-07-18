#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Create-dataset" data-toc-modified-id="Create-dataset-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Create dataset</a></span></li><li><span><a href="#PCA-modelling" data-toc-modified-id="PCA-modelling-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>PCA modelling</a></span></li><li><span><a href="#How-to-use-compopnent-and-explained-variance" data-toc-modified-id="How-to-use-compopnent-and-explained-variance-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>How to use compopnent and explained variance</a></span></li><li><span><a href="#PCA-as-dimensionality-reduction" data-toc-modified-id="PCA-as-dimensionality-reduction-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>PCA as dimensionality reduction</a></span></li></ul></div>

# # Introduction

# In[ ]:


"""
What? Introduction to PCA

We explore what is perhaps one of the most broadly used of unsuper‐ vised algorithms, principal component 
analysis (PCA). PCA is fundamentally a dimensionality reduction algorithm, but it can also be useful as a 
tool for visualization, for noise filtering, for feature extraction and engineering, and much more

VanderPlas, Jake. Python data science handbook: Essential tools for working with data. "O'Reilly Media, Inc.", 2016.
https://github.com/jakevdp/PythonDataScienceHandbook
"""


# # Imports
# <hr style = "border:2px solid black" ></hr>

# In[3]:


import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set()
from sklearn.decomposition import PCA 


# # Create dataset

# In[2]:


rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
plt.scatter(X[:, 0], X[:, 1])
plt.axis('equal');


# # PCA modelling

# In[ ]:


"""
The problem setting here is slightly different than regression: rather than attempting to predict the y values from 
the x values, the PCA attempts to learn about the relationship between the x and y values by finding a list of the 
principal axes in the data, and using those axes to describe the dataset.
"""


# In[12]:


pca = PCA(n_components=2)
pca.fit(X)


# In[13]:


print(pca.components_)


# In[14]:


print(pca.explained_variance_)


# # How to use compopnent and explained variance

# In[ ]:


"""
To see what these numbers mean, let’s visualize them as vectors over the input data, using the “components” 
to define the direction of the vector, and the “explained var~iance” to define the squared-length of the vector.
"""


# In[29]:


def draw_vector(v0, v1, ax=None): 
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                   linewidth=2,
                   shrinkA=0, shrinkB=0, color = "k")
    ax.annotate('', v1, v0, arrowprops=arrowprops)


# In[30]:


plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)
plt.axis('equal');


# In[ ]:


"""
These vectors represent the principal axes of the data, and the length shown in Figure is an indication
of how “important” that axis is in describing the distribution of the data—more precisely, it is a measure 
of the variance of the data when pro‐ jected onto that axis.
"""


# # PCA as dimensionality reduction

# In[ ]:


"""
Using PCA for dimensionality reduction involves zeroing out one or more of the smallest principal components, 
resulting in a lower-dimensional projection of the data that preserves the maximal data variance.
"""


# In[32]:


pca = PCA(n_components=1)
pca.fit(X)
X_pca = pca.transform(X) 
print("original shape: ", X.shape) 
print("transformed shape:", X_pca.shape)


# In[ ]:


"""
The transformed data has been reduced to a single dimension. To understand the effect of this dimensionality 
reduction, we can perform the inverse transform of this reduced data and plot it along with the original data
"""


# In[35]:


X_new = pca.inverse_transform(X_pca)
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8, c="k")
plt.axis('equal')


# In[ ]:


"""
The light points are the original data, while the dark points are the projected version. This makes clear what a 
PCA dimensionality reduction means: the information along the least important principal axis or axes is removed, 
leaving only the component(s) of the data with the highest variance. The fraction of variance that is cut out 
(proportional to the spread of points) is roughly a meas‐ ure of how much “information” is discarded in this 
reduction of dimensionality.


This reduced-dimension dataset is in some senses “good enough” to encode the most important relationships between 
the points: despite reducing the dimension of the data by 50% (from 2 down to 1), the overall relationship between 
the data points is mostly preserved.
"""

