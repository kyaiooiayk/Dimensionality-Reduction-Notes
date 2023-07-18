#!/usr/bin/env python
# coding: utf-8

# # Introduction

# In[ ]:


"""
What? PCA as Noise Filtering

PCA can also be used as a filtering approach for noisy data. The idea is this: any com‐ ponents with variance much 
larger than the effect of the noise should be relatively unaffected by the noise. So if you reconstruct the data 
using just the largest subset of principal components, you should be preferentially keeping the signal and throwing
out the noise.

VanderPlas, Jake. Python data science handbook: Essential tools for working with data. "O'Reilly Media, Inc.", 2016.
https://github.com/jakevdp/PythonDataScienceHandbook
"""


# # Import modules

# In[1]:


from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set()
from sklearn.decomposition import PCA 


# # Load dataset

# In[2]:


digits = load_digits()
digits.data.shape


# In[4]:


def plot_digits(data):
    fig, axes = plt.subplots(4, 10, figsize=(10, 4),
                            subplot_kw={'xticks':[], 'yticks':[]},
                            gridspec_kw=dict(hspace=0.1, wspace=0.1)) 
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8, 8),
        cmap='binary', interpolation='nearest', clim=(0, 16))
plot_digits(digits.data)


# # Adding noise

# In[ ]:


"""
Now let’s add some random noise to create a noisy dataset
"""


# In[5]:


np.random.seed(42)
noisy = np.random.normal(digits.data, 4)
plot_digits(noisy)


# # PCA training

# In[ ]:


"""
It’s clear by eye that the images are noisy, and contain spurious pixels. Let’s train a PCA on the noisy data, 
requesting that the projection preserve 50% of the variance:
"""


# In[6]:


pca = PCA(0.50).fit(noisy)
pca.n_components_


# In[ ]:


"""
Here 50% of the variance amounts to 12 principal components. Now we compute these components, and then use the 
inverse of the transform to reconstruct the filtered digits
"""


# In[7]:


components = pca.transform(noisy)
filtered = pca.inverse_transform(components)
plot_digits(filtered)


# In[ ]:


"""
This signal preserving/noise filtering property makes PCA a very useful feature selec‐ tion routine—for example, 
rather than training a classifier on very high-dimensional data, you might instead train the classifier on the 
lower-dimensional representation, which will automatically serve to filter out random noise in the inputs.
"""

