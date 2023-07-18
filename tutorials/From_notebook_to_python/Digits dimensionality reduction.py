#!/usr/bin/env python
# coding: utf-8

# # Introduction

# In[ ]:


"""
What? Digits dimensionality reduction

VanderPlas, Jake. Python data science handbook: Essential tools for working with data. "O'Reilly Media, Inc.", 2016.
https://github.com/jakevdp/PythonDataScienceHandbook
"""


# # Import modules

# In[28]:


from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from matplotlib import rcParams
import seaborn as sns
from sklearn.metrics import accuracy_score 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


# # Dataset

# In[ ]:


"""
The images data is a three-dimensional array: 1,797 samples, each consisting of an 8×8 grid of pixels. 
We need a two-dimensional, [n_samples, n_features] representation. We can accomplish this by treating 
each pixel in the image as a feature—that is, by flattening out the pixel arrays so that we have a 
length-64 array of pixel values representing each digit.
"""


# In[2]:


digits = load_digits()
digits.images.shape


# In[9]:


X = digits.data
X.shape


# In[10]:


y = digits.target
y.shape


# # Visualisation

# In[8]:


fig, axes = plt.subplots(10, 10, figsize=(15, 8),
                                     subplot_kw={'xticks':[], 'yticks':[]},
                                     gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='binary', interpolation='nearest') 
    ax.text(0.05, 0.05, str(digits.target[i]),
                        transform=ax.transAxes, color='green')


# # Unsupervised learning: Dimensionality reduction

# In[ ]:


"""
We’d like to visualize our points within the 64-dimensional parameter space, but it’s difficult to 
effectively visualize points in such a high-dimensional space. Instead we’ll reduce the dimensions to 2, 
using an unsupervised method. We'll make use if a manifold learning algorithm called Isomap.
"""


# In[12]:


iso = Isomap(n_components=2)
iso.fit(digits.data)
data_projected = iso.transform(digits.data)
data_projected.shape


# In[ ]:


"""
We see that the projected data is now two-dimensional. Let’s plot this data to see if we can learn anything 
from its structure
"""


# In[18]:


rcParams['figure.figsize'] = 15, 8
rcParams['font.size'] = 20
plt.scatter(data_projected[:, 0], data_projected[:, 1], c=digits.target,
                        edgecolor='none', alpha=0.5,
                        cmap=plt.cm.get_cmap('Accent', 10))
plt.colorbar(label='digit label', ticks=range(10))
plt.clim(-0.5, 9.5)


# In[ ]:


"""
This plot gives us some good intuition into how well various numbers are separated in the larger 64-dimensional 
space. For example, zeros (in black) and ones (in purple) have very little overlap in parameter space. Intuitively,
this makes sense: a zero is empty in the middle of the image, while a one will generally have ink in the middle. 
On the other hand, there seems to be a more or less continuous spectrum between ones and fours: we can understand 
this by realizing that some people draw ones with “hats” on them, which cause them to look similar to fours.
"""


# # Classification on digits

# In[ ]:


"""
The different groups appear to be fairly well separated in the parameter space: this tells 
us that even a very straightforward supervised classification algorithm should perform suitably on this data. 
Let’s give it a try. We'll fit a Gaussian naive Bayes model.
"""


# In[23]:


Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)
model = GaussianNB()
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)


# In[ ]:


"""
Now that we have predicted our model, we can gauge its accuracy by comparing the true values of the test set 
to the predictions:
"""


# In[25]:


accuracy_score(ytest, y_model)


# In[ ]:


"""
However, this single number doesn’t tell us where we’ve gone wrong— one nice way to do this is to use the
confusion matrix which SHOWS the frequency of misclassifications by our classifier.
"""


# In[29]:


mat = confusion_matrix(ytest, y_model)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')


# In[ ]:


"""
This shows us where the mislabeled points tend to be: for example, a large number of twos here are misclassified 
as either ones or eights. Another way to gain intuition into the characteristics of the model is to plot the inputs
again, with their predicted labels. We’ll use green for correct labels, and red for incorrect labels
"""


# In[31]:


fig, axes = plt.subplots(10, 10, figsize=(15, 8),
                                     subplot_kw={'xticks':[], 'yticks':[]},
                                     gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='binary', interpolation='nearest') 
    ax.text(0.05, 0.05, str(y_model[i]),
    transform=ax.transAxes,
    color='green' if (ytest[i] == y_model[i]) else 'red')


# In[ ]:




