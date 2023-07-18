#!/usr/bin/env python
# coding: utf-8

# # Introduction

# In[ ]:


"""
What? Importance of Feature Scaling for PCA

Reference: https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html#sphx-glr-auto-examples-preprocessing-plot-scaling-importance-py
"""


# # Imports

# In[9]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.pipeline import make_pipeline
from matplotlib import rcParams


# # Import dataset

# In[4]:


RANDOM_STATE = 42
FIG_SIZE = (10, 7)
features, target = load_wine(return_X_y=True)
# Make a train/test split using 30% test size
X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                    test_size=0.30,
                                                    random_state=RANDOM_STATE)


# # Model 

# In[ ]:


"""
Feature scaling through standardization (or Z-score normalization) = rescaling the features such that they
have the properties of a standard normal distribution with a mean of zero and a standard deviation of one.

What happens if you do not standardise the data? We can illustrate it using PCA.
In PCA we are interested in the components that maximize the variance. If one component (e.g. human height) 
varies less than another (e.g. weight) because of their respective scales (meters vs. kilos), PCA might determine
that the direction of maximal variance more closely corresponds with the ‘weight’ axis, if those features are not 
scaled. As a change in height of one meter can be considered much more important than the change in weight of one
kilogram, this is clearly incorrect.


The dataset used is the Wine Dataset available at UCI. This dataset has continuous features that are heterogeneous
in scale due to differing properties that they measure (i.e alcohol content, and malic acid). The transformed data 
is then used to train a naive Bayes classifier, and a clear difference in prediction accuracies is observed wherein
the dataset which is scaled before PCA vastly outperforms the unscaled version.
"""


# In[6]:


# Fit to data and predict using pipelined GNB and PCA.
unscaled_clf = make_pipeline(PCA(n_components=2), GaussianNB())
unscaled_clf.fit(X_train, y_train)
pred_test = unscaled_clf.predict(X_test)

# Fit to data and predict using pipelined scaling, GNB and PCA.
std_clf = make_pipeline(StandardScaler(), PCA(n_components=2), GaussianNB())
std_clf.fit(X_train, y_train)
pred_test_std = std_clf.predict(X_test)
# Show prediction accuracies in scaled and unscaled data.
print('\nPrediction accuracy for the normal test dataset with PCA')
print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test)))

print('\nPrediction accuracy for the standardized test dataset with PCA')
print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test_std)))

# Extract PCA from pipeline
pca = unscaled_clf.named_steps['pca']
pca_std = std_clf.named_steps['pca']

# Show first principal components
print('\nPC 1 without scaling:\n', pca.components_[0])
print('\nPC 1 with scaling:\n', pca_std.components_[0])

# Use PCA without and with scale on X_train data for visualization.
X_train_transformed = pca.transform(X_train)
scaler = std_clf.named_steps['standardscaler']
X_train_std_transformed = pca_std.transform(scaler.transform(X_train))


# # Plotting

# In[14]:


rcParams['figure.figsize'] = 14, 8
rcParams['font.size'] = 15

# visualize standardized vs. untouched dataset with PCA performed
fig, (ax1, ax2) = plt.subplots(ncols=2)

for l, c, m in zip(range(0, 3), ('blue', 'red', 'green'), ('^', 's', 'o')):
    ax1.scatter(X_train_transformed[y_train == l, 0],
                X_train_transformed[y_train == l, 1],
                color=c,
                label='class %s' % l,
                alpha=0.5,
                marker=m
                )

for l, c, m in zip(range(0, 3), ('blue', 'red', 'green'), ('^', 's', 'o')):
    ax2.scatter(X_train_std_transformed[y_train == l, 0],
                X_train_std_transformed[y_train == l, 1],
                color=c,
                label='class %s' % l,
                alpha=0.5,
                marker=m
                )

ax1.set_title('Training dataset after PCA')
ax2.set_title('Standardized training dataset after PCA')

for ax in (ax1, ax2):
    ax.set_xlabel('1st principal component')
    ax.set_ylabel('2nd principal component')
    ax.legend(loc='upper right')
    ax.grid()

plt.tight_layout()

plt.show()


# In[ ]:




