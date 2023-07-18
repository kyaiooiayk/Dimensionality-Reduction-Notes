#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
What? PCA -> Principal Component Analysis

PCA is a technique that comes from the field of linear algebra and can be used as a data preparation 
technique to create a projection of a dataset prior to fitting a model.
"""


# In[4]:


# Import python modules
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot


# ### First run

# In[3]:


# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5,
    random_state=7)

# define the pipeline
steps = [('pca', PCA(n_components=10)), ('m', LogisticRegression())]
model = Pipeline(steps=steps)

# evaluate model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1) 
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1) 

# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


# ### Optimising solution

# In[ ]:


"""
How do we know that reducing 20 dimensions of input down to 10 is good or the best we can do? We don’t; 
10 was an arbitrary choice. A better approach is to evaluate the same transform and model with different 
numbers of input features and choose the number of features (amount of dimensionality reduction) that 
results in the best average performance
"""


# In[6]:


# get the dataset
def get_dataset():
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
    return X, y

# get a list of models to evaluate
def get_models():
    models = dict()
    for i in range(1,21):
        steps = [('pca', PCA(n_components=i)), ('m', LogisticRegression())]
        models[str(i)] = Pipeline(steps=steps)
    return models

# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1) 
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1) 
    return scores

# define dataset
X, y = get_dataset()
# get the models to evaluate
models = get_models()

# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
    scores = evaluate_model(model, X, y)
    results.append(scores)
    names.append(name)
    print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))


# In[7]:


# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.xticks(rotation=45)
pyplot.show()


# In[ ]:


"""
We see a general trend of increased performance as the number of dimensions is increased. 
On this dataset, the results suggest a trade-off in the number of dimensions vs. the 
classification accuracy of the model. Interestingly, we don’t see any improvement beyond 
15 components. This matches our definition of the problem where only the first 15 components
contain information about the class and the remaining five are redundant.
"""


# ### PCA and logistic regression combination

# In[11]:


# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5,
    random_state=7)

# define the model
steps = [('pca', PCA(n_components=15)), ('m', LogisticRegression())] 
model = Pipeline(steps=steps)

# fit the model on the whole dataset
model.fit(X, y)

# make a single prediction
row = [[0.2929949, -4.21223056, -1.288332, -2.17849815, -0.64527665, 2.58097719,
    0.28422388, -7.1827928, -1.91211104, 2.73729512, 0.81395695, 3.96973717, -2.66939799,
    3.34692332, 4.19791821, 0.99990998, -0.30201875, -4.43170633, -2.82646737, 0.44916808]]
yhat = model.predict(row) 
print('Predicted Class: %d' % yhat[0])


# In[ ]:


"""
Running the example fits the Pipeline on all available data and makes a prediction on new data. 
Here, the transform uses the 15 most important components from the PCA transform, as we found 
from testing above. A new row of data with 20 columns is provided and is automatically transformed 
to 15 components and fed to the logistic regression model in order to predict the class label.
"""

