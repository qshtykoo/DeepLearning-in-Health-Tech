import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_swiss_roll

import pandas as pd
import numpy as np



swissData = make_swiss_roll(n_samples=5000, noise=0.0, random_state=None)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(swissData[0][:,0], swissData[0][:,1], swissData[0][:,2], c=swissData[1])
X = swissData[0]
y = swissData[1] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_train)

#convert the data into Pandas dataframe
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
labelDf = pd.DataFrame(data = y_train, columns=['label'])
finalDf = pd.concat([principalDf, labelDf], axis = 1)

#visualization method 2
g = sns.FacetGrid(finalDf, hue='label', size=12).map(plt.scatter, 'principal component 1', 'principal component 2').add_legend()
#plt.scatter(principalComponents[:,0], principalComponents[:,1], c=y_train)
