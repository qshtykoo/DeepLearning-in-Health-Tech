import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
import pandas as pd
import seaborn as sns
import numpy as np

mnist = fetch_mldata('MNIST original')


# normalize the data
X = mnist.data / 255.0
y = mnist.target

#sample 5000 samples for visualization
num_samples_to_plot = 5000
X_train, y_train = shuffle(X, y)
X_train, y_train = X_train[:num_samples_to_plot], y_train[:num_samples_to_plot]

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

#visualization method 1
plt.figure()
plt.scatter(finalDf['principal component 1'], finalDf['principal component 2'],
            c=finalDf['label'], edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('Spectral', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar();


plt.style.use('fivethirtyeight')


#visualize the weights assigned to each pixel using heatmap
fig, axarr = plt.subplots(1, 2, figsize=(12, 4))

sns.heatmap(pca.components_[0, :].reshape(28, 28), ax=axarr[0], cmap='gray_r')
sns.heatmap(pca.components_[1, :].reshape(28, 28), ax=axarr[1], cmap='gray_r')
axarr[0].set_title(
    "{0:.2f}% Explained Variance".format(pca.explained_variance_ratio_[0]*100),
    fontsize=12
)
axarr[1].set_title(
    "{0:.2f}% Explained Variance".format(pca.explained_variance_ratio_[1]*100),
    fontsize=12
)
axarr[0].set_aspect('equal')
axarr[1].set_aspect('equal')

plt.suptitle('2-Component PCA')
plt.show()

