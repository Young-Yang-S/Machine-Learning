# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 14:02:38 2020

@author: daiya
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                    'machine-learning-databases/wine/wine.data',
                     header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

print(df_wine['Class label'].unique())  #get classes number,3 classes total


X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.25, 
                     stratify=y,
                     random_state=10)
    

sc = StandardScaler() # standardizing data
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# here we should notice that when we do standardization, we should use sc.fit_transform for
# training data, and sc.transform to test data, which means we should use training mean
# and std to standardize the test data set rather rather using test data itself's mean
# and std to do the standardization.

cov_mat = np.cov(X_train_std.T)
# convariance matrics
eigen_vals, eigen_vecs = np.linalg.eigh(cov_mat)
# eigen vector and eigen value
print('\nEigenvalues \n%s' % eigen_vals)


# It would be better to use numpy.linalg.eigh, which has been designed for Hermetian matrices. It always returns real eigenvalues; whereas the numerically less stable np.linalg.eig can decompose nonsymmetric square matrices, you may find that it returns complex eigenvalues in certain cases. (S.R.)


total = sum(eigen_vals)
var_exp = [(i / total) for i in sorted(eigen_vals, reverse=True)]
# calculate the percentage of each eigen vector
cum_var_exp = np.cumsum(var_exp)
# calculate the cumulative sum


plt.bar(range(1, 14), var_exp, alpha=0.5, align='center',
        label='individual explained variance')
plt.step(range(1, 14), cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
# plt.savefig('images/05_02.png', dpi=300)
plt.show()


# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda k: k[0], reverse=True)


w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W:\n', w)

# here hstack is used to stack two vector into one matrics, 
# then the newaxis is to increase the demensions, for example if we have [2,0,2,0] 1D array, then if we use [:np.newaxis], we get [[2],[0],[2],[0]] 4*1 two D array, 
# if we use [np.newaxis,:], we get [[2,0,2,0]] 1*4 2D array.


X_train_pca = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train == l, 0], 
                X_train_pca[y_train == l, 1], 
                c=c, label=l, marker=m)
# here we use zip to make sure in one for loop we can go through all of these variables

plt.xlabel('PC 1D')
plt.ylabel('PC 2D')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('images/05_03.png', dpi=300)
plt.show() 

# above is how we achieve the pca manually, actually we can achieve simply in sklearn

from sklearn.decomposition import PCA

pca = PCA()
X_train_pca = pca.fit_transform(X_train_std)
pca.explained_variance_ratio_ # eigenvalue

plt.bar(range(1, 14), pca.explained_variance_ratio_, alpha=0.5, align='center')
plt.step(range(1, 14), np.cumsum(pca.explained_variance_ratio_), where='mid')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.show()

pca = PCA(n_components=2)  # select only two top eigenvalues
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
print(pca.explained_variance_ratio_)

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train == l, 0], 
                X_train_pca[y_train == l, 1], 
                c=c, label=l, marker=m)

plt.ylabel('PC 2D')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show() 

# then we use logistic regression to classify data after PCA
from sklearn.linear_model import LogisticRegression

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
lr = LogisticRegression()
lr = lr.fit(X_train_pca, y_train)


from matplotlib.colors import ListedColormap
# this part is to show the classification boundary and classified points
def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # here we get 2D features lower and upper boundry +-1 to prepare for drawing the boundary area
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    # meshgrid is a good way to get all possible points within this 2D boundary area, resolution limits the interval for each grid point 
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    # then we predict every point within this plot area, here ravel is to decrease 1 demension of array and then we apply np.array to get each pair of points within this area
    Z = Z.reshape(xx1.shape)
    # here we reshape Z to xx1 shape (actually here we put 1D Z into 2D Z)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    # Then contourf is to draw the context where the Z value is the same, and color we have chosen before (that means if y label is equal to an element in Z, they put the same color)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # then we restrict our interval to min max of our two axises

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.6, 
                    c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx], 
                    label=cl)

plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('images/05_04.png', dpi=300)
plt.show()

plot_decision_regions(X_test_pca, y_test, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('images/05_05.png', dpi=300)
plt.show()
# this is our test points, from which we can see after PCA (even only for 2 features), we can still keep the key information of our data, and classify them with a high accuarcy 

