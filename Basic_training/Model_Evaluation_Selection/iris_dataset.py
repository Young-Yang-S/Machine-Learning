# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 23:49:42 2020

@author: daiya
"""
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.datasets import load_iris

iris = load_iris()
print(iris.keys)
print(iris.target_names)
print(iris.DESCR)
print(iris.feature_names)

y = iris['target']
X = iris['data']

# split traning and test dataset
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=10)

kf = KFold(n_splits=5, shuffle=True, random_state=10)

# xgboost model 
xgb_model_list = []
xgb_mse_list = []
for train_index, test_index in kf.split(train_X):
    xgb_model = xgb.XGBClassifier().fit(train_X[train_index], train_y[train_index])
    # fit the model
    xgb_model_list.append(xgb_model)
    # append each model for each cv process
    predictions = xgb_model.predict(train_X[test_index])
    actuals = train_y[test_index]
    #print(confusion_matrix(actuals, predictions))
    mse = mean_squared_error(actuals, predictions)
    xgb_mse_list.append(mse)
    # here actually we are doing classification task, we shouldn't use mse,
    # but here we just want to use it to have a try
    
print ('xgb_mse_list:{}'.format(xgb_mse_list))
print ('xgb mse average:{}'.format(np.mean(xgb_mse_list)))


# random forest model
rf_model_list = []
rf_mse_list = []
for train_index, test_index in kf.split(train_X):
    rf = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=10)
    rf.fit(train_X[train_index], train_y[train_index])
    rf_model_list.append(rf)
    predictions = rf.predict(train_X[test_index])
    actuals = train_y[test_index]
    mse = mean_squared_error(actuals, predictions)
    rf_mse_list.append(mse)

print ('rf_mse_list:{}'.format(rf_mse_list))
print ('rf mse average:{}'.format(np.mean(rf_mse_list)))

# model evaluation and pick
if np.mean(rf_mse_list) <= np.mean(xgb_mse_list):
    min_mse = min(rf_mse_list)
    ind = rf_mse_list.index(mse)
    best_estimator = rf_model_list[ind]
    print('best estimator is random forest {}, mse is {}'.format(ind,min_mse))
else:
    min_mse = min(xgb_mse_list)
    ind = xgb_mse_list.index(min(xgb_mse_list))
    best_estimator = xgb_model_list[ind]
    print('best estimator is xgb {}, mse is {}'.format(ind, min_mse))
# here we are selecting the sub model from 10 models (5 from xgboost, 5 from random forest)
# actually we are choosing the best one which has lowest mse score (in normal task we should 
# choose the one with minimal accuarcy score because this is classification task)    


# then use the best model to predict the test dataset 
pred = best_estimator.predict(test_X)
mse = mean_squared_error(pred, test_y)
print ('test data mse is:{}'.format(mse))
print(confusion_matrix(test_y, pred))

# the performance of this best model is good because this is a simple dataset.
# While we should know that this model selection could be more accurate beacuse 
# in this example we just pick the best one in one process of cv whereas  the better
# way to do is to adjust the hyper parameter first and get the best model of one algorithm
# then use cv to select the most best from different best algorithms, that is our final best model.


