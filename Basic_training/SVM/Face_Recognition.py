# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 17:38:06 2020

@author: daiya
"""

import numpy as np
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.datasets import fetch_olivetti_faces
import matplotlib
matplotlib.use('Agg') #TKAgg can show GUI in imshow()
# If use Agg, it won't show in GUI 
import matplotlib.pyplot as plt

    
# print out images (data set)
def print_faces(images, target, top_n):
    # set up figure size in inches
    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(top_n):
        # print images in matrix 20x20
        p = fig.add_subplot(20, 20, i + 1, xticks=[], yticks=[])
        p.imshow(images[i], cmap=plt.cm.bone)
     # following is to label the image with target value
     #   p.text(0, 14, str(target[i]))
     #   p.text(0, 60, str(i))
    

# single fit function 
def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    # train the model 
    print ("Accuracy on training set:")
    print (clf.score(X_train, y_train))
    print ("Accuracy on testing set:")
    print (clf.score(X_test, y_test))

    y_pred = clf.predict(X_test)

    print ("Classification Report:")
    print (classification_report(y_test, y_pred))
    print ("Confusion Matrix:")
    print (confusion_matrix(y_test, y_pred))



# cv method to train the model
def evaluate_CV(clf, X, y, K):
    # create a k-fold cross validation iterator
    cv = KFold(K, shuffle=True, random_state=10)
    # essential parameters:
    # (1) n_splits: number of folds
    # (2) shuffle: whether shuffle the data
    # (3) random_state: random seed
    
    # score method of the estimator (accuracy)
    scores = cross_val_score(clf, X, y, cv=cv)
    # essential parameters:
    # (1) estimator: model object
    # (2) X feature
    # (3) Y label
    # (3) cv whehter to use cv method     
    print (scores)
    print ("Mean score: {0:.3f} (+/-{1:.3f})".format(np.mean(scores), np.std(scores))) 

# grdisearch to find the optimal hyper parameters
def parameter_adjuest(parameter_list,X_train, y_train):
    svc = SVC()
    clf = GridSearchCV(svc, parameter_list)
    clf.fit(X_train, y_train)
    print(clf.best_estimator_)
    print(clf.best_params_)
    print(clf.best_score_)
# gridsearch is timeconsuming, for this case it taks 4 mins, but accuarcy increases from 0.913 to 0.96


# this is to label photos with glasses to 1 and without glasses to 0 to achieve a binary classification
def create_label(num_sample,segments):
    # create a new y array of target size initialized with zeros
    y = np.zeros(num_sample)
    # put 1 in the specified segments
    for (start, end) in segments:
        y[start:end + 1] = 1
    return y


# main function     
def main():
    face_data = fetch_olivetti_faces() # face dataset from sklearn
    print(face_data.DESCR) # get description of data set
    print(face_data.keys()) # get keys of data set
    print(face_data.images.shape) 
    # 400 images, each pixel is 64*64
    print(face_data.data.shape)
    # change images (2D) into features, each vector has 4096 features 
    print(np.max(face_data.data))
    print(np.min(face_data.data))  
    # All the images pixels here have been compressed to 0 to 1

    print_faces(face_data.images, face_data.target, 400)
    
    svc_1 = SVC(kernel='linear')  # create a new instance
    # SVC is a multi classifier model, it has several essential parameters:
    # (1) C Regularization parameter
    # (2) Kernel: Kernel function: linear, poly, sigmoid, rbf and precomputed
    # (3) degree: polyinominal kernel
    # (4) gamma: for poly, rbf, sigmoid
    # (5) coef0: used for poly and sigmoid
    print(svc_1)
    
    
    X_train, X_test, y_train, y_test = train_test_split(
            face_data.data, face_data.target, test_size=0.25, random_state=10)
    # a method to split train and test data set
    
    evaluate_CV(svc_1, X_train, y_train, 5)
    # use cv method to evaluate
    
    train_and_evaluate(svc_1, X_train, X_test, y_train, y_test)
    # use single training to evaluate
    
           
    parameter = {'kernel':('linear','poly','rbf'), 'C':[1,5,10], 
                 'degree':[1,4,7],'coef0':[0,5,10]}
    parameter_adjuest(parameter,X_train, y_train)
    # gridsearch to find optimal hyper parameters
    

    # second task, to classify photos with glasses or without glasses
    glasses = [
        (10, 19), (30, 32), (37, 38), (50, 59), (63, 64),
        (69, 69), (120, 121), (124, 129), (130, 139), (160, 161),
        (164, 169), (180, 182), (185, 185), (189, 189), (190, 192),
        (194, 194), (196, 199), (260, 269), (270, 279), (300, 309),
        (330, 339), (358, 359), (360, 369)
    ]
    num_samples = face_data.target.shape[0]
    target_glasses = create_label(num_samples, glasses)
    # get the renewed label vector
    
    svc_2 = SVC(kernel='linear')
    X_train, X_test, y_train, y_test = train_test_split(
            face_data.data, target_glasses, test_size=0.25, random_state=10)
    evaluate_CV(svc_2, X_train, y_train, 5)
    train_and_evaluate(svc_2, X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    t_start = time.time()
    main()
    t_end = time.time()
    t_all = t_end - t_start
    print ('Toal running time: {:.2f} mins' .format(t_all / 60.0))
    # calculate total running time (unit: mins)




