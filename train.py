import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sn
import os
import re
import sys

#Pre-processing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale

# Splitting Data into Train and Test
from sklearn.model_selection import train_test_split

# Models
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegressionCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.multiclass import unique_labels
import xgboost as xgboost
from xgboost import XGBClassifier

# Save / Load models
import joblib

os.system('rm -rf models/')
os.system('mkdir models/')

### Load the Preprocessed data for Genre Prediction ###

labels = {'Hip':0,'Pop':1,'Vocal':2,'Rhythm':3,'Reggae':4,'Rock':5,'Techno':6}
data_train_genre = pd.read_csv('data/data_genre_training.csv')
ytrain_genre, xtrain_genre = data_train_genre['label'], data_train_genre[data_train_genre.columns[:len(data_train_genre.columns)-1]]
print('Loaded Genre Dataset!')

### Naive Bayes Classifier - Genre ###

naive = GaussianNB().fit(xtrain_genre, ytrain_genre)
joblib.dump(naive,'models/naive_bayes_genre_trained.pkl')

### Logistic Regression Classifier - Genre ###

logregcv = LogisticRegressionCV(cv=10, multi_class='multinomial').fit(xtrain_genre, ytrain_genre)
joblib.dump(logregcv,'models/logreg_genre_trained.pkl')

### Support Vector Machine Classifier - Genre ###

C_est = range(100, 1001, 100)
acc_test = []
for i in C_est:
    svc = SVC(C = i, probability = True, class_weight=dict(ytrain_genre.value_counts(normalize = True)))
    svc.fit(xtrain_genre, ytrain_genre)
    svm_pred_test = svc.predict(xval_genre)
    acc_test.append(accuracy_score(yval_genre,svm_pred_test))
C_est_opt = 50 * acc_test.index(np.max(acc_test))

svc = SVC(C = C_est_opt, probability = True, class_weight=dict(ytrain_genre.value_counts(normalize = True)))
svc.fit(xtrain_genre, ytrain_genre)
joblib.dump(svc,'models/svm_genre_trained.pkl')

### Random Forest Classifier - Genre ###

N_est = range(100, 1001, 100)
acc_test = []
for i in N_est:
    rf = RandomForestClassifier(n_estimators=i, min_samples_split=10)
    rf.fit(xtrain_genre,ytrain_genre)
    rf_pred_test = rf.predict(xval_genre)
    acc_test.append(accuracy_score(yval_genre,rf_pred_test))
N_est_opt = 100 * acc_test.index(np.max(acc_test))

rf = RandomForestClassifier(n_estimators=N_est_opt, min_samples_split=10)
rf.fit(xtrain_genre, ytrain_genre)
joblib.dump(rf,'models/random_forest_genre_trained.pkl')

### Gradient Boosting Tree Classifier - Genre ###

N_est = range(100, 1001, 100)
acc_test = []
for i in N_est:
    boost = XGBClassifier(n_estimators=i, max_depth=10, subsample=0.8, num_class = len(labels), objective='multi:softprob')
    boost.fit(xtrain_genre, ytrain_genre)
    boost_pred_test = boost.predict(xval_genre)
    acc_test.append(accuracy_score(yval_genre,boost_pred_test))
N_est_opt = 100 * acc_test.index(np.max(acc_test))

boost = XGBClassifier(n_estimators=N_est_opt, max_depth=10, subsample=0.8, num_class = len(labels), objective='multi:softprob')
boost.fit(xtrain_genre, ytrain_genre)
joblib.dump(boost,'models/xgboost_genre_trained.pkl')

print('Training models for Genre Prediction Complete!')

### Load the Preprocessed data for Hit Prediction ###

data_train = pd.read_csv('data/data_hit_training.csv')
ytrain, xtrain = data_train['Label'], data_train.iloc[:,:-1]
print('Loaded Billboard Hits Dataset!')

### K Nearest Neighbors Classifier - Hit Predictor ###

err_test = []
step_k = 1
k = range(1, 50, step_k)
for i in k:
    knn = KNeighborsClassifier(n_neighbors=i,p=2)        
    knn.fit(xtrain, ytrain)            
    predict_train = knn.predict(xtrain)       
    predict_test = knn.predict(xval)         
    err_test.append(np.mean(predict_test != yval))     

k_opt = 1 + step_k * err_test.index(np.min(err_test))    
print('Optimal K for Test data using Manhattan distance metric is', k_opt)

knn = KNeighborsClassifier(n_neighbors = k_opt, p = 1)   
knn.fit(xtrain, ytrain)
joblib.dump(knn,'models/knn_hit_trained.pkl')

### Logistic Regression Classifier - Hit Predictor ###

lreg = LogisticRegressionCV(cv = 10, solver = 'liblinear', penalty = 'l1',refit=True)
lreg.fit(xtrain,ytrain)
joblib.dump(lreg, 'models/logreg_hit_trained.pkl')

### Support Vector Machine Classifier - Hit Predictor ###

C_est = range(1, 101, 5)
acc_test = []
for i in C_est:
    svc = SVC(C = i, probability = True, class_weight=dict(ytrain.value_counts(normalize = True)))
    svc.fit(xtrain, ytrain)
    svc_pred_test = svc.predict(xval)
    acc_test.append(accuracy_score(yval,svc_pred_test))
C_est_opt = 5 * acc_test.index(np.max(acc_test))

svc = SVC(C = C_est_opt, probability = True, class_weight=dict(ytrain.value_counts(normalize = True)))
svc.fit(xtrain, ytrain)
joblib.dump(svc,'models/svm_hit_trained.pkl')

### Random Forest Classifier - Hit Predictor ###

N_est = range(100, 1001, 25)
acc_test = []
for i in N_est:
    rf = RandomForestClassifier(n_estimators=i, min_samples_split=10)
    rf.fit(xtrain,ytrain)
    rf_pred_test = rf.predict(xval)
    acc_test.append(accuracy_score(yval,rf_pred_test))
N_est_opt = 100 * acc_test.index(np.max(acc_test))

rf = RandomForestClassifier(n_estimators=N_est_opt, min_samples_split=10)
rf.fit(xtrain, ytrain)
joblib.dump(rf, 'models/random_forest_hit_trained.pkl')

### Gradient Boosting Trees - Hit Predictor ###

N_est = range(100, 1001, 100)
acc_test = []
for i in N_est:
    boost = XGBClassifier(n_estimators=i, max_depth=10, subsample=0.9, num_class = 2, objective='multi:softmax')
    boost.fit(xtrain, ytrain)
    boost_pred_test = boost.predict(xval)
    acc_test.append(accuracy_score(yval,boost_pred_test))
N_est_opt = 100 * (acc_test.index(np.max(acc_test))-1)

boost = XGBClassifier(n_estimators=N_est_opt, max_depth=10, subsample=0.9, num_class = 2, objective='multi:softmax')
boost.fit(xtrain, ytrain)
joblib.dump(boost, 'models/xgboost_hit_trained.pkl')

print('Training models for Hit Prediction Complete!')
