import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sn
import os
import re
import sys
import warnings

# Evaluation Metrics
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

# Save / Load models
import joblib

### Load the data for Validating the predicted Genre ###

labels = {'Hip':0,'Pop':1,'Vocal':2,'Rhythm':3,'Reggae':4,'Rock':5,'Techno':6}
data_val_genre = pd.read_csv('data/data_genre_validation.csv')
yval_genre, xval_genre = data_val_genre['label'], data_val_genre[data_val_genre.columns[:len(data_val_genre.columns)-1]]
os.system('rm -rf results/')
os.system('mkdir results/')
os.system('rm -rf results/conf_matrices/')
os.system('mkdir results/conf_matrices/')

### Naive Bayes Classifier - Genre ###

naive = joblib.load('models/naive_bayes_genre_trained.pkl')
naive_pred = naive.predict(xval_genre)
mat = confusion_matrix(yval_genre, naive_pred)
temp = pd.DataFrame(mat, index=list(labels.keys()), columns=list(labels.keys()))
fig = plt.figure()
sn.heatmap(temp, annot=True, fmt="d")
plt.xticks(rotation=45)
plt.title('Naive Bayes Model Genre: Confusion matrix')
fig.savefig('results/conf_matrices/genre_naive_bayes_cm.jpg')

### Logistic Regression Classifier - Genre ###

logregcv = joblib.load('models/logreg_genre_trained.pkl')
logreg_pred = logregcv.predict(xval_genre)
mat = confusion_matrix(yval_genre, logreg_pred)
temp = pd.DataFrame(mat, index=list(labels.keys()), columns=list(labels.keys()))
fig = plt.figure()
sn.heatmap(temp, annot=True, fmt="d")
plt.xticks(rotation=45)
plt.title('Logistic Regression Model Genre: Confusion matrix')
fig.savefig('results/conf_matrices/genre_logreg_cm.jpg')

### Support Vector Machine Classifier - Genre ###

svc = joblib.load('models/svm_genre_trained.pkl')
svm_pred = svc.predict(xval_genre)
mat = confusion_matrix(yval_genre, svm_pred)
temp = pd.DataFrame(mat, index=list(labels.keys()), columns=list(labels.keys()))
fig = plt.figure()
sn.heatmap(temp, annot=True, fmt="d")
plt.xticks(rotation=45)
plt.title('SVM Model Genre: Confusion matrix')
fig.savefig('results/conf_matrices/genre_svm_cm.jpg')

### Random Forest Classifier - Genre ###

rf = joblib.load('models/random_forest_genre_trained.pkl')
rf_pred = rf.predict(xval_genre)
mat = confusion_matrix(yval_genre, rf_pred)
temp = pd.DataFrame(mat, index=list(labels.keys()), columns=list(labels.keys()))
fig = plt.figure()
sn.heatmap(temp, annot=True, fmt="d")
plt.xticks(rotation=45)
plt.title('Random Forest Model Genre: Confusion matrix')
fig.savefig('results/conf_matrices/genre_random_forest_cm.jpg')

### Gradient Boosting Tree Classifier - Genre ###

boost = joblib.load('models/xgboost_genre_trained.pkl')
boost_pred = boost.predict(xval_genre)
mat = confusion_matrix(yval_genre, boost_pred)
temp = pd.DataFrame(mat, index=list(labels.keys()), columns=list(labels.keys()))
fig = plt.figure()
sn.heatmap(temp, annot=True, fmt="d")
plt.xticks(rotation=45)
plt.title('Gradient Boosted Tree Model Genre: Confusion matrix')
fig.savefig('results/conf_matrices/genre_xgboost_cm.jpg')

model_accuracy = []
model_accuracy.append(round(accuracy_score(yval_genre,naive_pred),3))
model_accuracy.append(round(accuracy_score(yval_genre,logreg_pred),3))
model_accuracy.append(round(accuracy_score(yval_genre,svm_pred),3))
model_accuracy.append(round(accuracy_score(yval_genre,rf_pred),3))
model_accuracy.append(round(accuracy_score(yval_genre,boost_pred),3))

model_fscore = []
model_fscore.append(round(f1_score(yval_genre,naive_pred,average='weighted'),3))
model_fscore.append(round(f1_score(yval_genre,logreg_pred,average = 'weighted'),3))
model_fscore.append(round(f1_score(yval_genre,svm_pred,average = 'weighted'),3))
model_fscore.append(round(f1_score(yval_genre,rf_pred,average = 'weighted'),3))
model_fscore.append(round(f1_score(yval_genre,boost_pred,average = 'weighted'),3))

res_list_genre = []
res_list_genre.append(model_accuracy)
res_list_genre.append(model_fscore)

eval_metrics_genre = ['Accuracy','F-Score']
models_to_test_genre = ['Naive Bayes','Logistic Regression','Support Vector Machine','Random Forest','XGBoost']
metrics_genre = pd.DataFrame(res_list_genre, columns=models_to_test_genre, index=eval_metrics_genre)
metrics_genre.index.name = 'Metric'
metrics_genre.to_csv('results/eval_metrics_training.csv')

### Load the data for validating Hit Prediction ###

data_val = pd.read_csv('data/data_hit_validation.csv')
yval, xval = data_val['Label'], data_val.iloc[:,:-1]

### K Nearest Neighbors Classifier - Hit Predictor ###

knn = joblib.load('models/knn_hit_trained.pkl')
predict_test = knn.predict(xval)
predict_test = predict_test[:,np.newaxis]
mat = confusion_matrix(yval, predict_test)
temp = pd.DataFrame(mat, index=['Hit Song','Normal'], columns=['Hit Song','Normal'])
fig = plt.figure()
sn.heatmap(temp, annot=True, fmt="d")
plt.title('Confusion Matrix')
fig.savefig('results/conf_matrices/hit_knn_cm.jpg')

### Logistic Regression Classifier - Hit Predictor ###

lreg = joblib.load('models/logreg_hit_trained.pkl')
pred_lreg = lreg.predict(xval)
mat = confusion_matrix(yval, pred_lreg)
temp = pd.DataFrame(mat, index=['Hit Song','Normal'], columns=['Hit Song','Normal'])
fig = plt.figure()
sn.heatmap(temp, annot=True, fmt="d")
plt.title('Confusion Matrix')
fig.savefig('results/conf_matrices/hit_logreg_cm.jpg')

### Support Vector Machine Classifier - Hit Predictor ###

svc = joblib.load('models/svm_hit_trained.pkl')
svm_pred = svc.predict(xval)
mat = confusion_matrix(yval, svm_pred)
temp = pd.DataFrame(mat, index=['Hit Song','Normal'], columns=['Hit Song','Normal'])
fig = plt.figure()
sn.heatmap(temp, annot=True, fmt="d")
plt.title('Confusion Matrix')
fig.savefig('results/conf_matrices/hit_svm_cm.jpg')

### Random Forest Classifier - Hit Predictor ###

rf = joblib.load('models/random_forest_hit_trained.pkl')
rf_pred = rf.predict(xval) 
mat = confusion_matrix(yval, rf_pred)
temp = pd.DataFrame(mat, index=['Hit Song','Normal'], columns=['Hit Song','Normal'])
fig = plt.figure()
sn.heatmap(temp, annot=True, fmt="d")
plt.title('Confusion Matrix')
fig.savefig('results/conf_matrices/hit_random_forest_cm.jpg')

### Gradient Boosting Trees - Hit Predictor ###

boost = joblib.load('models/xgboost_hit_trained.pkl')
boost_pred = boost.predict(xval)
mat = confusion_matrix(yval, boost_pred)
temp = pd.DataFrame(mat, index=['Hit Song','Normal'], columns=['Hit Song','Normal'])
fig = plt.figure()
sn.heatmap(temp, annot=True, fmt="d")
plt.title('Confusion Matrix')
fig.savefig('results/conf_matrices/hit_xgboost_cm.jpg')


model_accuracy = []
model_accuracy.append(round(accuracy_score(yval,predict_test),3))
model_accuracy.append(round(accuracy_score(yval,pred_lreg),3))
model_accuracy.append(round(accuracy_score(yval,svm_pred),3))
model_accuracy.append(round(accuracy_score(yval,rf_pred),3))
model_accuracy.append(round(accuracy_score(yval,boost_pred),3))

model_fscore = []
model_fscore.append(round(f1_score(yval,predict_test,average = None)[1],3))
model_fscore.append(round(f1_score(yval,pred_lreg,average = None)[1],3))
model_fscore.append(round(f1_score(yval,svm_pred,average = None)[1],3))
model_fscore.append(round(f1_score(yval,rf_pred,average = None)[1],3))
model_fscore.append(round(f1_score(yval,boost_pred,average = None)[1],3))

model_precision = []
model_precision.append(round(precision_score(yval,predict_test,average = None)[1],3))
model_precision.append(round(precision_score(yval,pred_lreg,average = None)[1],3))
model_precision.append(round(precision_score(yval,svm_pred,average = None)[1],3))
model_precision.append(round(precision_score(yval,rf_pred,average = None)[1],3))
model_precision.append(round(precision_score(yval,boost_pred,average = None)[1],3))

model_recall = []
model_recall.append(round(recall_score(yval,predict_test,average = None)[1],3))
model_recall.append(round(recall_score(yval,pred_lreg,average = None)[1],3))
model_recall.append(round(recall_score(yval,svm_pred,average = None)[1],3))
model_recall.append(round(recall_score(yval,rf_pred,average = None)[1],3))
model_recall.append(round(recall_score(yval,boost_pred,average = None)[1],3))

res_list = []
res_list.append(model_accuracy)
res_list.append(model_fscore)
res_list.append(model_precision)
res_list.append(model_recall)

eval_metrics_hit = ['Accuracy','F-Score','Precision','Recall']
models_to_test_hit = ['K-Nearest Neighbors','Logistic Regression','Support Vector Machine','Random Forest','XGBoost']
metrics_hit = pd.DataFrame(res_list, columns=models_to_test_hit, index=eval_metrics_hit)
metrics_hit.index.name = 'Metric'
emptydf = pd.DataFrame()
emptydf.to_csv('results/eval_metrics_training.csv', mode='a')
metrics_hit.to_csv('results/eval_metrics_training.csv', mode='a')

print('\nEvaluation Metrics for Genre and Hit Prediction respectively:\n')
os.system('column -t -s "," results/eval_metrics_training.csv')
print('\nModel Selection:\n\nFrom the above results, we can see that the Gradient Boosted Tree model works better for predicting \nthe Genre and the Random Forest model works best to predict if a given song will make it to the billboard or not. \nFurthermore, we want to see how these models perform on new data.\n')

