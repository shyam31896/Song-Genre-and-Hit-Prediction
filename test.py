import numpy as np
import pandas as pd
import os
import joblib

# Evaluation Metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

### Load the Test set to see the performance of the model in predicting the correct Genre ###

data_test_genre = pd.read_csv('data/data_genre_test.csv')
ytest_genre, xtest_genre = data_test_genre['label'], data_test_genre[data_test_genre.columns[:len(data_test_genre.columns)-1]]

best_model_genre = joblib.load('models/xgboost_genre_trained.pkl')
pred_best_model_genre = best_model_genre.predict(xtest_genre)
test_list_genre = []
eval_metrics_genre = ['Accuracy','F-Score']
test_list_genre.append(round(accuracy_score(ytest_genre,pred_best_model_genre),3))
test_list_genre.append(round(f1_score(ytest_genre,pred_best_model_genre,average='weighted'),3))
metrics_genre = pd.DataFrame(test_list_genre, columns=['XGBoost'], index=eval_metrics_genre)
metrics_genre.index.name = 'Metric'
metrics_genre.to_csv('results/eval_metrics_test.csv')

### Load the Test set to see the performance of the model in the Hit prediction of a Song ###

data_test = pd.read_csv('data/data_hit_test.csv')
ytest, xtest = data_test['Label'], data_test.iloc[:,:-1]

best_model_hit = joblib.load('models/random_forest_hit_trained.pkl')
pred_best_model_hit = best_model_hit.predict(xtest)
test_list_hit = []
test_list_hit.append(round(accuracy_score(ytest,pred_best_model_hit),3))
test_list_hit.append(round(f1_score(ytest,pred_best_model_hit,average='weighted'),3))
test_list_hit.append(round(precision_score(ytest,pred_best_model_hit,average = None)[1],3))
test_list_hit.append(round(recall_score(ytest,pred_best_model_hit,average = None)[1],3))
eval_metrics_hit = ['Accuracy','F-Score','Precision','Recall']
metrics_hit = pd.DataFrame(test_list_hit, columns=['Random Forest'], index=eval_metrics_hit)
metrics_hit.index.name = 'Metric'
emptydf = pd.DataFrame()
emptydf.to_csv('results/eval_metrics_test.csv', mode='a')
metrics_hit.to_csv('results/eval_metrics_test.csv', mode='a')

print('\nEvaluation Metrics for Genre and Hit Prediction on Test data respectively:\n')
os.system('column -t -s "," results/eval_metrics_test.csv')
