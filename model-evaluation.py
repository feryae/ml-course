import numpy as numpy
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

# label = 'Profit'
df_regression = pd.read_csv("https://raw.githubusercontent.com/maxalmina/maru/master/datasets/learning/week3_assignment_regression.csv", index_col=0)
# label = 'PASS'
df_classification = pd.read_csv("https://raw.githubusercontent.com/maxalmina/maru/master/datasets/learning/week3_assignment_classification.csv", index_col=0)

NIM = 1301174532
Nama = "Fery Ardiansyah Effendi"

print("Hi! I'm {} with student ID: {}".format(Nama, NIM))
  
# REGRESSION EVALUATION
# function for measuring error rate using RMSE
# OUTPUT: RMSE of the model
def error_rate(y_pred, y_actual):
  # write your code here
  rmse = numpy.sqrt(numpy.mean((y_actual-y_pred)**2)) 
  return rmse

# CLASSIFICATION EVALUATION
# function for measuring confusion matrix
# OUTPUT: two dimensional array of confusion matrix
def conf_matrix(y_pred, y_actual):
  # write your code here
  TP = numpy.sum((y_pred == "Yes") & (y_actual == "Yes"))
  FP =  numpy.sum((y_pred == "Yes") & (y_actual == "No"))
  FN = numpy.sum((y_pred == "No") & (y_actual == "Yes"))
  TN = numpy.sum((y_pred == "No") & (y_actual == "No"))


  return [[TP, FP], [FN, TN]]

# function for measuring precision, recall, accuracy
# OUTPUT value of precision, recall, accuracy
def class_performance(confusion_matrix):
  # write your code here
  TP = confusion_matrix[0][0]
  FP = confusion_matrix[0][1]
  FN = confusion_matrix[1][0]
  TN = confusion_matrix[1][1]
  
  precision = TP/(TP+FP)
  recall = TP/(TP+FN)
  accuracy = (TP+TN) / (TP+TN+FP+FN)
  
  return precision, recall, accuracy

# function for pick one of two models based on metrics that we used
# OUTPUT: string of a decision which model that we picked
def compare_it(metrics_a, metrics_b, type="regression"):
    if type == "regression":
        if metrics_a**2 < metrics_b**2:
            return "Pick Model A"
        else:
            return "Pick Model B"
    elif type == "classification":
        if metrics_a > metrics_b:
            return "Pick Model A"
        else:
            return "Pick Model B"
      
print("REGRESSION :")
profit = df_regression['Profit']
pred_a = df_regression['predict_a']
pred_b = df_regression['predict_b']
# A
error_a = error_rate(pred_a,profit)
# B
error_b = error_rate(pred_b,profit)
# Result of model evaluation analysis
compare_it(error_a, error_b)

print("CLASSIFICATION:")
# A
confusion_a = conf_matrix(df_classification['PASS'],df_classification['predict_a'])
precision_a, recall_a, accuracy_a = class_performance(confusion_a)
# B
confusion_b = conf_matrix(df_classification['PASS'],df_classification['predict_b'])
precision_b, recall_b, accuracy_b = class_performance(confusion_b)
# Result of model evaluation analysis
compare_it(accuracy_a, accuracy_b, type="classification")
