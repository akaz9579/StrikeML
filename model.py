import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import AdaBoostClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier 
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, r2_score
import matplotlib.pyplot as plt



#Import data, set up data
filePath = "/home/ahk23009/Law/data/PaperData"
os.chdir(filePath)

#training data set up 
trainData = "train.csv"
trainSet = pd.read_csv(trainData)
trainSet = trainSet[["CaseLocation", "CaseMajorCode", "MotionJurisNumber", "MotionResultCode"]]


#---------------------compute entropy for training---------------------
caseCounts = trainSet.groupby(["MotionJurisNumber", "CaseMajorCode"]).size().unstack(fill_value=0)
alpha = 1
smoothed = caseCounts + alpha
prob = smoothed.div(smoothed.sum(axis=1), axis=0)

entropy = -np.sum(prob * np.log(prob), axis=1)
trainSet = trainSet.merge(entropy.rename("AttorneyEntropy"), left_on="MotionJurisNumber", right_index=True)
print(trainSet.head())
#---------------------compute entropy for training---------------------

x_train = trainSet["CaseLocation", "CaseMajorCode", "MotionJurisNumber","AttorneyEntropy"]
y_train = trainSet["MotionResultCode"] # binary 1 or 0, 1 is granted 0 is denyed 

#test data set up 
testData = 'test.csv'
testSet = pd.read_csv(testData)
testSet = testSet["CaseLocation", "CaseMajorCode", "MotionJurisNumber", "MotionResultCode"]

#entropy:
caseCountsT = testSet.groupby(["MotionJurisNumber", "CaseMajorCode"]).size().unstack(fill_value=0)
smoothedT = caseCountsT + alpha
probT = smoothedT.div(smoothedT.sum(axis=1), axis=0)
entropyT = -np.sum(probT * np.log(probT), axis=1)
testSet = testSet.merge(entropyT.rename("AttorneyEntropy"), left_on="MotionJurisNumber", right_index=True)
print(testSet.head())


x_test =  None 
y_test = testSet["MotionResultCode"]


# Testing our Models: 

#adaBoost 
ada = AdaBoostClassifier()
dTree = DecisionTreeClassifier()
rForest= RandomForestRegressor()
gBoost = GradientBoostingRegressor()

models = [ada,dTree,rForest, gBoost]
#predict= 
for model in models:
    model.fit()

xgB = xgb.XGBClassifier()


ada.fit(x_train, y_train)
adaPrediction = ada.predict(x_test)
r2 = r2_score(y_test,adaPrediction)


