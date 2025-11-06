import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier 
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures as poly


#Import data, set up data
filePath = "/home/ahk23009/Law/data/PaperData"
os.chdir(filePath)

#training data set up 
trainData = "train.csv"
trainSet = pd.read_csv(trainData)
trainSet = trainSet[["CaseLocation", "CaseMajorCode", "MotionJurisNumber", "MotionResultCode"]]


    #compute entropy for training
caseCounts = trainSet.groupby(["MotionJurisNumber", "CaseMajorCode"]).size().unstack(fill_value=0)
alpha = 1
smoothed = caseCounts + alpha
prob = smoothed.div(smoothed.sum(axis=1), axis=0)

entropy = -np.sum(prob * np.log(prob), axis=1)
trainSet = trainSet.merge(entropy.rename("AttorneyEntropy"), left_on="MotionJurisNumber", right_index=True)


# test data set up
testData = 'test.csv'
testSet = pd.read_csv(testData)
testSet = testSet[["CaseLocation", "CaseMajorCode", "MotionJurisNumber", "MotionResultCode"]]

    #entropy:
caseCountsT = testSet.groupby(["MotionJurisNumber", "CaseMajorCode"]).size().unstack(fill_value=0)
smoothedT = caseCountsT + alpha
probT = smoothedT.div(smoothedT.sum(axis=1), axis=0)
entropyT = -np.sum(probT * np.log(probT), axis=1)
testSet = testSet.merge(entropyT.rename("AttorneyEntropy"), left_on="MotionJurisNumber", right_index=True)


#label encoding for the classifiers
labelCols = ["CaseLocation", "CaseMajorCode", "MotionJurisNumber"]
for col in labelCols:
    le = LabelEncoder()
    combined = pd.concat([trainSet[col], testSet[col]], axis=0).astype(str)
    le.fit(combined)
    trainSet[col] = le.transform(trainSet[col].astype(str))
    testSet[col] = le.transform(testSet[col].astype(str))




x_train = trainSet[["CaseLocation", "CaseMajorCode", "MotionJurisNumber", "AttorneyEntropy"]]
y_train = trainSet["MotionResultCode"].map({"GR": 1, "DN": 0})  # binary 1 or 0, GR is granted DN is denied

x_test = testSet[["CaseLocation", "CaseMajorCode", "MotionJurisNumber", "AttorneyEntropy"]]
y_test = testSet["MotionResultCode"].map({"GR": 1, "DN": 0})


#Training our Models: 
 
ada = AdaBoostClassifier()
dTree = DecisionTreeClassifier()
rForest= RandomForestClassifier()
gBoost = GradientBoostingClassifier()

models = [ada,dTree,rForest,gBoost]

for model in models:
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test,y_pred)
    print(f"{model.__class__.__name__} Was {acc:.3f}% accurate")

xgB = xgb.XGBClassifier()


for model in models:





