import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier, RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt



#Import data
filePath = "/home/ahk23009/Law/data/PaperData"
os.chdir(filePath)
trainData = "train.csv"

trainSet = pd.read_csv(trainData)
print("Data:\n", trainSet.head())

testData = 'test.csv'
testSet = pd.read_csv(testData)

#Filtering out our data:

trainSet.dropna(subset=[], inplace=True)
x_train = None 
y_train = trainSet["MotionResultCode"] # binary 1 or 0, 1 is granted 0 is denyed 

x_test =  None 
y_test = None



# Testing our Models: 

#adaBoost 
ada = AdaBoostClassifier()
dTree = DecisionTreeClassifier()
rForest= RandomForestRegressor()
gBoost = GradientBoostingRegressor()

models = [ada,dTree,rForest, gBoost]
predict= 
for model in models:
    model.fit()

xgB = xgb.XGBClassifier()


ada.fit(x_train, y_train)
adaPrediction = ada.predict(x_test)
r2 = r2_score(y_test,adaPrediction)


