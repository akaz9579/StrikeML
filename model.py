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

dataSet.dropna(subset=[], inplace=True)
y = dataSet["MotionResultCode"] # binary 1 or 0, 1 is granted 0 is denyed 





