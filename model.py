import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report



#Import data
filePath = "/home/ahk23009/Law/data"
os.chdir(filePath)
csvFile = "motionStrike_TVcodes_data.tsv"
dataSet = pd.read_csv(csvFile)
print("Data:\n", dataSet.head())


#Filtering out our data:

dataSet.dropna(subset=[], inplace=True)
y = dataSet["MotionResultCode"] # binary 1 or 0, 1 is granted 0 is denyed 





