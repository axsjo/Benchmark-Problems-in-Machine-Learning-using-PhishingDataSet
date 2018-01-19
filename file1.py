import scipy as sc
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.grid_search import GridSearchCV
import numpy as np
import csv
import pandas as pd

# Load data
from sklearn.utils import safe_mask

data = pd.read_csv(r"dataset_phishing_with_attributes.csv",nrows=1300)
X = pd.read_csv(r"dataset_phishing_with_attributes.csv",nrows=1300,usecols=[*range(0,9)])
Y = pd.read_csv(r"dataset_phishing_with_attributes.csv",nrows=1300,usecols=[*range(9,10)])

# 10 fold CV
kf = KFold(n_splits = 10, shuffle = True, random_state = 5)
X_train, X_test, y_train, y_test= train_test_split(X, Y,test_size=0.20, random_state=42)
X = X.values
Y = Y.values
i = 1

for train, test in kf.split(X):
    print("Current Iteration",i)
    i=i+1

    X_train, X_test = X[train], X[test]
    y_train, y_test = Y[train], Y[test]
    # Normalizing the Sets
    scaler = preprocessing.StandardScaler().fit(X_train)
    scaler.transform(X_train)
    scaler.transform(X_test)

    # Naive Bayes No Calibration
    naiveBayesModel = GaussianNB()
    naiveBayesModel.fit(X_train, y_train)
    y_pred_naiveBayesModel = naiveBayesModel.predict(X_test)
    f1_naiveBayesModel = f1_score(y_test, y_pred_naiveBayesModel, average='macro')
    print('F1 score for Naive Bayes ', f1_naiveBayesModel)

    # Decision Tree
    decisionTreeModel =  tree.DecisionTreeClassifier()
    decisionTreeModel.fit(X_train, y_train)
    y_pred_decision_tree = decisionTreeModel.predict(X_test)
    f1_decision_tree  = f1_score(y_test, y_pred_decision_tree, average='macro')
    print('F1 score for Decision Tree ', f1_decision_tree)

    # Random Forest
    RfModel = RandomForestClassifier(n_estimators=20)
    RfModel.fit(X_train, y_train)
    y_pred_Rf = RfModel.predict(X_test)
    f1_rf = f1_score(y_test, y_pred_Rf, average='macro')
    print('F1 score for Random Forest',f1_rf)

    # Support Vector Machine
    #svm_parameters = [{'kernel': ['rbf'], 'C': [ 1,10,100,1000]}]
    #SvmModel = GridSearchCV(svm.SVC(), svm_parameters, cv=3)
    svmModel = svm.SVC()
    svmModel.fit(X_train, y_train)
    y_pred_svm = svmModel.predict(X_test)
    f1_svm = f1_score(y_test, y_pred_svm, average='macro')
    print('F1 score for Support Vector Machine', f1_svm)

    # Neural Network
    NNmodel = MLPClassifier(hidden_layer_sizes=(2, 10));
    NNmodel.fit(X_train, y_train)
    y_pred_nn = NNmodel.predict(X_test)
    f1_nn = f1_score(y_test, y_pred_nn, average='macro')
    print('F1 score for NN',f1_nn)
