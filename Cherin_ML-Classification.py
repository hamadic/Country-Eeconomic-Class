# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 12:20:45 2018

@author: Cherin
"""
import os
os.getcwd()
os.chdir(r'C:\Users\Cherin\Desktop\Metro College\MachineLearning')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import libraries and function

#libraries for preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#import performance metric functions
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import precision_score, recall_score

# Import Dataframe
df= pd.read_csv ('SocioEco.csv')

# Preprocessing
# Check if there is nan

df.isnull().sum().sum()

# rearrange columns 
df.columns
new_order = [0,1,2,3,4,6,7,8,9,5]
df1 = df[df.columns[new_order]]
df2=df1.iloc[:,3:]


# take a sampleof 5000 rows
df2= df2.sample(n=5000,replace="False")

# check correlation 
import matplotlib.pyplot as plt

plt.matshow(df2.corr())

import seaborn as sns
corr = df2.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

pd.scatter_matrix(df2, alpha = 0.3, figsize = (14,8), diagonal = 'kde')



#extract independent variables
X = df2.iloc[:,:-1].values

#extract dependent variable
y = df2.iloc[:,-1]

lb= LabelEncoder()
df2["year1"] = lb.fit_transform(df2["year"])
df2["region51"] = lb.fit_transform(df2["region5"])
df2["regionUN1"] = lb.fit_transform(df2[ "regionUN"])
df2["class1"] = lb.fit_transform(df2["class"])

df3=df2.loc[:,["ses", "gdppc","yrseduc", "year1", "region51", "regionUN1", "class1" ]]

#extract independent variables
X = df3.iloc[:,:-1].values

#extract dependent variable
y = df3.iloc[:,-1].values

#label encoding for categorical features

onehotencoder = OneHotEncoder(categorical_features = [3,4,5])
X = onehotencoder.fit_transform(X).toarray()
X1=pd.DataFrame(X)

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#feature scaling for independent variables
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

#Split dataset into train and test samples
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0)

##Create instance of a classifier
from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression()

logreg.fit(X_train, y_train) 

'''LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
verbose=0, warm_start=False)'''


y_pred=logreg.predict(X_test)
y_pred_prob= logreg.predict_proba(X_test)

logreg.score(X_test, y_test) # 91%

# model performance
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import precision_score, recall_score
#See the performace of the classifier(s) using confusion matrix
cm(y_test,y_pred)



#Check classifier's accuracy
logreg.score(X_test,y_test)


from sklearn.metrics import precision_recall_fscore_support as score


precision, recall, fscore, support = score(y_test, y_pred)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

# use multinomial logistic regression
logreg1=LogisticRegression(multi_class= 'multinomial', solver='sag')

logreg1.fit(X_train, y_train) 
y_pred=logreg1.predict(X_test)
y_pred_prob1= logreg1.predict_proba(X_test)
logreg1.score(X_test, y_test) # 96%

#Cross Validation using K-fold "stratified Kfold"
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold()
skf.get_n_splits(X, y) # 3
print(skf)  
for train_index, test_index in skf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

###applying K-fold 
from sklearn.model_selection import cross_val_score
accuricies = cross_val_score(estimator =logreg, X= X_train, y= y_train, cv=10)### accuricies of 10 test sets (K-fold validation)
accuricies.mean() # 0.849
accuricies.std() # 0.108


# Gridsearch  for hyperparameter tuning 
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

params = {
            'alpha':[1,0.1,0.01,0.001,0.0001,0]
        }
logreg1=Ridge()
logreg = GridSearchCV(logreg1,params)
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
logreg.score(X_test,y_test)
logreg.best_params_
logreg.best_score_

#####################################################################################
#KNN Classifier 
#Create instance of a classifier
from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier()

#Train the classifier(s) on your training set
clf.fit(X_train,y_train)
clf.score(X_train, y_train)

#Test the classifier(s) on your training set
y_pred=clf.predict(X_test)
clf.score(X_test, y_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Check Classifier's precision (tp/tp+fp) and recall score (tp/tp+fp)
from sklearn.metrics import precision_score, recall_score
precision=precision_score(y_test,y_pred)
recall=recall_score(y_test,y_pred)

# model performance
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import precision_score, recall_score
#See the performace of the classifier(s) using confusion matrix
cm(y_test,y_pred)

#Check Classifier's accuracy
clf.score(X_test,y_test)

#Cross Validation using K-fold "stratified Kfold"
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold()
skf.get_n_splits(X, y) # 3
print(skf)  
for train_index, test_index in skf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

###applying K-fold 
from sklearn.model_selection import cross_val_score
accuricies = cross_val_score(estimator =clf, X= X_train, y= y_train, cv=10)### accuricies of 10 test sets (K-fold validation)
accuricies.mean() #0.79
accuricies.std() # 0.149


#run Gridsearch 
from sklearn.model_selection import GridSearchCV

knn = KNeighborsClassifier()
k_range = list(range(1, 30))
parameters = {'n_neighbors':[4,5,6,7],
              'leaf_size':[1,3,5],
              'algorithm':['auto', 'kd_tree'],
              'n_jobs':[-1]}

model = GridSearchCV(knn, param_grid=parameters)
model.fit(X_train,y_train.ravel())
model.score(X_train,y_train.ravel())
model.best_score_
model.best_params_


######################################################################################################
# Random Forest
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
RF.fit(X_train, y_train)
RF.score(X_train, y_train)

# Predicting the Test set results
y_pred = RF.predict(X_test)
RF.score(X_test, y_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Cross Validation using K-fold "stratified Kfold"
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold()
skf.get_n_splits(X, y) # 3
print(skf)  
for train_index, test_index in skf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

###applying K-fold 
from sklearn.model_selection import cross_val_score
accuricies = cross_val_score(estimator =RF, X= X_train, y= y_train, cv=10)### accuricies of 10 test sets (K-fold validation)
accuricies.mean() #925
accuricies.std() # 0.057

# run gridsearch
from sklearn.model_selection import GridSearchCV
params = {
            'n_estimators':[1, 10, 100],
            'max_depth':[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]
            
        }
Rf1=RandomForestClassifier()
RF = GridSearchCV(Rf1,params)
RF.fit(X_train,y_train)
RF.best_params_ # mx_depth= 30, n_estimators= 100

RF.best_score_

RF_best = RandomForestClassifier(max_depth= 30, n_estimators= 100)
RF_best.fit(X_train,y_train)
y_pred = RF_best.predict(X_test)
RF_best.score(X_test,y_test)

######################################################################################################

# SVM Model
# Fitting SVM to the Training set
from sklearn.svm import SVC
svm = SVC(kernel = 'linear', random_state = 0)
svm.fit(X_train, y_train)
svm.score(X_train, y_train)
# Predicting the Test set results
y_pred = svm.predict(X_test)
svm.score(X_test, y_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Cross Validation using K-fold "stratified Kfold"
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold()
skf.get_n_splits(X, y) # 3
print(skf)  
for train_index, test_index in skf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

###applying K-fold 
from sklearn.model_selection import cross_val_score
accuricies = cross_val_score(estimator =svm, X= X_train, y= y_train, cv=10)### accuricies of 10 test sets (K-fold validation)
accuricies.mean() #93
accuricies.std() # 0.06

# run gridsearch 
from sklearn.model_selection import GridSearchCV
params = {
            'C':[0.1, 1, 10],
            'kernel':['linear', 'poly', 'rbf'],
            'degree':[2,3,4],
            'gamma':[0.001, 0.01, 0.1],
            'tol':[0.001, 0.01, 0.1]        
        }

from sklearn.svm import SVC
clf = SVC()
svm = GridSearchCV(clf,params)
svm.fit(X_train,y_train)
svm.best_params_
svm.best_score_

