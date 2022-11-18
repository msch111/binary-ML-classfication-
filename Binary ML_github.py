import os

import numpy as np
import pandas as pd

import scipy.stats as stats
import random
import copy

from matplotlib import pyplot as plt, pyplot, offsetbox
from numpy import interp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, \
    roc_curve, auc
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns

plt.rcParams['font.family'] = 'Times New Roman'
pyplot.rcParams['font.family'] = 'Times New Roman'
plt.rc('font', size=12)
plt.rc('axes', labelsize=14)
plt.rc('legend', fontsize=13)
plt.rc('figure', titlesize=16)

# Read OF data
folder1 = "/File Path/"
subdir_names1=os.listdir(folder1)
print(subdir_names1)


folder2 = "/File Path/"
subdir_names2=os.listdir(folder2)
print(subdir_names2)


label=['free', 'harmonic']

# DataFrame
df_integ_dataAll=pd.DataFrame()
df_integ_data1=pd.DataFrame()
df_integ_data2=pd.DataFrame()

for file_name1 in subdir_names1:
    data1=pd.read_csv(folder1+"\\"+file_name1, sep="\t", header=None, engine='python', encoding='cp949')
    df_integ_data1=pd.concat([df_integ_data1, data1])

for file_name2 in subdir_names2:
    data2=pd.read_csv(folder2+"\\"+file_name2, sep="\t", header=None, engine='python', encoding='cp949')
    df_integ_data2=pd.concat([df_integ_data2, data2])

print("\n=========== clf assignment ==============\n")
df_integ_data1.insert(0, 'clf', 0)
df_integ_data2.insert(0, 'clf', 1)


print("\n=========== Data Integration ==============\n")
df_integ_dataAll=pd.concat([df_integ_dataAll, df_integ_data1, df_integ_data2])

df_integ_dataAll.columns = ['clf', 'systolic']
print(df_integ_dataAll)

array = df_integ_dataAll.values
X = array[:,1:]
y = array[:,0]
print(y)
print(X)
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.2, random_state=231)

# ###### single indicator
df_X_train = pd.DataFrame(X_train)
df_X_train.columns = ['systolic']

# Spot Check Algorithms # Performance of Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
models.append(('DTC', DecisionTreeClassifier()))
models.append(('RFC', RandomForestClassifier()))
# models.append(('NB', GaussianNB()))
models.append(('linear SVM', SVC(kernel='linear')))

# Evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=231)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

print('\n\n5-fold')
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=231)


##### 1. Logistic Regression
print('\n\n5-fold: LogisticRegression')
model1 = LogisticRegression(solver='liblinear', multi_class='auto')
tprs1 = []
aucs1 = []
mean_fpr1 = np.linspace(0, 1, 100)

# Make predictions on training dataset(k-fold)
for train, test in kfold.split(X_train, Y_train):
    model1.fit(X_train[train], Y_train[train])
    fpr, tpr, t = roc_curve(Y_train[test], model1.decision_function(X_train[test]))
    tprs1.append(interp(mean_fpr1, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs1.append(roc_auc)
    pred2 = model1.predict(X_train[test])
    print(confusion_matrix(Y_train[test], pred2))

mean_tpr1 = np.mean(tprs1, axis=0)
mean_tpr1[0] = 0.0
mean_tpr1[-1] = 1.0
std_tpr1 = np.std(tprs1, axis=0)
mean_auc1 = auc(mean_fpr1, mean_tpr1)
std_auc1 = np.std(aucs1)

tprs1_upper = np.minimum(mean_tpr1 + 0.5*std_tpr1, 1)
tprs1_lower = np.maximum(mean_tpr1 - 0.5*std_tpr1, 0)

print("mean aucs")
print(np.mean(aucs1))
print("std aucs")
print(np.std(aucs1))

print('\nvalidation: LogisticRegression')
# Make predictions on validation dataset
model1.fit(X_train, Y_train)
predictions1 = model1.predict(X_validation)

# print(accuracy_score(Y_validation, predictions1))
# print(precision_score(Y_validation, predictions1))
# print(recall_score(Y_validation, predictions1))
print(confusion_matrix(Y_validation, predictions1))
# print(classification_report(Y_validation, predictions1))
fpr1, tpr1, thresholds1 = roc_curve(Y_validation, model1.decision_function(X_validation))
print("Logistic Regression auc:", auc(fpr1, tpr1))


##### 2. Linear Discriminant Analysis
print('\n\n5-fold: Linear Discriminant Analysis')
model2 = LinearDiscriminantAnalysis(n_components=1)
tprs2 = []
aucs2 = []
mean_fpr2 = np.linspace(0, 1, 100)

# Make predictions on training dataset(k-fold)
for train, test in kfold.split(X_train, Y_train):
    model2.fit(X_train[train], Y_train[train])
    fpr, tpr, t = roc_curve(Y_train[test], model2.decision_function(X_train[test]))
    tprs2.append(interp(mean_fpr2, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs2.append(roc_auc)
    pred2 = model2.predict(X_train[test])
    print(confusion_matrix(Y_train[test], pred2))

mean_tpr2 = np.mean(tprs2, axis=0)
mean_tpr2[0] = 0.0
mean_tpr2[-1] = 1.0
std_tpr2 = np.std(tprs2, axis=0)
mean_auc2 = auc(mean_fpr2, mean_tpr2)

std_auc2 = np.std(aucs2)

tprs2_upper = np.minimum(mean_tpr2 + 0.5*std_tpr2, 1)
tprs2_lower = np.maximum(mean_tpr2 - 0.5*std_tpr2, 0)

print("mean aucs")
print(np.mean(aucs2))
print("std aucs")
print(np.std(aucs2))

print('\nvalidation: Linear Discriminant Analysis')
# Make predictions on validation dataset
model2.fit(X_train, Y_train)
predictions2 = model2.predict(X_validation)

print(confusion_matrix(Y_validation, predictions2))
fpr2, tpr2, thresholds2 = roc_curve(Y_validation, model2.decision_function(X_validation))
print("Linear Discriminant Analysis auc:", auc(fpr2, tpr2))


##### 3. Decision Trees
print('\n\n5-fold: Decision Trees')
model3 = DecisionTreeClassifier()
tprs3 = []
aucs3 = []
mean_fpr3 = np.linspace(0, 1, 100)

# Make predictions on training dataset(k-fold)
for train, test in kfold.split(X_train, Y_train):
    pred = model3.fit(X_train[train], Y_train[train]).predict_proba(X_train[test])
    fpr, tpr, t = roc_curve(Y_train[test], pred[:, 1])
    tprs3.append(interp(mean_fpr3, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs3.append(roc_auc)
    pred2 = model3.predict(X_train[test])
    print(confusion_matrix(Y_train[test], pred2))

mean_tpr3 = np.mean(tprs3, axis=0)
mean_tpr3[0] = 0.0
mean_tpr3[-1] = 1.0
std_tpr3 = np.std(tprs3, axis=0)
mean_auc3 = auc(mean_fpr3, mean_tpr3)
std_auc3 = np.std(aucs3)

tprs3_upper = np.minimum(mean_tpr3 + 0.5*std_tpr3, 1)
tprs3_lower = np.maximum(mean_tpr3 - 0.5*std_tpr3, 0)

print("mean aucs")
print(np.mean(aucs3))
print("std aucs")
print(np.std(aucs3))

print('\nvalidation: Decision Trees')
# Make predictions on validation dataset
model3.fit(X_train, Y_train)
predictions3 = model3.predict(X_validation)

print(confusion_matrix(Y_validation, predictions3))
fpr3, tpr3, thresholds3 = roc_curve(Y_validation, model3.fit(X_train, Y_train).predict_proba(X_validation)[:, 1])
print("Decision Trees auc:", auc(fpr3, tpr3))


##### 4. Random Forest
print('\n\n5-fold: Random Forest')
model4 = RandomForestClassifier()
tprs4 = []
aucs4 = []
mean_fpr4 = np.linspace(0, 1, 100)

# Make predictions on training dataset(k-fold)
for train, test in kfold.split(X_train, Y_train):
    pred = model4.fit(X_train[train], Y_train[train]).predict_proba(X_train[test])
    fpr, tpr, t = roc_curve(Y_train[test], pred[:, 1])
    tprs4.append(interp(mean_fpr4, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs4.append(roc_auc)
    pred2 = model4.predict(X_train[test])
    print(confusion_matrix(Y_train[test], pred2))

mean_tpr4 = np.mean(tprs4, axis=0)
mean_tpr4[0] = 0.0
mean_tpr4[-1] = 1.0
std_tpr4 = np.std(tprs4, axis=0)
mean_auc4 = auc(mean_fpr4, mean_tpr4)
std_auc4 = np.std(aucs4)

tprs4_upper = np.minimum(mean_tpr4 + 0.5*std_tpr4, 1)
tprs4_lower = np.maximum(mean_tpr4 - 0.5*std_tpr4, 0)

print("mean aucs")
print(np.mean(aucs4))
print("std aucs")
print(np.std(aucs4))

print('\nvalidation: Random Forest')
# Make predictions on validation dataset
model4.fit(X_train, Y_train)
predictions4 = model4.predict(X_validation)

print(confusion_matrix(Y_validation, predictions4))
fpr4, tpr4, thresholds4 = roc_curve(Y_validation, model4.fit(X_train, Y_train).predict_proba(X_validation)[:, 1])
print("Random Forest auc:", auc(fpr4, tpr4))


##### 5. Linear Support Vector Machine
print('\n\n5-fold: Linear Support Vector Machine')
model5 = SVC(kernel='linear')
tprs5 = []
aucs5 = []
mean_fpr5 = np.linspace(0, 1, 100)

# Make predictions on training dataset(k-fold)
for train, test in kfold.split(X_train, Y_train):
    model5.fit(X_train[train], Y_train[train])
    fpr, tpr, t = roc_curve(Y_train[test], model5.decision_function(X_train[test]))
    tprs5.append(interp(mean_fpr5, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs5.append(roc_auc)
    pred2 = model5.predict(X_train[test])
    print(confusion_matrix(Y_train[test], pred2))

mean_tpr5 = np.mean(tprs5, axis=0)
mean_tpr5[0] = 0.0
mean_tpr5[-1] = 1.0
std_tpr5 = np.std(tprs5, axis=0)
mean_auc5 = auc(mean_fpr5, mean_tpr5)
std_auc5 = np.std(aucs5)

tprs5_upper = np.minimum(mean_tpr5 + 0.5*std_tpr5, 1)
tprs5_lower = np.maximum(mean_tpr5 - 0.5*std_tpr5, 0)

print("mean aucs")
print(np.mean(aucs5))
print("std aucs")
print(np.std(aucs5))

print('\nvalidation: Linear Support Vector Machine')
# Make predictions on validation dataset
model5.fit(X_train, Y_train)
predictions5 = model5.predict(X_validation)

print(confusion_matrix(Y_validation, predictions5))
fpr5, tpr5, thresholds5 = roc_curve(Y_validation, model5.decision_function(X_validation))
print("Linear Support Vector Machine auc:", auc(fpr5, tpr5))

# ML - 5-fold cross validation
plt.plot(mean_fpr1, mean_tpr1, '-', lw=2, c='k', label='Logistic Regression (AUC = %0.2f)' % (mean_auc1))
plt.plot(mean_fpr2, mean_tpr2, '--', lw=2, c='b', label='Linear Discriminant Analysis (AUC = %0.2f)' % (mean_auc2))
plt.plot(mean_fpr3, mean_tpr3, ls=(0,(3,1,1,1)), lw=2, c='g', label='Decision Trees (AUC = %0.2f)' % (mean_auc3))
plt.plot(mean_fpr4, mean_tpr4, ':', lw=2, c='indigo', label='Random Forest (AUC = %0.2f)' % (mean_auc4))
plt.plot(mean_fpr5, mean_tpr5, '-.', lw=2, c='orange', label='Linear Support Vector Machine (AUC = %0.2f)' % (mean_auc5))
plt.plot([0, 1], [0, 1], '--', lw=2.5, c='r', label='Chance Rate')
plt.legend(loc='center right')
plt.legend(fontsize='10')


plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristics Curves')
plt.suptitle('Free and harmonic respiration condition', size = 16)
plt.show()


# ML - Test set validation
plt.plot(fpr1, tpr1, '-', lw=2, c='k', label='Logistic Regression (AUC = %0.2f)' % (auc(fpr1, tpr1)))
plt.plot(fpr2, tpr2, '--', lw=2, c='b', label='Linear Discriminant Analysis (AUC = %0.2f)' % (auc(fpr2, tpr2)))
plt.plot(fpr3, tpr3, ls=(0,(3,1,1,1)), lw=2, c='g', label='Decision Trees (AUC = %0.2f)' % (auc(fpr3, tpr3)))
plt.plot(fpr4, tpr4, ':', lw=2, c='indigo', label='Random Forest (AUC = %0.2f)' % (auc(fpr4, tpr4)))
plt.plot(fpr5, tpr5, '-.', lw=2, c='orange', label='Linear Support Vector Machine (AUC = %0.2f)' % (auc(fpr5, tpr5)))
plt.plot([0, 1], [0, 1], '--', lw=2.5, c='r', label='Chance Rate')
plt.legend(loc='center right')
plt.legend(fontsize='10')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristics Curves')
plt.suptitle('Free and harmonic respiration condition', size = 16)
plt.show()