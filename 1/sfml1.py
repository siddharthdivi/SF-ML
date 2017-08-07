import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn import tree,svm,metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from mlxtend.evaluate import lift_score

#df is a Pandas Data Frame.
df = pd.read_csv("/media/siddharth/New Volume/Github/SF ML/1/bank-additional/bank-additional-full.csv",sep=";")
#Tuple representation of the dimensions of df.
df.shape
#GIves the name of the columns alone
df.columns
#Getting a feel for the data.
df.describe(include="all")
#Mentions the x-axis and y-axis information.
df.axes
#Returns the datatype of each atribute column.
df.dtypes
#Returns the numpy representation of the NDFrame.
df.values

        
#PERFORMING FEATURE SELECTION.
from sklearn.model_selection import cross_val_score
from mlxtend.evaluate import lift_score

#Selecting only those attributes which are in the final list of selected attributes.
vector = ["housing","month","duration","campaign","emp.var.rate","cons.price.idx","cons.conf.idx","euribor3m"]
dfcopy2 = df[vector]

#Performing one-hot encoding on the categorical attributes.
encoder1 = pd.get_dummies(dfcopy2['housing'])
del encoder1['unknown']
encoder1.columns = ['housing_no','housing_yes']

encoder2 = pd.get_dummies(dfcopy2['month'])

del dfcopy2['housing']
del dfcopy2['month']


#Concatenating all the data frames.
dfcopy2 = pd.concat([dfcopy2,encoder1],axis=1,join='inner')
dfcopy2 = pd.concat([dfcopy2,encoder2],axis=1,join='inner')

df.replace(('yes','no'),(1,0),inplace=True)

#Splitting the target column into a vector.
Y = df['y']
del df['y']

Ycopy=[]
for i in range(0,len(Y)):
    if(Y[i]==1):
        Ycopy.append(1)
    else:
        Ycopy.append(0)

#Randomly splits the dataset into train and test sets.
X_train,X_test,y_train,y_test = train_test_split(dfcopy2,Y,test_size=0.33)

scaler = preprocessing.StandardScaler().fit(dfcopy2)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

#LogisticRegression
logreg = linear_model.LogisticRegression()
logreg.fit(X_train,y_train)
predictions = logreg.predict(X_test)
np.mean(y_test==predictions)
#Obtained Accuracy : 90.9585%

#Decision Tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
pred = clf.predict(X_test)
np.mean(y_test==pred)
#Obtained Accuracy : 89.0016%

param_grid = [
 {'gamma' : [2**(-15),2**(-1.4),2**(-7.8),2**(-4.2),2**(-0.6),2**(3)]}
]


#SVM
#svmclf = GridSearchCV(svm.SVC(C=3),param_grid)
svmclf = svm.SVC(gamma=0.378929141628,C=3,probability=True)
svmclf.fit(X_train,y_train)
preds = svmclf.predict(X_test)
np.mean(preds == y_test)
#Obtained Accuracy : 91.1277%

#USING ANN : (MULTI LAYER PERCEPTRON.)
nn = MLPClassifier(solver='lbfgs',hidden_layer_sizes=(6,))
nn.fit(X_train,y_train)
nnpred = nn.predict(X_test)

np.mean(nnpred == y_test)
np.mean(cross_val_score(MLPClassifier(solver='lbfgs',hidden_layer_sizes=6),X_train,y_train,scoring='roc_auc',cv=5))
# NN MEAN ROC : 0.9389

lrroc= []
#[0.92364654165675153, 0.92364654165675153, 0.92364654165675153, 0.92364654165675153, 0.92364654165675153, 0.92364654165675153, 0.92364654165675153, 0.92364654165675153, 0.92364654165675153, 0.92364654165675153, 0.92364654165675153, 0.92364654165675153, 0.92364654165675153, 0.92364654165675153, 0.92364654165675153, 0.92364654165675153, 0.92364654165675153, 0.92364654165675153, 0.92364654165675153, 0.92364654165675153]
lrlift=[]
#[5.8444572366667735, 5.8444572366667735, 5.8444572366667735, 5.8444572366667735, 5.8444572366667735, 5.8444572366667735, 5.8444572366667735, 5.8444572366667735, 5.8444572366667735, 5.8444572366667735, 5.8444572366667735, 5.8444572366667735, 5.8444572366667735, 5.8444572366667735, 5.8444572366667735, 5.8444572366667735, 5.8444572366667735, 5.8444572366667735, 5.8444572366667735, 5.8444572366667735]
dtroc=[]
#[0.72727958061417508, 0.72701683591730659, 0.72958782556841106, 0.73276972236611027, 0.73151686250447623, 0.73020152490906687, 0.73131214965169422, 0.72891514675400149, 0.73079971868516647, 0.72853267529095089, 0.7264248609775994, 0.73281686770296384, 0.72895609417721285, 0.72873134247497473, 0.73288904531461352, 0.72973469648294087, 0.73186450348878296, 0.72979142871481772, 0.72951297178990537, 0.73231114772694483]
dtlift=[]
#[4.5174733834511844, 4.5810107454667559, 4.569932696231696, 4.5449067811840829, 4.5451435146305821, 4.5532990605114785, 4.5478747176262289, 4.5811924811253224, 4.5838725027860656, 4.5922372336305664, 4.5122602880687559, 4.5926138165701786, 4.5477938499871984, 4.5728377476366084, 4.5923049517600614, 4.5589468044107146, 4.5672713736829129, 4.570094072459117, 4.4926348508920135, 4.5047710159880348]
svmroc=[]
#[0.88263268289645347, 0.88263268289645347, 0.88263268289645347, 0.88263268289645347, 0.88263268289645347, 0.88263268289645347, 0.88263268289645347, 0.88263268289645347, 0.88263268289645347, 0.88263268289645347, 0.88263268289645347, 0.88263268289645347, 0.88263268289645347, 0.88263268289645347, 0.88263268289645347, 0.88263268289645347, 0.88263268289645347, 0.88263268289645347, 0.88263268289645347, 0.88263268289645347]
svmlift=[]
#[5.8584786818546366, 5.8584786818546366, 5.8584786818546366, 5.8584786818546366, 5.8584786818546366, 5.8584786818546366, 5.8584786818546366, 5.8584786818546366, 5.8584786818546366, 5.8584786818546366, 5.8584786818546366, 5.8584786818546366, 5.8584786818546366, 5.8584786818546366, 5.8584786818546366, 5.8584786818546366, 5.8584786818546366, 5.8584786818546366, 5.8584786818546366, 5.8584786818546366]
nnroc=[]
#[0.94044089933919284, 0.94020629663908706, 0.94153217736353023, 0.94141669669910433, 0.94030537729740193, 0.94106338219364394, 0.94054414471374981, 0.9418599578999608, 0.93852472081745586, 0.93927745801890816, 0.93997267919385086, 0.94112049046897894, 0.94228469226077505, 0.94069177142271165, 0.94113649504758512, 0.94058691214918433, 0.94041712376758435, 0.93973065148586188, 0.94155086753491946, 0.94031552986476874]
nnlift=[]
#[5.809329018865621, 5.6958378536618968, 5.750455476916188, 5.581911522947423, 5.7258475367686508, 5.7478546377136039, 5.5005274763345575, 5.7197287482838757, 5.4862674468853312, 5.5199691067830763, 5.7183026858487525, 5.470425196689539, 5.4086289327370336, 5.5650052384150435, 5.6026265694887041, 5.6237386403244036, 5.702143513662123, 5.8500736668384699, 5.7962898594623038, 5.6389277381207892]
#R=20

#Performing holdout with R distinct runs.
for i in range(0,20):
  #Entire thing to be put in a loop.
  #LogisticRegression
  print ("ITERATION %d" % i)
  #Randomly splits the dataset into train and test sets.
  X_train,X_test,y_train,y_test = train_test_split(dfcopy2,Y,test_size=0.33)
  logreg = linear_model.LogisticRegression()
  logreg.fit(X_train,y_train)
  predictions = logreg.predict(X_test)
  np.mean(y_test==predictions)
  #Obtained Accuracy : 90.44361 (with 'duration')
  lrroc.append(np.mean(cross_val_score(linear_model.LogisticRegression(),X_train,y_train,scoring='roc_auc',cv=5)))
  #roc_auc : 0.9286
  lrlift.append(lift_score(y_test,predictions))
  #ALIFT : 5.6524
  #Decision Tree
  clf = tree.DecisionTreeClassifier()
  clf = clf.fit(X_train,y_train)
  pred = clf.predict(X_test)
  np.mean(y_test==pred)
  #Obtained Accuracy : 88.39844% (with 'duration')
  #Calculating the roc_auc scores.
  dtroc.append(np.mean(cross_val_score(tree.DecisionTreeClassifier(),X_train,y_train,scoring='roc_auc',cv=5)))
  #roc_auc : 0.7157
  dtlift.append(lift_score(y_test,pred))
  #ALIFT : 4.3097
  #SVM
  #svmclf = GridSearchCV(svm.SVC(C=3),param_grid)
  svmclf = svm.SVC(gamma=0.378929141628,C=3,probability=True)
  svmclf.fit(X_train,y_train)
  preds = svmclf.predict(X_test)
  np.mean(preds == y_test)
  #Obtained Accuracy : 90.6716%
  #Calculating the roc_auc scores.
  svmroc.append(np.mean(cross_val_score(svm.SVC(C=3,gamma=0.378929141628),X_train,y_train,scoring='roc_auc',cv=5)))
  #roc_auc : 0.8740
  svmlift.append(lift_score(y_test,preds))
  #ALIFT : 5.6052
  #NN
  #USING ANN : (MULTI LAYER PERCEPTRON.)
  nn = MLPClassifier(solver='lbfgs',hidden_layer_sizes=(6,))
  nn.fit(X_train,y_train)
  nnpred = nn.predict(X_test)
  np.mean(nnpred == y_test)
  #roc_auc
  nnroc.append(np.mean(cross_val_score(MLPClassifier(solver='lbfgs',hidden_layer_sizes=(6,)),X_train,y_train,scoring='roc_auc',cv=5)))
  #ALIFT
  nnlift.append(lift_score(y_test,nnpred))



# LR MEAN ROC : 0.9236
# LR MEAN LIFT : 5.8444 (using MLEXTEND)

# DT MEAN ROC : 0.7300
# DT MEAN LIFT : 4.5564 (using MLEXTEND)

# SVM MEAN ROC : 0.8826
# SVM MEAN LIFT : 5.8584 (using MLEXTEND)

# NN MEAN ROC : 0.9406
# NN MEAN LIFT : 5.6456 (using MLEXTEND)

#MODEL PERSISTENCE.
#Saving the model using joblib.
from sklearn.externals import joblib
joblib.dump(logreg,'logreg.pkl')
joblib.dump(clf,'clf.pkl')
joblib.dump(svmclf,'svmclf.pkl')
joblib.dump(nn,'nn.pkl')

#TO load back the pickled model:
#logreg = joblib.load('logreg.pkl')
#clf = joblib.load('clf.pkl')
#svmclf = joblib.load('svmclf.pkl')
#nn = joblib.load('nn.pkl')


#ROLLING WINDOW PHASE.
#No. of updates : U=L/K
#L = Size of the dataset.
#W = 20000; K = 10;


#PLOTTING OF THE ROC AND LIFT CURVES.

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random

prob_preds1 = svmclf.predict_proba(X_test)

plt.figure(0).clf()

fpr , tpr , threshold = roc_curve(y_test,prob_preds1[:,1],pos_label=1)
roc_auc1 = auc(fpr,tpr)
plt.title('Receiver Operating Characteristics')
plt.plot(fpr,tpr,'b',label= 'SVM = %0.2f' % roc_auc1)
#plt.plot(fpr,tpr)
#plt.plot(fpr1,tpr1)
plt.plot([0,1],[0,1],'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

prob_preds2 = logreg.predict_proba(X_test)

fpr1 , tpr1 , threshold = roc_curve(y_test,prob_preds2[:,1],pos_label=1)
#fpr1 , tpr1 , threshold = roc_curve(ytest,prob_preds1[:,0],pos_label=1)
roc_auc2 = auc(fpr1,tpr1)
plt.title('Receiver Operating Characteristics')
#plt.plot(fpr,tpr,'b',label= 'AUC = %0.2f' % roc_auc)
plt.plot(fpr1,tpr1,label='LR = %0.2f' % roc_auc2)

prob_preds3 = clf.predict_proba(X_test)

fpr3 , tpr3 , threshold = roc_curve(y_test,prob_preds3[:,1],pos_label=1)
#fpr1 , tpr1 , threshold = roc_curve(ytest,prob_preds1[:,0],pos_label=1)
roc_auc3 = auc(fpr3,tpr3)
plt.title('Receiver Operating Characteristics')
#plt.plot(fpr,tpr,'b',label= 'AUC = %0.2f' % roc_auc)

plt.plot(fpr3,tpr3,label='DT = %0.2f' % roc_auc3)

prob_preds4 = nn.predict_proba(X_test)
fpr4 , tpr4 , threshold = roc_curve(y_test,prob_preds4[:,1],pos_label=1)
#fpr1 , tpr1 , threshold = roc_curve(ytest,prob_preds1[:,0],pos_label=1)
roc_auc4 = auc(fpr4,tpr4)
plt.title('Receiver Operating Characteristics')
#plt.plot(fpr,tpr,'b',label= 'AUC = %0.2f' % roc_auc)
plt.plot(fpr4,tpr4,label='NN = %0.3f' % roc_auc4)

plt.legend(loc=0)
plt.show()


#FEATURE IMPORTANCE
from sklearn.ensemble import ExtraTreesClassifier
#Build a forest and compute the feature importances.
forest = ExtraTreesClassifier(n_estimators=250,random_state=0)

forest.fit(X_train,y_train)
importances = forest.feature_importances_

std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
indices = np.argsort(importances)[::-1]

#Print the feature ranking.
print ("Feature Ranking : ")

for f in range(X_train.shape[1]):
     print ("%d. feature %d (%f)" %(f+1,indices[f],importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()

#Generation of LIFT Curves.
        
#Randomly splits the dataset into train and test sets.
X_train,X_test,y_train,y_test = train_test_split(dfcopy2,Y,test_size=0.33)
y_test = pd.DataFrame(y_test,columns=['Actual Class'])
X_test = scaler.fit_transform(X_test)
X_test = pd.DataFrame(X_test)

nnpred1 = nn.predict(X_test)
nnpred = nn.predict_proba(X_test)[:,1]
nnpred = pd.DataFrame(nnpred,columns=['Predicted Class Probability'])
X_test = pd.concat([X_test,nnpred],axis=1)

X_test = pd.concat([X_test,y_test],axis=1)
nnpred1 = pd.DataFrame(nnpred1,columns=['Predicted Class'])
X_test = pd.concat([X_test,nnpred1],axis=1)

X_test['Predicted Class'] = np.where(X_test['Predicted Class']==1,1,-1)
X_test['newcol1'] = np.where(X_test['Predicted Class']==X_test['Actual Class'],1,0)

X_test.sort_values(['Predicted Class Probability'],inplace=True,ascending=False)


outercounter=100
dfcounter=0
cnt=0
capsum=0.0
newdf = pd.DataFrame(columns=['No. of total data','Target data(no. of 1s)','Correctly Classified Samples'])
while outercounter <= 13953:
    
    #cnt1 = X_test[0:outercounter].astype(bool).sum(axis=0)[21]
    #cnt = X_test[:outercounter].astype(bool).sum(axis=0)[19]
    #newdf.loc[dfcounter] = [(outercounter*1),cnt,cnt1]
    cnt=0
    cnt1=0
    dfcounter += 1
    outercounter += 900
    
plt.figure(0).clf    
#plt.scatter(newdf[newdf.columns[0]],(newdf[newdf.columns[2]]*1.0)/newdf[newdf.columns[1]])   
plt.scatter(newdf[newdf.columns[0]],newdf[newdf.columns[1]]) 
plt.scatter(newdf[newdf.columns[0]],newdf[newdf.columns[2]]) 
plt.plot([0,14000],[0,1600],'r--')
#plt.xlim(0,10000)
#plt.ylim(0,3500)
plt.legend()
plt.show()
    
    
#Corrected version of LIFT curve plots.
# (ref. : eric.univ-lyon2.fr )
plt.figure(0).clf()

from __future__ import division
data = nn.predict_proba(X_test)
score = data[:,1]

#Changes being introduced.
#transforming in 0/1 (dummy variables) the Y_test factor.
#pos = pd.get_dummies(y_test[y_test.columns[0]]).as_matrix()
pos = pd.get_dummies(y_test[y_test.columns[0]]).as_matrix()


#Get the second column. 
pos = pos[:,1]

#Number of positive instances.
import numpy as np

#The number of positive instances.
npos = np.sum(pos)

#Indices that would sort according to the score.
index = np.argsort(score)

#Invert the indices, first the indices with the highest score.
index = index[::-1]

#Sort the class membership according to the indices.
sort_pos = pos[index]

#Cumulated sum.
cpos = np.cumsum(sort_pos)

#Recall column.
rappel = (cpos*1.0)/npos

#Number of instances into the test set.
n = y_test.shape[0]

#Target Size.
taille = np.arange(start=0,stop=len(y_test),step=1)

#Target size in percentage.
taille = (taille*1.0)/ n

#import matplotlib.pyplot as plt
#plt.xlim(0,1)
#plt.ylim(0,1)
#plt.scatter(taille,taille,marker='.',color="blue")
#plt.scatter(taille,rappel,marker='.',color="red")
#plt.ylabel("Cumulative Sum.")
#plt.xlabel("Percentage of Dataset.")
#plt.legend(loc=0)
#plt.show()


#LR LR LR LR LR RL LR LR LR
probas = logreg.predict_proba(X_test)
score1 = probas[:,1]

#Changes being introduced.
#transforming in 0/1 (dummy variables) the Y_test factor.
#Get the second column. 
pos1 = pd.get_dummies(y_test[y_test.columns[0]]).as_matrix()

#Get the second column. 
pos1 = pos1[:,1]
#Number of positive instances.
npos1 = np.sum(pos1)

#Indices that would sort according to the score.
index1 = np.argsort(score1)

#Invert the indices, first the indices with the highest score.
index1 = index1[::-1]

#Sort the class membership according to the indices.
sort_pos1 = pos1[index1]

#Cumulated sum.
cpos1 = np.cumsum(sort_pos1)

#Recall column.
rappel1 = (cpos1)/ npos1

import matplotlib.pyplot as plt
plt.xlim(0,1)
plt.ylim(0,1)
plt.scatter(taille,taille,marker='.')
#plt.scatter(taille,rappel,marker='.')
plt.scatter(taille,rappel1,marker='.')
plt.ylabel("Cumulative Sum.")
plt.xlabel("Percentage of Dataset.")
plt.legend(loc=0)
plt.show()

    
    