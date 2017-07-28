import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_excel(io="/home/siddharth/SF ML/2/data.xls",header=1)

df.shape
# (30000,25)

df.head(4)

df.columns

df.dtypes

y = df['default payment next month']

y.shape
#(30000,)

del df['default payment next month']

df.shape 
#(30000,24)

#Check for NULL values.
print (df.isnull().sum())
#NO NULL or missing values present.


#Scaling the dataset.
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(df)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df,y,test_size=0.33)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# 1. SVM Classifier
from sklearn import svm
import numpy as np
svmclf = svm.SVC()
svmclf.fit(X_train,y_train)

svmpred = svmclf.predict(X_test)

#Accuracy : 0.7783 (without scaling)
#Accuracy : 0.8199 (with scaling)
np.mean(svmpred == y_test)

#Prediction errors
#Train error : 0.1741
1 - (1.0 * sum(svmclf.predict(X_train) == y_train))/len(y_train)
#Test error : 0.1800 
1 - (1.0 * sum(svmpred == y_test))/len(y_test)

#Model Persistence.
from sklearn.externals import joblib
joblib.dump(svmclf,'svmclf.pkl')

# 2. LOGISTIC REGRESSION.
from sklearn import linear_model

lr  = linear_model.LogisticRegression()
lr.fit(X_train,y_train)
lrpred = lr.predict(X_test)

#Accuracy : 0.8074
np.mean(lrpred == y_test)


#Prediction errors
#Train error : 0.1888
1 - (1.0 * sum(lr.predict(X_train) == y_train))/len(y_train)
#Test error : 0.1925 
1 - (1.0 * sum(lrpred == y_test))/len(y_test)


# 3. Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnbpred = gnb.fit(X_train,y_train).predict(X_test)

#Accuracy : 0.7185
np.mean(gnbpred == y_test)


#Prediction errors
#Train error : 0.2723
1 - (1.0 * sum(gnb.predict(X_train) == y_train))/len(y_train)
#Test error : 0.2814
1 - (1.0 * sum(gnbpred == y_test))/len(y_test)

# 4. DECISION TREES
from sklearn import tree

dt = tree.DecisionTreeClassifier()
dtpred = dt.fit(X_train,y_train).predict(X_test)

#Accuracy : 0.7237
np.mean(dtpred == y_test)


#Prediction errors
#Train error : 0.0 ?
1 - (1.0 * sum(dt.predict(X_train) == y_train))/len(y_train)
#Test error : 0.2762
1 - (1.0 * sum(dtpred == y_test))/len(y_test)

# 5. ANN
from sklearn.neural_network import MLPClassifier

nn = MLPClassifier(hidden_layer_sizes=(10,2),solver='lbfgs')
nn.fit(X_train,y_train)
nnpred = nn.predict(X_test)


#Accuracy : 0.8161
np.mean(nnpred == y_test)

#Prediction errors
#Train error : 0.1558
1 - (1.0 * sum(nn.predict(X_train) == y_train))/len(y_train)
#Test error : 0.2013
1 - (1.0 * sum(nnpred == y_test))/len(y_test)

# 6. KNN
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=15)
neigh.fit(X_train,y_train)

neighpred = neigh.predict(X_test)

#Accuracy : 0.8105
np.mean(neighpred == y_test)

#Prediction errors
#Train error : 0.1567
1 - (1.0 * sum(neigh.predict(X_train) == y_train))/len(y_train)
#Test error : 0.2075
1 - (1.0 * sum(neighpred == y_test))/len(y_test)


# 7. Discriminant Analysis (DA)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
da = LinearDiscriminantAnalysis()
da.fit(X_train,y_train)
dapred = da.predict(X_test)

#Accuracy : 0.8081
np.mean(dapred == y_test)

#Prediction errors
#Train error : 0.1868
1 - (1.0 * sum(da.predict(X_train) == y_train))/len(y_train)
#Test error : 0.1918
1 - (1.0 * sum(dapred == y_test))/len(y_test)

#Predictive Accuracy of probability by default.

#Sort by predicted probability.
    
#Predictive Probabilities

from __future__ import division
data = nn.predict_proba(X_test)
score = data[:,1]

#Changes being introduced.
#transforming in 0/1 (dummy variables) the Y_test factor.
pos = pd.get_dummies(y_test).as_matrix()

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
taille = np.arange(start=1,stop=9901,step=1)

#Target size in percentage.
#taille = (taille*1.0)/ n
taille = (taille*1.0)/ n

import matplotlib.pyplot as plt
plt.xlim(0,1)
plt.ylim(0,1)
plt.scatter(taille,taille)
plt.scatter(taille,rappel)
plt.show()

ylist = pd.DataFrame(columns = ['0','1'])

count=50
for i in range(0,len(y_test)):
    
    ylist[i] = [count,data[i]]
    count += 1
ylist = ylist.transpose()
ylist.dropna()

iter1=0
ylist1 = pd.DataFrame(columns = ['Class'])
for index_val, series_val in y_test.iteritems():
    ylist1[iter1]=[series_val]
    iter1 += 1
    
ylist1 = ylist1.transpose().dropna()   

ylist = pd.concat([ylist,ylist1],axis=1)
ylist.columns = ['Index','Probability','Class']

ylist.sort_values(['Probability'],ascending=True,inplace=True)

ylist = ylist.dropna()

del ylist['Index']

yl = pd.DataFrame(columns=['Index'])
for i in range(0,9900):
    yl.loc[i] = i

ylist = pd.concat([yl,ylist],axis=1)   

ylist.sort_values(['Probability'],ascending=True,inplace=True)

#SSM
n = 50   
lim = ( len(ylist) - 50 ) 
#List of real probabilities calculated.
probs = pd.DataFrame(columns=["Actual Probability"])
val = 0.0

j=0
for i in range(50,lim):
    count = i
    counter = 1
    while counter <= 50:
        temp  = ylist.loc[ylist['Index']==count]
        val += temp.iat[0,2]
        count = count - 1
        counter += 1
    count = i+1
    counter = 1
    while counter <= 50:
        temp  = ylist.loc[ylist['Index']==count]
        val += temp.iat[0,2]
        count += 1
        counter += 1
    val = (val*1.0)/((2*n)+1)    
    probs.loc[j] = val
    j += 1
    val=0

#List of Predicted Probabilities vs Actual Probabilities.
proba = pd.DataFrame(columns = ['Predicted','Actual'])

proba = pd.concat([ylist,probs],axis=1)

list1=[]
#Dropping the last 150 rows from the proba dataframe.
for i in range(9800,9900):
    list1.append(i)
proba.drop(list1,inplace=True)

proba.sort_values(['Probability'],ascending=True,inplace=True)

#Generating the scatter plots.
import matplotlib.pyplot as plt

plt.scatter(proba['Probability'],proba['Actual Probability'])
plt.ylim(0,1)
plt.xlim(0,1)
plt.show()

#Trying to plot the LIFT curves.
X_train,X_test,y_train,y_test=train_test_split(df,y,test_size=0.33)
scaler1 = preprocessing.StandardScaler().fit(df)
X_test = scaler1.transform(X_test)
pp = nn.predict_proba(X_test)[:,1]
pp1 = pd.DataFrame(pp,columns=['Probability'])

X_test = pd.DataFrame(X_test)
temporary = nn.predict(X_test)
tempo = pd.DataFrame(temporary,columns=['Predicted Class'])
X_test = pd.concat([X_test,tempo],axis=1)
y1 = pd.DataFrame(y_test)
y1 = y1.reset_index()
del y1['index']
X_test = pd.concat([X_test,y1],axis=1)
X_test = pd.concat([X_test,pp1],axis=1)


X_test['newcol'] = np.where(X_test['Predicted Class']==1,1,-1)

X_test['newcol1'] = np.where(X_test['newcol']==X_test['default payment next month'],1,0)


X_test.sort_values(['Probability'],inplace=True,ascending=False)
X_test = X_test.reset_index()
del X_test['index']

outercounter=500
dfcounter=0
cnt=0
newdf = pd.DataFrame(columns=['No. of total data','Target data(no. of 1s)','Correctly Classified Samples'])
while outercounter <= 9500:
    cnt1 = X_test[0:outercounter].astype(bool).sum(axis=0)[28]
    cnt = X_test[:outercounter].astype(bool).sum(axis=0)[25]
    newdf.loc[dfcounter] = [(outercounter*1),cnt,cnt1]
    cnt=0
    cnt1=0
    dfcounter += 1
    outercounter += 500
    
plt.figure(0).clf    
plt.scatter(newdf[newdf.columns[0]],(newdf[newdf.columns[1]]*1.0)/newdf[newdf.columns[2]])   
plt.scatter(newdf[newdf.columns[0]],newdf[newdf.columns[1]]) 
plt.scatter(newdf[newdf.columns[0]],newdf[newdf.columns[2]]) 
plt.plot([0,10000],[0,3500],'r--')
plt.xlim(0,10000)
plt.ylim(0,3500)
plt.legend()
plt.show()


from scipy import stats
x = X_test[X_test.columns[26]].tolist()
y = probs[probs.columns[0]].tolist()
slope, intercept = stats.linregress(x, y) # your data x, y to fit


#Get the R^2 for 'nn' model.
#R^2 -> Coefficient of Determination.
from sklearn.metrics import r2_score
r2_score(y_test,nnpred)

# y = mx + c
# DA : array([-0.04862398,  0.22981695])

#ANN :
