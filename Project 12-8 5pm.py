#!/usr/bin/env python
# coding: utf-8

# # loading data into Python

# In[1]:


import MySQLdb

data = MySQLdb.connect(host="localhost", user='username', passwd='password', db='db_name')
cursor = data.cursor()
cursor.execute('select * from Diabetes')
rows = cursor.fetchall()
print(len(rows))
print(type(rows))
rows


# # converting tuple to df

# In[2]:


import pandas as pd
df = pd.read_sql('select * from Diabetes', con=data)
df


# # Checking NA values

# In[3]:


null = df.isnull()
display(null)


# In[4]:


dfc = df.copy()
dfc.dropna()
dfc
# no na values!


# # number of positive symptoms for subjects with and without diabetes

# In[5]:


polyuriaDP = cursor.execute("select * from Diabetes where Polyuria ='Yes' and Class ='Positive'")
print('Polyuria with diabetes:', polyuriaDP)
polyuriaDN = cursor.execute("select * from Diabetes where Polyuria ='Yes' and Class ='Negative'")
print('Polyuria without diabetes', polyuriaDN)


polydipsiaDP = cursor.execute("select * from Diabetes where Polydipsia ='Yes' and Class ='Positive'")
print('Polydipsia with diabetes:', polydipsiaDP)
polydipsiaDN = cursor.execute("select * from Diabetes where Polydipsia ='Yes' and Class ='Negative'")
print('Polydipsia without diabetes', polydipsiaDN)

weightDP = cursor.execute("select * from Diabetes where Sudden_Weight_Loss ='Yes' and Class ='Positive'")
print('weight loss with diabetes:', weightDP)
weightDN = cursor.execute("select * from Diabetes where Sudden_Weight_Loss ='Yes' and Class ='Negative'")
print('weight loss without diabetes', weightDN)

weaknessDP = cursor.execute("select * from Diabetes where Weakness ='Yes' and Class ='Positive'")
print('weakness with diabetes:', weaknessDP)
weaknessDN = cursor.execute("select * from Diabetes where Weakness ='Yes' and Class ='Negative'")
print('weakness without diabetes', weaknessDN)

polyphagiaDP = cursor.execute("select * from Diabetes where Polyphagia ='Yes' and Class ='Positive'")
print('polyphagia with diabetes:', polyphagiaDP)
polyphagiaDN = cursor.execute("select * from Diabetes where Polyphagia ='Yes' and Class ='Negative'")
print('polyphagia without diabetes', polyphagiaDN)

genitalDP = cursor.execute("select * from Diabetes where Genital_Thrush ='Yes' and Class ='Positive'")
print('genital thrush with diabetes:', genitalDP)
genitalDN = cursor.execute("select * from Diabetes where Genital_Thrush ='Yes' and Class ='Negative'")
print('genital thrush without diabetes', genitalDN)

blurringDP = cursor.execute("select * from Diabetes where Visual_Blurring ='Yes' and Class ='Positive'")
print('visual blurring with diabetes:', blurringDP)
blurringDN = cursor.execute("select * from Diabetes where Visual_Blurring ='Yes' and Class ='Negative'")
print('visual blurring without diabetes', blurringDN)

itchDP = cursor.execute("select * from Diabetes where Itching ='Yes' and Class ='Positive'")
print('itching with diabetes:', itchDP)
itchDN = cursor.execute("select * from Diabetes where Itching ='Yes' and Class ='Negative'")
print('itching without diabetes', itchDN)

irritableDP = cursor.execute("select * from Diabetes where Irritability ='Yes' and Class ='Positive'")
print('irritability with diabetes:', irritableDP)
irritableDN = cursor.execute("select * from Diabetes where Irritability ='Yes' and Class ='Negative'")
print('irritability without diabetes', irritableDN)

healingDP = cursor.execute("select * from Diabetes where Delayed_Healing ='Yes' and Class ='Positive'")
print('delayed healing with diabetes:', healingDP)
healingDN = cursor.execute("select * from Diabetes where Delayed_Healing ='Yes' and Class ='Negative'")
print('delayed healing without diabetes', healingDN)

paresisDP = cursor.execute("select * from Diabetes where Partial_Paresis ='Yes' and Class ='Positive'")
print('partial paresis with diabetes:', paresisDP)
paresisDN = cursor.execute("select * from Diabetes where Partial_Paresis ='Yes' and Class ='Negative'")
print('partial paresis without diabetes', paresisDN)

muscleDP = cursor.execute("select * from Diabetes where Muscle_Stiffness ='Yes' and Class ='Positive'")
print('muscle stiffness with diabetes:', muscleDP)
muscleDN = cursor.execute("select * from Diabetes where Muscle_Stiffness ='Yes' and Class ='Negative'")
print('muscle stiffness without diabetes', muscleDN)

alopeciaDP = cursor.execute("select * from Diabetes where Alopecia ='Yes' and Class ='Positive'")
print('alopecia with diabetes:', alopeciaDP)
alopeciaDN = cursor.execute("select * from Diabetes where Alopecia ='Yes' and Class ='Negative'")
print('alopecia without diabetes', alopeciaDN)

obesityDP = cursor.execute("select * from Diabetes where Obesity ='Yes' and Class ='Positive'")
print('obesity with diabetes:', obesityDP)
obesityDN = cursor.execute("select * from Diabetes where Obesity ='Yes' and Class ='Negative'")
print('obesity without diabetes', obesityDN)

diabetesP = cursor.execute("select * from Diabetes where Class ='Positive'")
print('diabetes positive', diabetesP)
diabetesN = cursor.execute("select * from Diabetes where Class ='Negative'")
print('diabetes negagtive', diabetesN)


# In[21]:


import matplotlib.pyplot as plt
import numpy as np
atrbt_list = [polyuriaDP, polyuriaDN, polydipsiaDP, polydipsiaDN, weightDP, weightDN, weaknessDP, weaknessDN, polyphagiaDP, polyphagiaDN, genitalDP, genitalDN, blurringDP, blurringDN, itchDP, itchDN, irritableDP, irritableDN, healingDP, healingDN, paresisDP, paresisDN, muscleDP, muscleDN, alopeciaDP, alopeciaDN, obesityDP, obesityDN, diabetesP, diabetesN]
# seperating the values of attributes with and without diabetes
list1=  atrbt_list[0::2]
list2=  atrbt_list[1::2] 

labels = ['Polyuria', 'Polydipsia', 'Sudden Weight Loss', 'Weakness', 'Polyphagia', 'Genital Thrush', 'Visual Blurring', 'Itching', 'Irritability', 'Delayed Healing', 'Partial Paresis', 'Muscle Stiffness', 'Alopecia', 'Obesity', 'Diabetes']

Diabetes_Positive = list1
Diabetes_Negative = list2

# setting the location of labels
x = np.arange(len(labels))  
# setting the width of the bars
width = 0.45  

fig = plt.figure(figsize=(22, 11))
ax = fig.add_subplot(111)
bar1 = ax.bar(x - width/2, Diabetes_Positive, width, label='Diabetes Positive')
bar2 = ax.bar(x + width/2, Diabetes_Negative, width, label='Diabetes Negative')

ax.set_ylabel('Number of subjects')
ax.set_title('Frequency of Symptoms and Diabetes')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

def autolabel(bars):
    """To Attach the value label above each bar in, displaying its height."""
    for i in bars:
        height = i.get_height()
        ax.annotate('{}'.format(height),
                    xy=(i.get_x() + i.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(bar1)
autolabel(bar2)
fig.tight_layout()
plt.show()


# # change values to binary

# In[7]:


import pandas as pd
df = pd.read_csv ('diabetes.csv')
df.columns = ['Age','Gender','Polyuria','Polydipsia','Sudden_Weight_Loss','Weakness','Polyphagia','Genital_Thrush','Visual_Blurring','Itching','Irritability','Delayed_Healing','Partial_Paresis','Muscle_Stiffness','Alopecia','Obesity','Class']

df = df.replace(to_replace=['No', 'Yes'], value=[0, 1])
df.Gender = df.Gender.replace(to_replace=['Male', 'Female'], value=[0, 1])
df.Class = df.Class.replace(to_replace=['Negative', 'Positive'], value=[0, 1])
df


# # summary stats (counts)

# In[8]:


countgender = df['Gender'].value_counts()
countpolyuria = df['Polyuria'].value_counts()
countpolydipsia = df['Polydipsia'].value_counts()
countweightloss = df['Sudden_Weight_Loss'].value_counts()
countweakness = df['Weakness'].value_counts()
countpolyphagia = df['Polyphagia'].value_counts()
countgenitalthrush = df['Genital_Thrush'].value_counts()
countvisualblurring = df['Visual_Blurring'].value_counts()
countitching = df['Itching'].value_counts()
countirritability = df['Irritability'].value_counts()
countdelayedhealing = df['Delayed_Healing'].value_counts()
countparesis = df['Partial_Paresis'].value_counts()
countmusclestiffness = df['Muscle_Stiffness'].value_counts()
countalopecia = df['Alopecia'].value_counts()
countobesity = df['Obesity'].value_counts()
        
print(countgender)
print(countpolyuria)
print(countpolydipsia)
print(countweightloss)
print(countweakness)
print(countpolyphagia)
print(countgenitalthrush)
print(countvisualblurring)
print(countitching)
print(countirritability)
print(countdelayedhealing)
print(countparesis)
print(countmusclestiffness)
print(countalopecia)
print(countobesity) 


# # Correlation

# In[9]:


df.corr()


# # Correlation Heatmap

# In[10]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
cor = df.corr()
x = 17
coll = cor.nlargest(x, 'Class')['Class'].index
z = np.corrcoef(df[coll].values.T)
plt.figure(figsize=(18,11))
sns.heatmap(z, yticklabels = coll.values, xticklabels = coll.values, annot = True)


# In[19]:


corrMatrix = df.corr()
pearson = df.corr(method='pearson',min_periods=1)
#print(pearson)
sns.heatmap(corrMatrix, xticklabels=corrMatrix.columns, yticklabels=corrMatrix.columns, annot=True)
plt.rcParams['figure.figsize']= (30,20)
plt.rcParams['font.size'] = 10
plt.show()


# # Shapiro test

# In[12]:


from scipy.stats import shapiro
test1, p1 = shapiro(df)
print('test = %.3f, p = %.3f' % (test1, p1))
plt.hist(df)


# # Logisitic Regression

# In[13]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

#log reg on all variables

X = df
y = df['Class']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)

lg = LogisticRegression()
lg.fit(X_train,y_train)
y_pred = lg.predict(X_test)


confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True)

print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
plt.show()


# In[14]:


#log reg on polyuria and polydipsia

X = df[['Polyuria', 'Polydipsia']]
y = df['Class']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)

lg = LogisticRegression()
lg.fit(X_train,y_train)
y_pred = lg.predict(X_test)


confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True)

print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
plt.show()


# In[15]:


#logreg on polyuria, polydipsia, weight loss, and paresis

X = df[['Polyuria', 'Polydipsia', 'Sudden_Weight_Loss', 'Partial_Paresis', 'Alopecia', 'Polyphagia']]
y = df['Class']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)

lg = LogisticRegression()
lg.fit(X_train,y_train)
y_pred=lg.predict(X_test)


confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True)

print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
plt.show()


# In[16]:


#def split80_20(examples):
#    sampleIndices = random.sample(range(len(examples)),len(examples)//5)
#    trainingSet, testSet = [], []
#    for i in range(len(examples)):
#        if i in sampleIndices:
#            testSet.append(examples[i])
#        else:
#            trainingSet.append(examples[i])
#    return trainingSet, testSet


# In[17]:


log = LogisticRegression()
log.fit(X_train,y_train)
predicted =lg.predict(X_test)
expected = y_test
matches = (predicted == expected)
print(matches.sum())
print("Overall Accuracy for Logistic Regression = ", matches.sum()/float(len(matches)))

from sklearn import metrics
print(metrics.classification_report(expected, predicted))

plt.rcParams['figure.figsize'] = [5, 5]
predicted_probas = log.predict_proba(X_test)
import matplotlib.pyplot as plt
import scikitplot as skplt
skplt.metrics.plot_roc(expected, predicted_probas)

plt.show()


# # Comparison of other ML models

# In[20]:


import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import pandas as pd
import os

y = df.iloc[:,16]
x = df.iloc[:,:16]

#logistic regression
lr = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr').fit(x,y)
lr.predict(x.iloc[519:,:])
print('Logistic Regression Score:', round(lr.score(x,y), 4))

#support vector machines
svm = svm.LinearSVC()
svm.fit(x,y)
svm.predict(x.iloc[519:,:])
print('SVM Score:', round(svm.score(x,y), 4))

#random forest
rf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
rf.fit(x,y)
rf.predict(x.iloc[519:,:])
print('Random Forest Score:', round(rf.score(x,y), 4))

#neural networks
nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)
nn.fit(x,y)
nn.predict(x.iloc[519:,:])
print('Neural Networks Score:', round(nn.score(x,y), 4))


# In[ ]:





# In[ ]:




