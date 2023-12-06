import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder 

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

iris_flower_file=pd.read_csv("iris flower file.csv")

iris_flower_file.head(16)

iris_flower_file.shape

iris_flower_file.info()

iris_flower_file.describe()

iris_flower_file.isnull().sum()

iris_flower_file.describe()

iris_flower_file['sepal_length'].hist()

iris_flower_file['sepal_width'].hist()

iris_flower_file['petal_length'].hist()

iris_flower_file['petal_width'].hist()

colors=['red','Black','teal']

species=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

for i in range(3):

    x=iris_flower_file[iris_flower_file['species']==species[i]]
    
    plt.scatter(x['sepal_length'],x['sepal_width'],c=colors[i],label=species[i])
    
plt.xlabel("Sepal Length")

plt.ylabel("Sepal Width")

plt.legend()

for i in range(3):

    x=iris_flower_file[iris_flower_file['species']==species[i]]
    
    plt.scatter(x['petal_length'],x['petal_width'],c=colors[i],label=species[i])
    
plt.xlabel("Petal Length")

plt.ylabel("Petal Width")

plt.legend()

for i in range(3):

    x=iris_flower_file[iris_flower_file['species']==species[i]]
    
    plt.scatter(x['sepal_length'],x['petal_length'],c=colors[i],label=species[i])
    
plt.xlabel("Sepal Length")

plt.ylabel("Petal Length")

plt.legend()

for i in range(3):

    x=iris_flower_file[iris_flower_file['species']==species[i]]
    
    plt.scatter(x['sepal_width'],x['petal_width'],c=colors[i],label=species[i])
    
plt.xlabel("Sepal Width")

plt.ylabel("Petal Width")

plt.legend()

numeric_columns=iris_flower_file.drop(columns='species')

corr=numeric_columns.corr()

fig,axis=plt.subplots(figsize=(5,5))

sns.heatmap(corr,annot=True,ax=axis,cmap='coolwarm')

le=LabelEncoder()

iris_flower_file['species']=le.fit_transform(iris_flower_file['species'])

iris_flower_file.head(16)

x=iris_flower_file.drop(columns='species')

y=iris_flower_file['species']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

LR=LogisticRegression()

LR.fit(x_train,y_train)

KNN=KNeighborsClassifier()

KNN.fit(x_train,y_train)

DT=DecisionTreeClassifier()

DT.fit(x_train,y_train)

LR_accuracy=LR.score(x_test,y_test)*100

KNN_accuracy=KNN.score(x_test,y_test)*100

DT_accuracy=DT.score(x_test,y_test)*100

print(f"Accuracy by using Logistic Regression: {LR_accuracy}%")

print(f"Accuracy by using K Nearest Neighbors Algorithm: {KNN_accuracy}%")

print(f"Accuracy by using Decision Tree Classifier: {DT_accuracy}%")

