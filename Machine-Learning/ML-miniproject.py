import pandas as pd
from matplotlib import pyplot as plt

titanic_train=pd.read_csv('train.csv')
titanic_test=pd.read_csv('test.csv')

titanic_train.head()

titanic_train.shape

titanic_train['Survived'].value_counts()

plt.figure(figsize=(5,5))
plt.bar(list(titanic_train['Survived'].value_counts().keys()),list(titanic_train['Survived'].value_counts()),color="r")
plt.show()

titanic_train['Pclass'].value_counts()

titanic_train['Sex'].value_counts()

plt.figure(figsize=(5,5))
plt.bar(list(titanic_train['Sex'].value_counts().keys()),list(titanic_train['Sex'].value_counts()),color="Green")
plt.show()

plt.figure(figsize=(5,7))
plt.hist(titanic_train['Age'])
plt.title("Distribution of Age")
plt.xlabel("Age")
plt.show()

sum(titanic_train['Survived'].isnull())

titanic_train=titanic_train.dropna()

x_train=titanic_train[['Age']]
y_train=titanic_train[['Survived']]

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(x_train,y_train)

sum(titanic_test['Age'].isnull())

x_test=titanic_test[['Age']]
y_pred=dtc.predict(x_test)

y_pred
