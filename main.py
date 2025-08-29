#libraries used in the project
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
data=pd.read_csv('heart.csv')
data.head()

data.shape
data.target.value_counts()
data.isna().sum()

import seaborn as sns
sns.countplot(x=data["target"])
categorical_val = []
continous_val = []
for column in data.columns:
    if len(data[column].unique())<=9:
        categorical_val.append(column)
    else:
        continous_val.append(column)
categorical_val
continous_val

plt.figure(figsize=(15, 15))
for i, column in enumerate(categorical_val, 1):
    plt.subplot(3, 3, i)
    data[data["target"] == 0][column].hist(bins=35, color='blue', label='Have Heart Disease = NO', alpha=0.6)
    data[data["target"] == 1][column].hist(bins=35, color='red', label='Have Heart Disease = YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)

plt.figure(figsize=(15, 15))
for i, column in enumerate(continous_val, 1):
    plt.subplot(3, 3, i)
    data[data["target"] == 0][column].hist(bins=35, color='blue', label='Have Heart Disease = NO', alpha=0.6)
    data[data["target"] == 1][column].hist(bins=35, color='red', label='Have Heart Disease = YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)
categorical_val.remove('target')
dataset = pd.get_dummies(data, columns = categorical_val)
print(data.columns)
print(dataset.columns)

from sklearn.preprocessing import StandardScaler
s_sc = StandardScaler()
col_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[col_to_scale] = s_sc.fit_transform(dataset[col_to_scale])
dataset.head()

from sklearn.model_selection import train_test_split
X = dataset.drop('target', axis=1) #features
y = dataset.target #target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) #split into train and test

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
lr_clf = LogisticRegression()
lr_clf.fit(X_train, y_train)
y_pred1=lr_clf.predict(X_test)
print('Accuracy: ',metrics.accuracy_score(y_test, y_pred1))

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test, y_pred1))

print(confusion_matrix(y_test, y_pred1))

conf_matrix=confusion_matrix(y_test, y_pred1)

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


#import the KNeighborsClassifier class from sklearn
from sklearn.neighbors import KNeighborsClassifier
k_range = range(1,26) #1-25
scores = {}
scores_list = []
for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)  #model building
        knn.fit(X_train,y_train) #training
        y_pred2=knn.predict(X_test) #testing
        scores[k] = metrics.accuracy_score(y_test,y_pred2)
        scores_list.append(metrics.accuracy_score(y_test,y_pred2))
scores

knn = KNeighborsClassifier(n_neighbors=19)
knn.fit(X_train,y_train)  #model is trained
y_pred2=knn.predict(X_test)
print('Accuracy: ',metrics.accuracy_score(y_test, y_pred2))

print(classification_report(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred2))
pickle.dump(knn, open('heartmodel.pkl', 'wb'))


#implementation of support vector machine
from sklearn.svm import SVC
clf=SVC(kernel='rbf',C=100,gamma=0.001) #model creation
clf.fit(X_train,y_train) #training the model
y_pred3=clf.predict(X_test) #predicting the data using the model
print('Accuracy: ',metrics.accuracy_score(y_test, y_pred3)) #finding the accuracy

from sklearn.tree import DecisionTreeClassifier
###code goes here for decision tree

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('heart.csv')

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


df = df.dropna()
X = df.drop('target', axis=1)
y = df['target']

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=np.unique(y).astype(str))
plt.title("Decision Tree Visualization")
plt.show()

from sklearn.ensemble import RandomForestClassifier
###code goes here for Random forest

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('heart.csv')

print("First 5 Rows")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rfc = RandomForestClassifier(n_estimators=100, random_state=42)

rfc.fit(X_train, y_train)


y_pred = rfc.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

importances = pd.Series(rfc.feature_importances_, index=X.columns)
importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Feature Importances')
plt.show()

from sklearn.model_selection import GridSearchCV
param_grid={'C':[0.1,1.0,10.0,100.0,1000.0],
           'gamma':[1,0.1,0.01,0.001,0,0.0001],
           'kernel':['linear', 'poly', 'rbf', 'sigmoid']}

grid=GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
grid.fit(X_train, y_train)

