# iris-data-classify-the-flowers-into-among-the-three-species

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
url="http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names=["sepal_length","sepal_width","petal_length","petal_width","class1"]
data=pd.read_csv(url,names=names)
data

data.describe()
data.info()
data.head()
data.groupby("class1").size()
#for barplot of data

data.plot(kind="bar",subplots=True,layout=(2,2))
##for histogram 
data.hist()

#pairplot of data
sns.pairplot(data,hue=species)

#correlation of data
corr=data.corr()
corr
sns.heatmap(corr)

#relation between datas
plt.scatter(data.sepal_length,data.petal_length,color=["red,"blue"]
plt.title("sepal_length vs petal_length")
data.xlabel("sepal_length")
data.ylabel("petal_length")
plt.show()


plt.scatter(data.petal_width,data.petal_length,color=["red,"blue"]
plt.title("petal_width vs petal_length")
data.xlabel("petal_width")
data.ylabel("petal_length")
plt.show()


plt.scatter(data.sepal_length,data.sepal_width,color=["red,"blue"]
plt.title("sepal_length vs sepal_width")
data.xlabel("sepal_length")
data.ylabel("sepal_width")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
#data visualization
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

lin=DecisionTreeRegressor()
lin.fit(x_train,y_train)

y_pred=lin.predict(x_test)

#check the value of actual and predicted (y_test)
pd.DataFrame({"actual":y_test,"predicted":y_pred})

#checking the accuracy and matrix of data
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

#predict the value for class1
lin.predict([[6,2,3,4]])


thanks
..
