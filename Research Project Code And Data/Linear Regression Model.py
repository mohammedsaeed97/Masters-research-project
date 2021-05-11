
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#Loading the datasets and showing the dataset
df=pd.read_csv('/Users/Momo/Desktop/FTSE-DATA.csv', date_parser=True)
df
df.describe()

#Setting the independant and dependant variables
x=df[['Open Price','High Price', 'Low Price', 'Volume']].values
x
y=df[['Close Price']].values


#Splitting data into testing and training sets
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=0)

#Building Linear regression model
regressior=LinearRegression()
regressior.fit(x_train, y_train)


#Coefficient of determinantion
print(regressior.coef_)


predicted=regressior.predict(x_test)
regressior.score(x_train, y_train)
dframe=pd.DataFrame({'Actual':y_test.flatten(),'Predicted':predicted.flatten()})

#Showing real values and predicted values
dframe.head(10)
#root Mean sqaured error

print('root Mean Squared Error:', math.sqrt(metrics.mean_squared_error(y_test, predicted) ))

#Visualizing the data
graph=dframe.tail(100)
graph.plot(kind='line')

plt.figure(figsize=(50,4))
plt.plot(dframe)
plt.show()

regressior
regressior.score(x_test, y_test)




