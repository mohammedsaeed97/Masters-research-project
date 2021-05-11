
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import math
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.linear_model import LinearRegression

#loading the dataset
df=pd.read_csv('/Users/Momo/Desktop/FTSE-DATA.csv', date_parser=True)
df
df.head()

#Plotting FTSE 100 Data
plt.figure(figsize=(20,4))
plt.plot(df['Close Price'])
plt.legend(['Closing Price'])
plt.title('FTSE 100')
plt.show()

#Creating a dataframe for close price only

data=df.filter(['Close Price'])
data

#Seeing how big our training data should be(training data length)

dataset=data.values
training_data_len=math.ceil(len(dataset)*0.8)
training_data_len

#Used to scale data-helps the model

scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)
scaled_data


#Creating training data
train_data=scaled_data[0:training_data_len,:]#contains values from 0 to training data length

#splitting the data into x train and y train, create empty lists
X_train=[]
y_train=[]

#allows us to use past 60 days into the X train,
for i in range(60, len(train_data)):
    X_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])#contains 61st value



#converting x train and y train into a Numpy array
X_train,y_train=np.array(X_train),np.array(y_train)
X_train.shape, y_train.shape
X_train

#This reshapes x train data set, LSTM expects a 3 dimensional dataset, its currently 2 dimensional
                            #No. of rows        #No. of colums
X_train=np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_train.shape

#Creating LSTM model- model architecture

model=Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25)) #adds a dense layer, 25 neurons
model.add(Dense(1))

#Running our LSTM model
model.compile(optimizer='adam', loss='mean_squared_error')

#Trains the LSTM model
model.fit(X_train, y_train, batch_size=32, epochs=10)

#creating a testing set similar to training set

test_data=scaled_data[training_data_len-60:,:]

#lists for testing set
X_test=[]
y_test=dataset[training_data_len:,:]

for i in range(60,len(test_data)):
    X_test.append(test_data[i-60:i,0])


# Converting x test into a numpy array then reshaping it
X_test=np.array(X_test)
X_test.shape

X_test=np.reshape(X_test, (X_test.shape[0],X_test.shape[1], 1))

#getting the predictions
predictions=model.predict(X_test)
predictions=scaler.inverse_transform(predictions)

#rmse metrics
rmse=np.sqrt(np.mean(predictions-y_test)**2)
rmse

#visualizing the data
train=data[:training_data_len]
valid=data[training_data_len:]
valid['Predictions']=predictions

plt.figure(figsize=(20,4))
plt.plot(train['Close Price'])
plt.plot(valid[['Close Price', 'Predictions']])
plt.legend(['Train', 'Validation', 'Test'])
plt.title('LSTM Model Output')
plt.show()


valid.head(10)
validc=valid['Close Price']
validc
validp=valid['Predictions']

#Coefficient of determinantion metrics
correlation_matrix = np.corrcoef(validc, validp)
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2
r_squared

