import numpy as np
import pandas as pd
import run

df = pd.read_csv('DailyDelhiClimateTrain.csv')
dfTest = pd.read_csv('DailyDelhiClimateTest.csv')

#reading only temp
#temp_train = df.iloc[:,1:2]
#temp_test = dfTest.iloc[:,1:2]

#pd frame to list conversion
temp_train = df['meantemp'].tolist()
temp_test = dfTest['meantemp'].tolist()
print(len(temp_train))
a = 50
print(len(temp_train[-a:]))
print(len(temp_train[:-a]))
#timeseries_data = []
timeseries_data = temp_train
# choose a number of time steps
n_steps = 50
n_features = 1
# split into samples
X, y = run.prepare_data(timeseries_data, n_steps)

#print(X), print(y)

X = X.reshape((X.shape[0], X.shape[1], n_features))
#print(X.shape)

model = run.model_build(50,'relu',n_steps,n_features,'adam')
#print(model.summary())

model = run.model_train(model,X,y,10)
#print(model)

#print(len(temp_test))
temp_test=temp_test[:n_steps]
#print(len(temp_test))

pred_output = run.model_pred(model,temp_test,n_steps,n_features,10)
print(pred_output)


