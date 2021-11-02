import numpy as np


# preparing independent and dependent features
def prepare_data(timeseries_data, n_steps):
	X, y =[],[]
	for i in range(len(timeseries_data)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(timeseries_data)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = timeseries_data[i:end_ix], timeseries_data[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)
    
#defining the model
def model_build(nodes,activation,n_steps,n_features,optimizer):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Flatten

    # define model
    model = Sequential()
    model.add(LSTM(nodes, activation=activation, return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(nodes, activation=activation))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mse')
    return model

#training the model
def model_train(model,X,y,epochs):
    model = model
    model.fit(X, y, epochs=epochs, verbose=2)
    return model

def model_pred(model,input_sequence,n_steps,n_features,n_pred):
    # demonstrate prediction for next N days
    model = model
    input_sequence = np.array(input_sequence)
    new_sequence=list(input_sequence)
    output_sequence=[]
    i=0
    while(i<n_pred):    
        if(len(new_sequence)>n_steps):
            input_sequence=np.array(new_sequence[1:])
            input_sequence = input_sequence.reshape((1, n_steps, n_features))
            prediction = model.predict(input_sequence, verbose=0)
            new_sequence.append(prediction[0][0])
            new_sequence=new_sequence[1:]
            output_sequence.append(prediction[0][0])
            i=i+1
        else:
            input_sequence = input_sequence.reshape((1, n_steps, n_features))
            prediction = model.predict(input_sequence, verbose=0)
            new_sequence.append(prediction[0][0])
            output_sequence.append(prediction[0][0])
            i=i+1
        

    return output_sequence


