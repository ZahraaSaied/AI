# Predicting number of videos views based on number of likes, dislikes and subscribers

# import the data
import pandas as pd 
X = pd.read_csv("StatsVideosXALL.csv")
y = pd.read_csv("StatsVideosYALL.csv")

# split the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# scale the data
from sklearn.preprocessing import scale
x_train = scale(x_train)
x_test = scale(x_test)


# build the network
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(units = 10, activation = 'relu', input_shape = (3,)))
model.add(Dense(units = 10, activation = 'relu'))
model.add(Dense(units = 10, activation = 'relu'))
model.add(Dense(units = 10, activation = 'relu'))
model.add(Dense(units = 10, activation = 'relu'))
model.add(Dense(units = 1,))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')


# Early Stopping function
from keras.callbacks import EarlyStopping
earlystopper = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 15, verbose = 1, mode = 'auto')


history = model.fit(x = x_train, 
                    y = y_train, 
                    epochs = 300, 
                    validation_split = 0.1, 
                    shuffle = True, 
                    verbose = 0, 
                    callbacks = [earlystopper])

# plotting
import matplotlib.pyplot as plt
history_dict = history.history
loss_value = history_dict['loss']
val_loss_value = history_dict['val_loss']
plt.plot(loss_value, 'bo', label = 'training loss')
plt.plot(val_loss_value, 'r', label = 'training loss val')

# predicting 
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

# r2 score of training and testing data
from sklearn.metrics import r2_score
print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred)))
print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_test_pred)))


