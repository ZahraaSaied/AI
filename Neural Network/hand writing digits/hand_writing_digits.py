# import tenserflow 
import tensorflow as tf  

# import the minst dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# normalize the data (scales data between 0 and 1)
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

# Build the network
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense

# 1- basic feed-forward model
model = Sequential() 
 
# 2- takes our 28x28 and makes it 1x784
model.add(Flatten()) 

#  3- two fully-connected layer, 128 units, relu activation
model.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu')) 

# 4- output layer 
model.add(Dense(units = 10, activation = 'softmax'))

# Configure the learning process
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model on the training set
model.fit(x_train, y_train, epochs = 3)  

# Evaluate the out of sample data with model
val_loss, val_acc = model.evaluate(x_test, y_test)  
print(val_loss)  # model's loss
print(val_acc)  # model's accuracy

# Save the model
model.save('num_reader.model')

# Load it back:
new_model = tf.keras.models.load_model('num_reader.model')

# Finally, let's make predictions!
predictions = new_model.predict(x_test)
print(predictions)

##############
import matplotlib.pyplot as plt
import numpy as np

print(np.argmax(predictions[9]))

plt.imshow(x_test[9], cmap = plt.cm.binary)
plt.show()