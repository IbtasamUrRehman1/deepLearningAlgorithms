import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import datetime
# load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split ther dataset into training and testig sets
X_train, X_test, y_train ,y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Standarddize the features
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

# Build a simple neural network using tensorFlow keras
model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))


# Visualize the neural network arctitecture and save it as image
plot_model(model, to_file='neural_network_architecture.png', show_shapes=True, show_layer_names=True)

# compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# setup tensorboard callback
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%M%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# train the model with tensor board
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[tensorboard_callback])

# Evalute the model on the test set
loss, accuracy= model.evaluate(X_test, y_test)
print(f"Test loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Display the neural network architecture image ( optional )
img = plt.imread('neural_network_architecture.png')
plt.imshow(img)
plt.show()

















