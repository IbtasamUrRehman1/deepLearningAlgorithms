import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt


# Define the CNN Arcitechture

def create_cnn(input_shape, num_classes):
    model = models.Sequential()

    # Convolutional Layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Flatten layers
    model.add(layers.Flatten())

    # Fully connected layers
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
                  )
    return model


# Specify the input shape and mumber of classes
input_shape = (28, 28, 1)  # Adjust dimensions based on your dataset
num_classes = 10  # Adjust based on the classification task

# create CNN model
cnn_model = create_cnn(input_shape, num_classes)

# Visualize the model acritecture and  save the plot
plot_model(cnn_model, to_file='cnn_model.png', show_shapes=True, show_layer_names=True)

# Display the model summary
cnn_model.summary()

# Show the visualiztion plot
img = plt.imread('cnn_model.png')
plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.axis('off')
plt.show()
