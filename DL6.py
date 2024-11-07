import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load CIFAR-10 dataset directly from TensorFlow
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1 for better model performance
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert class labels to one-hot encoded format for categorical classification
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Step 2: Set up the VGG16 model with custom classification layers
# Load VGG16 without the top classification layer and with pre-trained ImageNet weights
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze all layers in the base model to retain pretrained features
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top of the base model for CIFAR-10 classification
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(10, activation='softmax')(x)

# Create the final model by specifying inputs and outputs
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

# Step 3: Train the model with the custom layers
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))

# Optional Step 4: Fine-tune the model by unfreezing the last 4 layers of the base model
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Recompile the model with a reduced learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Fine-tune the model on the CIFAR-10 dataset
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))

# Step 5: Make predictions and visualize results
predicted_values = model.predict(x_test)

# Map CIFAR-10 classes to their labels for easy interpretation
labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Visualize a single prediction by selecting an image index, e.g., 890
n = 890  # Change this index to view different images
plt.imshow(x_test[n])
plt.title(f"Predicted: {labels[np.argmax(predicted_values[n])]}, Actual: {labels[np.argmax(y_test[n])]}")
plt.show()
