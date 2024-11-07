
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split


# Cell 2: Load and Inspect Dataset

path = 'http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv'
data = pd.read_csv(path, header=None)
data.head()
# Cell 3: Data Information
data.info()
# Cell 4: Split Data into Features and Target
features = data.drop(140, axis=1) 
target = data[140]
x_train, x_test, y_train, y_test = train_test_split(
 features, target, test_size=0.2
)

# Cell 5: Scale Training Data (for Label = 1)
train_index = y_train[y_train == 1].index
train_data = x_train.loc[train_index]
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
x_train_scaled = min_max_scaler.fit_transform(train_data.copy())

x_test_scaled = min_max_scaler.transform(x_test.copy())

# Cell 6: Define Autoencoder Model
class AutoEncoder(Model):
    def __init__(self, output_units, ldim=8):
        super().__init__()
        # Define the encoder part of the Autoencoder
        self.encoder = Sequential([
            Dense(64, activation='relu'),
            Dropout(0.1),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(16, activation='relu'),
            Dropout(0.1),
            Dense(ldim, activation='relu')
        ])
        # Define the decoder part of the Autoencoder
        self.decoder = Sequential([
            Dense(16, activation='relu'),
            Dropout(0.1),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(64, activation='relu'),
            Dropout(0.1),
            Dense(output_units, activation='sigmoid')
        ])

    def call(self, inputs):
        # Forward pass through the Autoencoder
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded
# Cell 7: Instantiate and Compile Model
# Create an instance of the AutoEncoder model with the appropriate output units
model = AutoEncoder(output_units=x_train_scaled.shape[1])

# Compile the model with Mean Squared Logarithmic Error (MSLE) loss and Mean Squared Error (MSE) metric
model.compile(loss='msle', metrics=['mse'], optimizer='adam')

# Cell 8: Train Model
# Train the model using the scaled training data
history = model.fit(
    x_train_scaled,  # Input data for training
    x_train_scaled,  # Target data for training (autoencoder reconstructs the input)
    epochs=20,       # Number of training epochs
    batch_size=512,  # Batch size
    validation_data=(x_test_scaled, x_test_scaled),  # Validation data
    shuffle=True     # Shuffle the data during training
)

# Cell 9: Plot Training and Validation Loss
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()

# Cell 10: Define Functions to Set Threshold and Get Predictions
 
def find_threshold(model, x_train_scaled):
    # Reconstruct the data using the model
    recons = model.predict(x_train_scaled)
    # Calculate the mean squared log error between reconstructed data and the original data
    recons_error = tf.keras.metrics.msle(recons, x_train_scaled)
    # Set the threshold as the mean error plus one standard deviation
    threshold = np.mean(recons_error.numpy()) + np.std(recons_error.numpy())
    return threshold

# Function to make predictions for anomalies based on the threshold
def get_predictions(model, x_test_scaled, threshold):
    # Reconstruct the data using the model
    predictions = model.predict(x_test_scaled)
    # Calculate the mean squared log error between reconstructed data and the original data
    errors = tf.keras.losses.msle(predictions, x_test_scaled)
    # Create a mask for anomalies based on the threshold 
    anomaly_mask = pd.Series(errors) > threshold
    # Map True (anomalies) to 0 and False (normal data) to 1
    preds = anomaly_mask.map(lambda x: 0.0 if x else 1.0)
    return preds

# Find the threshold for anomalies
# Cell 11: Calculate Threshold and Get Predictions
threshold = find_threshold(model, x_train_scaled)
print(f"Threshold: {threshold}")


predictions = get_predictions(model, x_test_scaled, threshold)
accuracy = accuracy_score(predictions, y_test)
print(f"Accuracy Score: {accuracy}")
