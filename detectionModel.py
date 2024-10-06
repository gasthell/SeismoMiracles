import numpy as np
import pandas as pd
import os
from datetime import datetime

# Paths to the folders containing seismic data and the timestamp file
seismic_folder = ".\\space_apps_2024_seismic_detection\\data\\mars\\training\\data"
timestamp_file = ".\\space_apps_2024_seismic_detection\\data\\mars\\training\\catalogs\\Mars_InSight_training_catalog_final.csv"

# Load the timestamp file
timestamps_df = pd.read_csv(timestamp_file)

# Function to create labels for a single seismic data file based on timestamps
def label_seismic_data(seismic_data, earthquake_timestamps, sampling_rate, window_size=10):
    num_samples = seismic_data.shape[0]
    labels = np.zeros(num_samples)
    
    # Convert relative time (seconds) to sample indices
    for time_rel in earthquake_timestamps:
        earthquake_idx = int(time_rel * sampling_rate)
        start_idx = max(0, earthquake_idx - window_size // 2)
        end_idx = min(num_samples, earthquake_idx + window_size // 2)
        labels[start_idx:end_idx] = 1  # Label the region as earthquake
    
    return labels

# Iterate over seismic data files mentioned in the timestamp file
seismic_data_list = []
labels_list = []

for filename in timestamps_df['filename'].unique():
    # Load seismic data for the current file from CSV
    seismic_data_path = os.path.join(seismic_folder, filename + ".csv")
    
    # Read the CSV into a DataFrame
    seismic_df = pd.read_csv(seismic_data_path)
    
    # Extract the relevant data: time_rel and velocity
    time_rel_data = seismic_df['time_rel(sec)'].values
    velocity_data = seismic_df['velocity(m/s)'].values
    
    # Get the earthquake timestamps for this file
    earthquake_timestamps = timestamps_df[timestamps_df['filename'] == filename]['time_rel(sec)'].values
    
    # Assuming a known sampling rate for the seismic data (samples per second)
    # This depends on how densely the data is sampled; for now, we assume it’s provided or inferred
    sampling_rate = 8  # Replace with the actual sampling rate for your data
    
    # Create labels for the seismic data
    labels = label_seismic_data(velocity_data, earthquake_timestamps, sampling_rate)
    
    # Append the data and labels to the lists
    seismic_data_list.append(velocity_data)
    labels_list.append(labels)

# Combine data from all files
seismic_data_all = np.hstack(seismic_data_list)  # Horizontal stack for time-series data
labels_all = np.hstack(labels_list)

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(seismic_data_all, labels_all, test_size=0.2, random_state=42)

# Reshape the data to fit the LSTM input format (samples, time steps, features)
X_train = X_train.reshape(-1, 1, 1)  # Adjust as necessary if data has multiple features
X_test = X_test.reshape(-1, 1, 1)

# Scale the data (optional)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

# Build an LSTM model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=2, batch_size=32, validation_split=0.2)

# Save the entire model in the TensorFlow SavedModel format
model.save('seismic_mars_lstm_model')

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test_scaled, y_test)
print(f"Test Accuracy: {test_acc:.2f}")
model = tf.keras.models.load_model('seismic_lstm_model')