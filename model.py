import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import numpy as np
from obspy import read
import matplotlib.pyplot as plt


class MoonDM():
    def __init__(self):
        self.model = tf.keras.models.load_model('seismic_moon_lstm_model')
        self.scaler = StandardScaler()

    def fit_scaler(self, seismic_data):
        seismic_data = seismic_data.reshape(-1, 1)  # Reshape for fitting scaler
        self.scaler.fit(seismic_data)  # Fit the scaler with data

    def analyze(self, seismic_data):
        new_seismic_data = seismic_data.reshape(-1, 1, 1)  # Reshape as per model input
        
        # Scale the data using the fitted scaler
        new_seismic_data_scaled = self.scaler.transform(new_seismic_data.reshape(-1, 1)).reshape(new_seismic_data.shape)
        
        # Make predictions using the model
        predictions = self.model.predict(new_seismic_data_scaled)

        return predictions


class MarsDM():
    def __init__(self):
        self.model = tf.keras.models.load_model('seismic_mars_lstm_model')
        self.scaler = StandardScaler()

    def fit_scaler(self, seismic_data):
        seismic_data = seismic_data.reshape(-1, 1)  # Reshape for fitting scaler
        self.scaler.fit(seismic_data)  # Fit the scaler with data

    def analyze(self, seismic_data):
        new_seismic_data = seismic_data.reshape(-1, 1, 1)  # Reshape as per model input
        
        # Scale the data using the fitted scaler
        new_seismic_data_scaled = self.scaler.transform(new_seismic_data.reshape(-1, 1)).reshape(new_seismic_data.shape)
        
        # Make predictions using the model
        predictions = self.model.predict(new_seismic_data_scaled)

        return predictions