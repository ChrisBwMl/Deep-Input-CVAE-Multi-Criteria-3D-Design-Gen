import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

# Laden der latenten Vektoren und der Trainingsdaten
z = np.load('z.npy')
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')

# Größe der Input- und Output-Formen
input_shape = 9   # y_label
output_shape = 274  # z
Nr_epochs = 150
batch_size = 128

# Definition des Neuronalen Netzwerks von y nach z
Input = keras.Input(shape=(input_shape,), name='Designs')
x = Dense(units=16, activation='relu')(Input)
x = Dense(units=32, activation='relu')(x)
x = Dense(units=256, activation='relu')(x)
x = Dense(units=512, activation='relu')(x)
x = Dense(units=512, activation='linear')(x)
Output = Dense(units=output_shape)(x)

NNyz = Model(Input, Output, name='NN_y_to_z')

# Kompilieren des Modells
ADAM = keras.optimizers.Adam(learning_rate=0.0002)
NNyz.compile(optimizer=ADAM, loss='mean_absolute_error')

# Training des Modells
history = NNyz.fit(y_train, z, validation_split=0.2, shuffle=True, epochs=Nr_epochs, batch_size=batch_size, verbose=1)

# Speichern der trainierten Gewichte
NNyz.save_weights('NN_y_to_z_weights.h5')

# Visualisierung des Trainingsverlaufs
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Auswählen einer bestimmten Anzahl von Designs und Überprüfen der latenten Raumrekonstruktion
indices = list(range(13345, 13544, 2))
y_concrete = np.take(y_train, indices, axis=0)
y_concrete = np.squeeze(y_concrete, axis=0)
z_pred_y_train = NNyz.predict(y_concrete)

# Speichern der vorhergesagten z-Werte
np.save('z_predicted_All_fromNN', z_pred_y_train)
