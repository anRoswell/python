#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.cbook as cbook


# In[32]:


#Crear un modelo que permita convertir grados celsius a grados Fahrenheit sin utilizar una fórmula
temperature_df = pd.read_csv("celsius_a_fahrenheit.csv")
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))
model.compile(optimizer=tf.keras.optimizers.Adam(1), loss='mean_squared_error')
epochs_hist = model.fit(temperature_df['Celsius'], temperature_df['Fahrenheit'], epochs = 100)
Temp_C = 0
Temp_F = model.predict([Temp_C])
print("Temperatura de Prediccion: " + str(Temp_F))
Temp_F = 9/5 * Temp_C + 32
print("Temperatura de Ecuacion: " + str(Temp_F))

