import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as srn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
import tensorflow as tf

encountersDataHeading = np.array(['race', 'gender', 'age', 'time in hospital', 'num procedures', 'num medications', 'readmitted'])
# 
encountersData = pd.read_csv('./final_data.csv', names=['age', 'time in hospital', 'num procedures', 'num medications', 'readmitted', 'race', 'gender'], index_col=0, header=None)

encountersTargetY = encountersData['readmitted']
encountersTargetX = encountersData.drop(columns = ['readmitted'], axis=1) #remove label readmitted from column axis


# Split the data we have into x_train y_train x_test y_test
x_train, x_test, y_train, y_test = train_test_split(encountersTargetX, encountersTargetY, random_state=42) #random_state should be a seed

x_train_scaled = x_train
x_test_scaled = x_test

######### Standardization
# scaler = StandardScaler()
# xScaler = scaler.fit(x_train)
# x_train_scaled = xScaler.transform(x_train)
# x_test_scaled = xScaler.transform(x_test)

######### Normalization
x_train_scaled = normalize(x_train, axis=0)
x_test_scaled = normalize(x_test, axis=0)


numInputFeatures = len(x_train_scaled[0])
hiddenLayerOneNumNodes = 7
hiddenLayerTwoNumNodes = 14
hiddenLayerThreeNumNodes = 21
nn = tf.keras.models.Sequential()

nn.add(tf.keras.layers.Dense(units=hiddenLayerOneNumNodes, input_dim=numInputFeatures, activation='relu'))
nn.add(tf.keras.layers.Dense(units=hiddenLayerTwoNumNodes, activation='relu'))
nn.add(tf.keras.layers.Dense(units=hiddenLayerThreeNumNodes, activation='relu'))
nn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

nn.summary()    

nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
nn.fit(x_train_scaled, y_train, validation_split=0.15, epochs=100) # minibatch here

nnLoss, nnAccuracy = nn.evaluate(x_test_scaled, y_test, verbose=2)
print('Model: Neural Network')
print(f"Loss: {nnLoss}, Accuracy: {nnAccuracy}")

