import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as srn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf

labelEncoder = LabelEncoder()
smote = SMOTE()
overSampler = RandomOverSampler()

# encountersDataHeading = np.array(['race', 'gender', 'age', 'time in hospital', 'num procedures', 'num medications', 'readmitted'])

# notEncodedEncountersData = pd.read_csv('./cleaned_data.csv', index_col=0, names=encountersDataHeading)

encountersData = pd.read_csv('./final_data.csv', names=['age', 'time in hospital', 'num procedures', 'num medications', 'readmitted', 'race', 'gender'], index_col=0, header=None)

# encountersData = notEncodedEncountersData
encountersData['race'] = labelEncoder.fit_transform(encountersData['race'])
encountersData['gender'] = labelEncoder.fit_transform(encountersData['gender'])
encountersTargetY = encountersData['readmitted']
encountersTargetX = encountersData.drop(columns = ['readmitted'], axis=1) #remove label readmitted from column axis

"""
Data Augmentation
"""
""" SMOTE method """
# encountersTargetX, encountersTargetY = smote.fit_resample(encountersTargetX, encountersTargetY)
""" Random Oversampling method """
# encountersTargetX, encountersTargetY = overSampler.fit_resample(encountersTargetX, encountersTargetY)

# print(len(encountersTargetY[encountersTargetY == 0]))
# print(len(encountersTargetY[encountersTargetY == 1]))

# print((encountersData['readmitted'] == 0).sum()) #give me the sum of the mask that has readmitted == 0 (mask means filter)
# print(len(encountersData[encountersData['readmitted'] == 0])) #apply the mask to the dataframe and give me the length of my new dataframe

# Split the data we have into x_train y_train x_test y_test
x_train, x_test, y_train, y_test = train_test_split(encountersTargetX, encountersTargetY, random_state=42) #random_state should be a seed

x_train_scaled = x_train
x_test_scaled = x_test

# ######## Standardization
# scaler = StandardScaler()
# xScaler = scaler.fit(x_train)
# x_train_scaled = xScaler.transform(x_train)
# x_test_scaled = xScaler.transform(x_test)

######### Normalization
x_train_scaled = normalize(x_train, axis=0)
x_test_scaled = normalize(x_test, axis=0)

""" Feature Selection """
k = 5
selector = SelectKBest(score_func=chi2, k=k)
x_train_scaled = selector.fit_transform(x_train_scaled, y_train)
x_test_scaled = selector.transform(x_test_scaled)

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

