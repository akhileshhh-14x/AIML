# pip install tensorflow
from tensorflow import keras
# pip install numpy
import numpy as np
# pip install pandas
import pandas as pd

n = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])

df = pd.DataFrame(n, columns = ['X1','X2','y'])

y = df.y
X = df = df.drop('y', axis=1)

model = keras.Sequential([
    keras.layers.Dense(4, activation='sigmoid'),
    keras.layers.Dense(4, activation='sigmoid'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=1000)

weights = model.get_weights()
print(weights)
