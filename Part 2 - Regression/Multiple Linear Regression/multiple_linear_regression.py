import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

dataset = pd.read_csv('./50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Treinando o modelo de Regressão Linear Míltipla a partir do 'Training Set'
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Prevendo os resultados a partir do 'Test Set'
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
y_pred_vertical = y_pred.reshape(len(y_pred), 1)
y_test_vertical = y_test.reshape(len(y_test), 1)
print(np.concatenate((y_pred_vertical, y_test_vertical), axis=1))

# Realizando predição específica:
# R&D Spend = 160000
# Administration Spend = 130000
# Marketing Spend = 300000
# State = 'California' = 1, 0, 0
# Ou seja, o modelo prevê que uma Startup nessas condições terá um lucro de [X]
print('\n-----------------------')
print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))
print('-----------------------\n')

# Definindo equação final da regressão linear
print('-----------------------')
angular_coef = regressor.coef_
linear_coef = regressor.intercept_
print(f"Profit = {angular_coef[0]:.4f} * DummyStateOne + {angular_coef[1]:.4f} * DummyStateTwo + {angular_coef[2]:.4f} * DummyStateThree + {angular_coef[3]:.4f} * R&D Spend + {angular_coef[4]:.4f} * Administration Spend + {angular_coef[5]:.4f} * Marketing Spend + {linear_coef:.4f}")

