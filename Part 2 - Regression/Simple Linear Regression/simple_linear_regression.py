import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Treinando o modelo de Regressão Linear Simples no 'Training Set'
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Prevendo os resultados do 'Test Set'
y_pred = regressor.predict(X_test)

# Visualizando os resultados do 'Training Set'
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary x Experience (TRAINING SET)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualizando os resultados do 'Test Set'
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary x Experience (TEST SET)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Realizando previsão de salário para funcionário com [[x]] anos de experiência
print('\n-----------------------')
print(regressor.predict([[12]]))
print('-----------------------\n')

# Definindo equação final da regressão linear (Salário = regressor.coef_ * anosDeExperiencia + regressor.intercept_)
print('-----------------------')
angular_coef = regressor.coef_
linear_coef = regressor.intercept_
print(f"Salary = {angular_coef} * yearsOfExperience + {linear_coef}")
print(f"y = {angular_coef}x + {linear_coef}")
print('-----------------------')

