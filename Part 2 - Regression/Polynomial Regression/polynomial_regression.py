import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv('./Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Treinando um modelo de Regressão Linear Polinomial
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)
regressor = LinearRegression()
regressor.fit(X_poly, y)

# Treinando um modelo de Regressão Linear Simples
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Plotando o resultado caso realizássemos uma Regressão Linear Simples
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='green')
plt.title('Level x Salary (SIMPLE LINEAR REGRESSION)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

# Plotando o resultado caso realizássemos uma Regressão Linear Polinomial
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X_poly), color='blue')
plt.title('Level x Salary (POLYNOMIAL LINEAR REGRESSION)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

# Realizando previsão de salário de uma pessoa de nível [[X]] na Regressão Linear Simples
print(lin_reg.predict([[6.5]]))

# Realizando previsão de salário de uma pessoa de nível [[X^0, X^1, X^2, ..., X^n]] na Regressão Linear Polinomial
print(regressor.predict([[6.5**0, 6.5**1, 6.5**2]]))
