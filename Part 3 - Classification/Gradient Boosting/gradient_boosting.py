import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Carregamento do dataset
dataset = pd.read_csv('./Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Separando o dataset em 'Training Set' e 'Test Set'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Realizando 'Feature-Scaling' nas colunas de valores numéricos
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Treinando o modelo de Gradient Boosting a partir do 'Training Set'
classifier = GradientBoostingClassifier(n_estimators=10, random_state=0)
classifier.fit(X_train, y_train)

# Comparando os resultados no 'Test Set' com os valores previstos pelo modelo
print("\n---------------------------------")
y_pred = classifier.predict(X_test)
y_pred_vertical = y_pred.reshape(len(y_pred), 1)
y_test_vertical = y_test.reshape(len(y_test), 1)
print(np.concatenate((y_pred_vertical, y_test_vertical), axis=1))
print("---------------------------------")

# Computando a precisão da predição feita pelo modelo
print("---------------------------------")
print(f"Matriz: {confusion_matrix(y_test, y_pred)}")
print(f"Precisão: {accuracy_score(y_test, y_pred)}")
print("---------------------------------")

# Prevendo um resultado para um cliente de 30 anos com salário estimado de $87.000
print("---------------------------------")
print(classifier.predict(sc.transform([[30, 87000]])))
print("---------------------------------\n")