# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Initialize weights randomly.

2.Compute predicted values.

3.Compute gradient of loss function.

4.Update weights using gradient descent.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: PULI NAGA NEERAJ
RegisterNumber:  212223240130
*/
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    # Add a column of ones to X for the intercept term 
  X = np.c_[np.ones(len(X1)), X1]
  # Initialize theta with zeros
  theta = np.zeros(X.shape[1]).reshape(-1,1)
  for _ in range(num_iters):
    predictions = (X).dot(theta).reshape(-1, 1)
    errors = (predictions - y).reshape(-1,1)
    theta -= learning_rate* (1 / len(X1)) * X.T.dot(errors)
  return theta

data = pd.read_csv('50_Startups.csv', header=None)
print(data.head())
# Assuming the last column is your target variable 'y' and the preceding column 
X = (data.iloc[1:, :-2].values)
print(X)
X1=X.astype(float)
scaler = StandardScaler()
y = (data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)
theta = linear_regression(X1_Scaled, Y1_Scaled)

# Predict target value for a new data point
new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction = np.dot(np.append(1, new_Scaled), theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```

## Output:
![ML-3 1](https://github.com/PuliNagaNeeraj/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/138849173/3c1e3a87-6017-4a53-bda0-52bbeb109620)

![ML-3 2](https://github.com/PuliNagaNeeraj/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/138849173/e9671b95-4e38-4897-9739-496df6115d76)

![ML-3 3](https://github.com/PuliNagaNeeraj/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/138849173/77c98f25-e0ef-4d70-8bb5-b5400528b049)

![ML-3 4](https://github.com/PuliNagaNeeraj/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/138849173/02b6b513-85b0-4658-9a29-7aec7a4e5ef1)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
