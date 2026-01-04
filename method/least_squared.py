import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def reg(x_train, y_train, x_test, y_test):
    print("least squared")

    regression = LinearRegression(fit_intercept=False).fit(x_train, y_train)

    y_hat_train = regression.predict(x_train)
    train_error = sum(y_hat_train - y_train) / len(y_train)

    y_hat_test = regression.predict(x_test)
    test_error = sum(y_hat_test - y_test) / len(y_test)

    coef = np.append(regression.intercept_, regression.coef_)

    print(coef)



    return

def dof(y_hat, y):

    covariance = sum( (y_hat - np.mean(y_hat))*(y - np.mean(y)) )/len(y)
    print(covariance / np.var(y_hat - y))

    return