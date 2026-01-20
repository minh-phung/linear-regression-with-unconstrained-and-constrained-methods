import numpy as np
import sympy as sp
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from method import least_squared

# following 3.4.1 hastie et al
# pdf page 83
# lambda (hastie) = alpha (scikit learn)
def reg(x_train, y_train, x_test, y_test):
    U, D, Vh = np.linalg.svd(x_train)

    dof_val   = np.array(range(x_train.shape[1]))+1
    lambd_val = np.zeros(dof_val.shape)

    for i in range(len(dof_val)):
        lambd_val[i] = lambd(D, dof_val[i])

    out = np.zeros((len(dof_val), x_train.shape[1]+3))

    for i in range(len(dof_val)):
        if dof_val[i] == 0:
            regression = LinearRegression(fit_intercept=False)
        else:
            regression = Ridge(alpha = lambd_val[i], fit_intercept=False)

        regression.fit(x_train, y_train)

        y_hat_train = regression.predict(x_train).reshape(-1, 1)
        train_error = sum((y_hat_train - y_train) ** 2) / len(y_train)

        y_hat_test = regression.predict(x_test).reshape(-1, 1)
        test_error = sum((y_hat_test - y_test) ** 2) / len(y_test)

        out[i, 0] = dof_val[i]
        out[i, 1] = train_error[0]
        out[i, 2] = test_error[0]
        out[i, 3:] = regression.coef_.flatten()

    return out

def lambd(D, dof):

    x = sp.symbols("x")
    expr = 0
    for i in range(D.shape[0]):
        expr += D[i]**2 / (D[i]**2 + x)
    eq = sp.Eq(expr, dof)

    result = sp.solveset(eq, x, domain=sp.Interval(0, sp.oo))

    return max(result)