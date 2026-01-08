import numpy as np
import sympy as sp

# following 3.4.1 hastie et al
# pdf page 83
def reg(x_train, y_train, x_test, y_test):
    U, D, Vh = np.linalg.svd(x_train)

    dof_val   = np.array(range(x_train.shape[1]))+1
    lambd_val = np.zeros(dof_val.shape)

    for i in range(len(dof_val)):
        lambd_val[i] = lambd(D, dof_val[i])

    print(np.vstack((dof_val, lambd_val)))

    return

def lambd(D, dof):

    x = sp.symbols("x")
    expr = 0
    for i in range(D.shape[0]):
        expr += D[i]**2 / (D[i]**2 + x)
    eq = sp.Eq(expr, dof)

    result = sp.solveset(eq, x, domain=sp.Interval(0, sp.oo))

    return max(result)