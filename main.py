import pandas as pd
import numpy as np
import method
import random


data = pd.read_csv("Walmart_Sales.csv").dropna()
data.drop("Store", axis=1, inplace=True)

data["Date - datetime"] = pd.to_datetime(data["Date"], format="%d-%m-%Y")
data.drop("Date", axis=1, inplace=True)

data["Quarter"] = ['']*data.shape[0]
print(data.shape)


for i in range(data.shape[0]):
    each = data.loc[i, "Date - datetime"].month
    out = ""
    if each <= 3:
        out = "1st"
    elif each <= 6:
        out = "2st"
    elif each <= 9:
        out = "3st"
    else :
        out = "4th"
    data.loc[i, "Quarter"] = out


print(data.head())
print(data.columns)

predictor = ['Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI','Unemployment', 'Quarter']
target    = ['Weekly_Sales']

print("---------------------------------------------")

# #----------------------------------------------------------------------------------------------------------
# k-fold validation - equal stratum for categorical predictors (holiday flag, quarter)



# #----------------------------------------------------------------------------------------------------------
# for each k-fold
# method(train_predict, train_target, test_predict, test_target)

# least_squared
# return(dof, test_error, coefficient)
# row - dof: 1

# all_subset
# return(dof, test_error, coefficient)
# row - dof: 1, 2, .., num(predictor)

# ridge, lasso
# return(dof, test_error, coefficient)
# row - dof: discrete dof (step size tbd) (continuous?)

# pcr
# return(dof, test_error, coefficient)
# row - dof: 1, 2, .., num(predictor)


#----------------------------------------------------------------------------------------------------------
# match by dof, avg test_error - cv_error, avg_coef
# compare them with table from page 82 - least dof within 1 sd of minimum per method (all except least_squared)


# plot dof vs cv_error for all method (with the chosen dof for each method)
# select best model also by using least dof within 1 sd of minimum


