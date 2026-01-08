import pandas as pd
import numpy as np
import method
from sklearn.model_selection import StratifiedKFold


data = pd.read_csv("Walmart_Sales.csv").dropna()
data.drop("Store", axis=1, inplace=True)

data["Date - datetime"] = pd.to_datetime(data["Date"], format="%d-%m-%Y")
data.drop("Date", axis=1, inplace=True)

data["Quarter"] = ['']*data.shape[0]
data['Quarter 1'] = np.zeros(data.shape[0])
data['Quarter 2'] = np.zeros(data.shape[0])
data['Quarter 3'] = np.zeros(data.shape[0])
data['Quarter 4'] = np.zeros(data.shape[0])

print(data.shape)

for i in range(data.shape[0]):
    each = data.loc[i, "Date - datetime"].month
    out = ""
    if each <= 3:
        out = "1st"
        data.loc[i,'Quarter 1'] = 1
    elif each <= 6:
        out = "2st"
        data.loc[i,'Quarter 2'] = 1
    elif each <= 9:
        out = "3st"
        data.loc[i,'Quarter 3'] = 1
    else :
        out = "4th"
        data.loc[i,'Quarter 4'] = 1
    data.loc[i, "Quarter"] = out

data['Intercept'] = np.ones(data.shape[0])

print(data.head())
print(data.columns)

predictor = ['Intercept', 'Holiday_Flag', 'Temperature',
             'Fuel_Price', 'CPI', 'Unemployment',
             'Quarter 1', 'Quarter 2', 'Quarter 3', 'Quarter 4']

target    = ['Weekly_Sales']

data_predictor = np.array(data[predictor])
data_target    = np.array(data[target])

# #----------------------------------------------------------------------------------------------------------
return_field = ["fold", "dof", "train_error", "test_error"]
return_field = np.concatenate((return_field, predictor))

# k-fold validation - equal stratum for categorical predictor (quarter)
k = 5

folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)
strata = data["Quarter"]

#------------------------------------------------------------------------------------
least_squared_result = pd.DataFrame(np.nan, index = range(k), columns= return_field)



for i, (train_index, test_index) in enumerate(folds.split(data, strata)):
    print("-------------------------------------------")
    print("fold ", i)
    x_train, x_test = data_predictor[train_index], data_predictor[test_index]
    y_train, y_test = data_target[train_index],    data_target[test_index]

    #least_squared_result.loc[i] = np.append(i, method.least_squared.reg(x_train, y_train, x_test, y_test))
    method.all_subset.reg(x_train, y_train, x_test, y_test)



print("-------------------------------------------")



# #----------------------------------------------------------------------------------------------------------
# for each k-fold
# method(train_predict, train_target, test_predict, test_target)

# least_squared
# return(dof, train_error, test_error, coefficient)
# row - dof: 1

# all_subset
# return(dof, train_error, test_error, coefficient)
# row - dof: 1, 2, .., num(predictor)

# ridge, lasso
# return(dof, train_error, test_error, coefficient)
# row - dof: discrete dof (step size tbd) (continuous?)

# pcr
# return(dof, train_error, test_error, coefficient)
# row - dof: 1, 2, .., num(predictor)


#----------------------------------------------------------------------------------------------------------
# match by dof, avg test_error - cv_error, avg_coef
# compare them with table from page 82 - least dof within 1 sd of minimum per method (all except least_squared)


# plot dof vs cv_error for all method (with the chosen dof for each method)
# select best model also by using least dof within 1 sd of minimum

