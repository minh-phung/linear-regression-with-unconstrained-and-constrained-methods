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
#----------------------------------------------------------------------------------------------------------

linear_regression_data = pd.DataFrame(index = range(n), columns = predictor)

# ---------------------------------------------------------------------------
# k-fold validation instead (?)

# 75, 25 split for data for train, test
index_list = range(data.shape[0])
frac = int(0.75 * len(index_list))

train_index = np.array(random.sample(index_list, frac))
test_index = np.delete(index_list, train_index)

print(train_index, test_index)
# ---------------------------------------------------------------------------
# compute coefficient for each method
# method(train_index, test_index, predictor, target)
# return(coefficient, train_error, test_error, dof)
# effective dof page 252


#----------------------------------------------------------------------------------------------------------
# compare them with table from page 82
# train_error, test_error comparison


