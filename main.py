import pandas as pd
import numpy as np
import method


data = pd.read_csv("Walmart_Sales.csv").dropna()
data.drop("Store", axis=1, inplace=True)

data["Date - datetime"] = pd.to_datetime(data["Date"], format="%d-%m-%Y")
data.drop("Date", axis=1, inplace=True)

data["Quarter"] = ['']*data.shape[0]
print(data.shape)


for i in range(data.shape[0]):
    each = data.loc[i, "Date - datetime"].month
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

# 75, 25 split for data for train, test, 100 randomization

# compute coefficient for each method, compare them with table from page 82

# compute effective degrees of freedom (page 252)
# plot train error, test error, aic as function of effective dof

# bootstrap?
