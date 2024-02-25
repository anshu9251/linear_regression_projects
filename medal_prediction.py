import pandas as pd

file_path = r"D:\teams.csv" 
delimiter = "," 

teams= pd.read_csv(file_path, delimiter=delimiter)
#print(data)
teams = teams[["team","country","year","athletes","age","prev_medals","medals"]]
#print(teams)

#corr = teams.corr()["medals"]
import matplotlib.pyplot as plt
import seaborn as sns
sns.lmplot(x = "athletes",y = "medals",data = teams,fit_reg=True,ci=None)
#sns.displot(teams["athletes"])
#plt.show()

teams[teams.isnull().any(axis=1)]
#removing all the null values
teams = teams.dropna()
#print(teams)

train = teams[teams["year"]<2012].copy()
test = teams[teams["year"]>=2012].copy()

print(train.shape)
print(test.shape)

from sklearn.linear_model import LinearRegression

predcitors = ["athletes","prev_medals"]
target = "medals"

reg = LinearRegression()

reg.fit(train[predcitors],train["medals"])

predictions = reg.predict(test[predcitors])

#print(predictions)

#rescaling the prediction:-
# adding the column here 

test["predictions"] = predictions
print(test)

test.loc[test["predictions"]<0,"predictions"]=0
test["predictions"] = test["predictions"].round()

#print(test)

from sklearn.metrics import mean_absolute_error

err = mean_absolute_error(test["medals"],test["predictions"])
print(err)

dsc = test.describe()["medals"]
#print(dsc)

usa_data = test[test["team"]=="USA"]
#print(usa_data)

ind_data = test[test["team"]=="IND"]
#print(ind_data)

errors = (test["medals"]-test["predictions"]).abs()
print(errors)

error_by_team = errors.groupby([test["team"]]).mean()
print(error_by_team)

medal_by_team = test["medals"].groupby([test["team"]]).mean()

error_ratio = error_by_team/medal_by_team
#print(error_ratio)

print(error_ratio[~pd.isnull(error_ratio)])

import numpy as np
error_ratio = error_ratio[np.isfinite(error_ratio)]
print(error_ratio)

print(error_ratio.sort_values())

france_data = test[test["team"]=="FRA"]\

print(france_data)
# Plotting actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(test["medals"], test["predictions"], color='blue', label='Actual vs Predicted Medals')
plt.plot(test["medals"], test["medals"], color='red', linestyle='--', label='Perfect Prediction')

# Adding labels and title
plt.title('Actual vs Predicted Medals')
plt.xlabel('Actual Medals')
plt.ylabel('Predicted Medals')
plt.legend()
plt.grid(True)

# Show plot
plt.show()
