import pandas as pd

file_path = "D:\Salary_dataset.csv"

data = pd.read_csv(file_path,delimiter=",")

print(data.shape)
#print(data)

data.drop(columns=["Unnamed: 0"],inplace=True)
print(data)


x = data["YearsExperience"].values.reshape(-1,1)
y = data["Salary"]
#creating plonomial features

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=3)
x_poly = poly.fit_transform(x)


import matplotlib.pyplot as plt
import seaborn as sns

#sns.lmplot(x = "YearsExperience",y = "Salary",data=data,fit_reg=True,ci=None)
#plt.show()

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x_poly,y,test_size=0.2,random_state=42)

#training our model
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(x_train,y_train)

#predict on the test set
y_predict = reg.predict(x_test)

from sklearn.metrics import mean_squared_error

err = mean_squared_error(y_test,y_predict)
print("Mean Squared Error (Polynomial Regression - Degree 2):", err)

comparison = pd.DataFrame({"Actual": y_test, "Predicted": y_predict})
print(comparison.head())

plt.figure(figsize=(10, 6))
plt.scatter(x_test[:,1],y_test,c= "r",label = "actual")
plt.scatter(x_test[:,1],y_predict,c="b",label="predicted")

plt.title("Actual vs. Predicted Salary (Polynomial Regression - Degree 2)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.show()


