"""
David Edwards
CSC525 - Principles of Machine Learning
Module 3 - Critical Thinking - Option 2
Dr. Issac Gang
8/7/2022
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# read in the data
df = pd.read_csv("position_salaries.csv")

# We're interested in the years experience as the pertinent Feature
X = df.iloc[:, 1:2].values
# Salary will be our target
y = df.iloc[:, 2].values

# Fitting Polynomial Regression to the dataset
# I used degrees of 4 because experimenting in my Jupyter notebook made anything less look not well-fit
# I wanted to use the minimum fit level that seemed to match the data.
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y)

# Get the predicted years of experience
while True:
    try:
        experience = int(input("Enter the years experience: "))
    except ValueError:
        print("Invalid Years of experience")
        continue
    else:
        break

# The prediction requires the feature in a 2d array
salary = pol_reg.predict(poly_reg.fit_transform([[experience]]))
# salary returned in an array, we just want the value
salary = salary[0]
print("The predicted Salary is: ${:.2f}".format(salary))
