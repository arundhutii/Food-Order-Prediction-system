#!/usr/bin/env python
# coding: utf-8

# # Demand Forecasting
# 
# # CSE Batch 1- 
# [ Arundhuti Sen, Arko Chatterjee, Debasmita Raha, Deepankar Mehta ]

# # Introduction to Daily Demand Forecasting Model For Restaurant Orders
# This project involves creating a machine learning model to predict daily demand based on past order data. 
# By forecasting future orders, businesses can optimize inventory, reduce waste, and better meet customer needs. 
# 
# The project demonstrates how simple machine learning techniques like linear regression or time series analysis can improve operational efficiency.

# In[25]:


# importing intrensic libraries 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")


# # Reading the dataset 
# This test dataset contains the historical sales data.
# With the help of this data we will perform our exploitary data analysis followed by immplementation of machine learning model.

# In[2]:


df = pd.read_csv("Daily Demand Forecasting Orders.csv")
df.head()


# # Columns had to be renamed because from the dataset we cannot infer which column represented what. 

# In[3]:


df.rename(columns={'Week of the month (first week, second, third, fourth or fifth week':'Week','Day of the week (Monday to Friday)':'Day'}, inplace=True)
df.columns


# # Now we will perform basic data analysis to find out what is in store of the dataset

# In[4]:


df.shape
df.info()


# # Describing the statistics of the dataset  

# In[5]:


df.describe()


# # Pairplot analysis 
# This plot is done to find which data is related to each other and it returns which data is forming clusters when correlated.

# In[6]:


sns.pairplot(df)


# # Correlation and Heatmap
# Corr() is used to find the correlation score between each column.
# And then Heatmap is plotted using the correlated data.
# 

# In[7]:


plt.figure(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True)


# # Implementation of ML Algorithm
# For that we cannot take all the columns . We have to perform the operation on few related columns which we can do with the info from the above created Heatmap

# In[8]:


df.columns


# # 
# Selecting specific columns that are most relevant for modeling
# This step involves choosing features that are expected to have a significant impact on the target variable
# 

# In[9]:


df1 = df[["Urgent order", 'Order type A', 'Order type B', 'Banking orders (1)', "Target (Total orders)"]].copy()


# # 
# Displaying the first few rows of the selected columns to ensure correct selection

# In[10]:


df1.head()


# #
# Visualizing pairwise relationships with a Kernel Density Estimate (KDE) on the diagonal
# #
# KDE provides a smoothed estimate of the distribution of data, helping to understand its underlying structure

# In[11]:


sns.pairplot(df1, diag_kind='kde')


# # 
# Checking the distribution of the 'Week' feature
# Understanding how data is distributed across different weeks can help in feature engineering

# In[12]:


df.Week.value_counts


# #
# Visualizing the distribution of data across different weeks
# Countplot shows the frequency of occurrences for each category in the 'Week' feature

# In[13]:


sns.countplot(df.Week)


# # 
# Visualizing the distribution of data across different days
# Similar to the 'Week' feature, this helps understand how demand varies by day

# In[14]:


sns.countplot(df.Day)


# # 
# Applying one-hot encoding to categorical variables ('Week' and 'Day')
# 
# One hot encoding is a technique that we use to represent categorical variables as numerical values in a machine learning model.

# In[15]:


#One hot encoding

df1 = pd.get_dummies(df, columns=["Week", "Day"])
df1.head()


# # 
# Checking the shape of the transformed dataset
# #
# After one-hot encoding, the dataset typically increases in the number of columns

# In[16]:


df1.shape


# # 
# Splitting the dataset into features (X) and target variable (y)
# The target variable is what we want to predict, and features are the inputs that will be used to make predictions

# In[17]:


#splitting of train-test data

y = df1["Target (Total orders)"].copy() # Target variable
y.head()


# In[18]:


X = df1.drop("Target (Total orders)", axis=1)  # Features (all other columns)
X.head()


# # 
# Splitting the dataset into training and testing sets with a 67-33 split. 
# The training set is used to train the model, while the test set is used to evaluate its performance

# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=51)

print(len(X_train))
print(len(X_test))


# # 
# Scaling the features using StandardScaler for better model performance
# 
# Scaling standardizes the features to have a mean of 0 and a standard deviation of 1, which helps in faster convergence and better performance of the model

# In[20]:


#Scaling

scaler = StandardScaler()


# # 
# Fitting the scaler on the training data and transforming both training and testing data
# 
# It's important to fit the scaler only on the training data to avoid data leakage

# In[21]:


X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# # 
# Printing the R² score of the model on the test data
# R² (coefficient of determination) indicates how well the model explains the variability of the target variable
# #
# 
# Printing the coefficients of the features
# The coefficients indicate the strength and direction of the relationship between each feature and the target variable
# #
# 
# Printing the shape of the coefficients array
# This shows how many coefficients (one per feature) were calculated
# #
# 
# Printing the intercept of the model
# The intercept represents the expected value of the target variable when all features are zero
# #
# 
# Calculating and printing the Mean Squared Error (MSE) of the model's predictions
# MSE is a measure of the average squared difference between the actual and predicted values; lower values indicate better performance
# 

# In[22]:


#Linear Regression

reg = LinearRegression().fit(X_train, y_train)
print(reg.score(X_test, y_test))
print(reg.coef_)
print(reg.coef_.shape)
print(reg.intercept_)
print(mean_squared_error(y_test, reg.predict(X_test)))


# # Hence, we can see that we've got 0.9151813551083667 
# # => 91% as our model accuracy score which means our model is 91% accurate.  

# # 
# Printing the predicted values for the test set
# These are the demand forecasts generated by the model based on the test data

# In[23]:


print(reg.predict(X_test))


# # Comparison of Actual vs Predicted Values with Linear Regression Line for Training and Test Data
# 
# 
# 
# 
# 
# 
# 
# 

# In[33]:


# Predicting values for the entire dataset using the trained model
y_pred_train = reg.predict(X_train)
y_pred_test = reg.predict(X_test)

# Plotting the training data and the regression line
plt.figure(figsize=(12, 6))
plt.scatter(y_train, y_pred_train, color='blue', label='Train Data')
plt.scatter(y_test, y_pred_test, color='green', label='Test Data')

# Plotting the linear regression line
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='-', label='Regression Line')

plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Linear Regression Line with Actual vs Predicted')
plt.legend()
plt.show()


# # Interpolated Data and Extrapolated Data with Linear Regression Line
# 

# In[35]:


# Define the range of interpolation
min_train, max_train = y_train.min(), y_train.max()

# Identifying interpolated and extrapolated data points
interp_indices = (y_test >= min_train) & (y_test <= max_train)
extrap_indices = interp_indices

# Plotting interpolated data
plt.figure(figsize=(12, 6))
plt.scatter(y_test[interp_indices], y_pred_test[interp_indices], color='blue', label='Interpolation')
plt.plot([min_train, max_train], [min_train, max_train], color='red', linestyle='-', label='Regression Line')

plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Interpolated Data with Linear Regression Line')
plt.legend()
plt.show()

# Plotting extrapolated data
plt.figure(figsize=(12, 6))
plt.scatter(y_test[extrap_indices], y_pred_test[extrap_indices], color='green', label='Extrapolation')
plt.plot([min(y_test[extrap_indices]), max(y_test[extrap_indices])], 
         [min(y_pred_test[extrap_indices]), max(y_pred_test[extrap_indices])], 
         color='red', linestyle='-', label='Regression Line')

plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Extrapolated Data with Linear Regression Line')
plt.legend()
plt.show()


# # Predicted vs Actual Values with Linear Regression Line 

# In[27]:


# Predict the values using the test data
y_pred = reg.predict(X_test)

# Plotting the actual vs predicted values as a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')

# Plotting the regression line
# Fit a line to the predicted vs. actual data points
m, b = np.polyfit(y_test, y_pred, 1)
plt.plot(y_test, m * y_test + b, color='red', label='Regression Line')

# Adding labels and title
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Linear Regression Line for Predicted vs. Actual Values')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




