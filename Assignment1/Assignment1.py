#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


# # Loading the dataset

# In[33]:


df = pd.read_csv("C:/Users/Poghosyan_Syuzanna/Desktop/SyuzannaPoghosyan/morg-2014-emp.csv", low_memory=False)
df


# # Filtering for Financial Specialists as I have been one for years (Occupation Code: from 0800 to 0950)

# In[34]:


df = df[(df['occ2012'] >= 800) & (df['occ2012'] <= 950)].copy()
df


# # Computing earnings per hour

# In[35]:


df['earnings_per_hour'] = df['earnwke'] / df['uhours']


# # Data Cleaning

# In[43]:


print(df.columns)
df = df[['earnwke', 'uhours', 'marital', 'earnings_per_hour', 'grade92', 'age', 'sex', 'race', 'class']].dropna()

df


# In[44]:


# Defining the dependent variable (earnings per hour)
y = df['earnings_per_hour']


# In[45]:


# Defining potential predicting variables (x)
predictors = ['grade92', 'age', 'sex', 'race', 'marital', 'class']


# In[46]:


columns = ['earnings_per_hour']

plt.figure(figsize=(14, 10))

for i, column in enumerate(columns, 1):
    plt.subplot(3, 5, i) 
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot of {column}')

plt.tight_layout()
plt.show()


# In[47]:


columns_with_outliers = ['earnings_per_hour']

def remove_outliers(df, columns):
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

df = remove_outliers(df, columns_with_outliers)

df


# # Analyzing the dataset

# In[48]:


plt.figure(figsize=(10, 6))
sns.histplot(y, kde=True, color='red')
plt.title('Distribution of Earnings per Hour')
plt.xlabel('Earnings per Hour')
plt.ylabel('Frequency')
plt.show()


# In[51]:


# Correlation between the dependent variable and the potential predictors
correlation_matrix = df[['grade92', 'age', 'sex', 'race', 'marital', 'class', 'earnings_per_hour']].corr()
                
# Visualizing with a correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# In[52]:


round(df[['grade92', 'age', 'sex', 'race', 'class', 'marital', 'earnings_per_hour']].describe(), 3)


# # Defining all the models

# In[53]:


# Model features for each of 4 models
model_features = [
    ['age'],
    ['age', 'sex', 'race'],
    ['age', 'sex', 'race', 'grade92'],
    ['age', 'sex', 'race', 'grade92', 'marital', 'class']  # Most complex model with all predictors
]


# In[59]:


from sklearn.model_selection import KFold, cross_val_score

def preprocess_df(df, features):
    X = pd.get_dummies(df[features], drop_first=True)
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.apply(lambda x: x.astype(int) if x.dtype == 'bool' else x)
    return sm.add_constant(X)

results = []
plt.figure(figsize=(10, 8))

# Going through the models and computing some performance metrics
for i, features in enumerate(model_features):
    X_subset = preprocess_df(df, features)
    y = pd.to_numeric(df['earnings_per_hour'], errors='coerce')
    
    # Fitting OLS model
    model = sm.OLS(y, X_subset).fit()
    y_pred = model.predict(X_subset)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    bic = model.bic
        
    # Cross-validation of RMSE
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    lin_reg = LinearRegression()
    cv_rmse = np.sqrt(-cross_val_score(lin_reg, X_subset, y, cv=kf, scoring='neg_mean_squared_error').mean())
   
    results.append([f'Model {i+1}', rmse, cv_rmse, model.bic])
    
    # Plotting the residuals
    plt.subplot(2, 2, i+1)
    sns.scatterplot(x=y_pred, y=(y - y_pred))
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'Predicted values and residuals for Model {i+1}')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')

plt.tight_layout()
plt.show()


    


# In[60]:


results_df = pd.DataFrame(results, columns=['Model', 'RMSE', 'CV RMSE', 'BIC'])
print(results_df)


# In[61]:


# Creating Dataframe from results
metrics_df = pd.DataFrame(results, columns=['Model', 'RMSE', 'CV RMSE', 'BIC'])

# Defining metric names and labels
metrics = ['RMSE', 'CV RMSE', 'BIC']
titles = ['RMSE for Different Models', 'Cross-validated RMSE for Different Models', 'BIC for Different Models']

# Plotting the metrics
plt.figure(figsize=(12, 5))

for i, (metric, title) in enumerate(zip(metrics, titles), 1):
    plt.subplot(1, 3, i)
    sns.lineplot(x=metrics_df['Model'], y=metrics_df[metric], marker='o')
    plt.title(title)
    plt.xlabel('Model')
    plt.ylabel(metric)

plt.tight_layout()
plt.show()

