#!/usr/bin/env python
# coding: utf-8

# # Predicting House Prices for Informed Real Estate Recommendations
# 
# **Author: Karina**

# ## Overview
# 
# In this project, our aim is to provide actionable insights to real estate agents, empowering them to offer informed recommendations to homeowners regarding the potential value enhancement of their properties through strategic renovations. Central to our analysis are key factors such as the number of bedrooms and floors, each of which holds significant sway over the pricing dynamics of residential properties.
# 
# Utilizing regression modeling techniques, we delve into the intricate relationships between these pivotal factors and house prices. This regression model uses predictive statistics, so throughout this project I am to improve the modelling to get a more accurate prediction. This will invlove cleaning the data, log transformation, scaling and normalisation along with other techniques to improve the model until we get an accurate outcome. 
# 
# Ultimately, this project endeavors to empower real estate professionals with the knowledge and tools necessary to navigate the complex landscape of property valuation, facilitating sound decision-making and optimizing outcomes for homeowners and buyers alike.

# In[1]:


# Import relevant packages
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf

import scipy.stats as stats

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
get_ipython().run_line_magic('matplotlib', 'inline')


# # Import, Clean and Understand the Data

# In[2]:


df = pd.read_csv ('kc_house_data.csv', index_col=0)
df.head()


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


#identifying total number of unique zipcodes to see if this data is relavent
df['zipcode'].nunique()


# In[6]:


#checking to see how many properties have a waterfront.
df['waterfront'].value_counts()


# In[7]:


df['yr_renovated'].value_counts()


# ## Removing Irrelevant columns
# 
# From looking through the above, it is clear that we can exlude several columns as they will not be relevant to this study and either will not have an influence on price or are not something within thr real estates control so would not be useful for them when we make reccomendations around what renovations would increase house prices. 
# 
# There are only 70 unique zipcodes out of 21,597 entries and all are based in the Washington State area so this would not be the best independant variable to use in order to get an accurate result from this data. In removing this, the latitude and longitude no longer are relavent. 
# 
# I have also removed the columns with a high number of NaN or 0 values including waterfront, sqft_basement & yr_ renovated
# 
# And finally, I removed columns with irrelavant data for this study including date, view (view refers to number of times property was viewed), sqft_above (no longer relavent once basement was removed).

# In[8]:


columns_to_remove = ['date', 'view', 'waterfront','yr_renovated', 'zipcode', 'lat', 'long', 'sqft_above', 'sqft_basement' ]


# In[9]:


df1 = df.drop(columns=columns_to_remove)
df1


# ## Identifying Categorical Variables

# In[10]:


fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(16,3))

for xcol, ax in zip(['bedrooms', 'bathrooms', 'floors', 'condition', 'grade'], axes):
    df1.plot(kind='scatter', x=xcol, y='price', ax=ax, alpha=0.4, color='b')


# Looking at the above return, we can see there an outlier in 'bedrooms' which needs to be removed. Outliers can effect regression lines, making the regression lines less accurate in predicting other data.

# In[11]:


ordered_df1 = df1.sort_values(by='bedrooms', ascending=False)
top_10 = ordered_df1.head(10)
top_10


# In[12]:


outlier_to_drop = 33
df1 = df1[df1['bedrooms'] != outlier_to_drop]


# In[13]:


fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(16,3))

for xcol, ax in zip(['bedrooms', 'bathrooms', 'floors', 'condition', 'grade'], axes):
    df1.plot(kind='scatter', x=xcol, y='price', ax=ax, alpha=0.4, color='b')


# In[14]:


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16,3))

for xcol, ax in zip(['sqft_living', 'sqft_lot', 'sqft_living15', 'sqft_lot15'], axes):
    df1.plot(kind='scatter', x=xcol, y='price', ax=ax, alpha=0.4, color='b')


# In[15]:


df1[['bedrooms', 'bathrooms', 'floors', 'condition', 'grade', 'yr_built' ]].nunique()


# In[16]:


df1['bedrooms'].unique()


# In[17]:


df1['bathrooms'].unique()


# In[18]:


df1['grade'].unique()


# ## Creating Dummies from Categorical Variables
# It is clear from the above scatterplots which columns are categorical and which are continuous. Dummy variables are useful because they allow us to include categorical variables in our analysis, which would otherwise be difficult to include due to their non-numeric nature. They can also help us to control for confounding factors and improve the validity of our results.

# In[19]:


dummy_variables_floors = pd.get_dummies(df1['floors'], prefix='floors')
dummy_variables_condition = pd.get_dummies(df1['condition'], prefix='condition')

df1 = pd.concat([df1, dummy_variables_floors, dummy_variables_condition], axis=1)

df1.drop(['floors', 'condition'], axis=1, inplace=True)
df1.head()


# After Checking the number of unique values in each of the categorical variables,it is clear that several of them will need to be grouped priror to creating dummies in order to make the data cleaner and not to have too many columns.
# 
# I will do this for the following.
# 
# 1. bedrooms
# 2. bathrooms
# 3. grade
# 4. yr_built

# In[20]:


# Define bins for the years
bins = [1900, 1950, 2000, 2015]

# Create labels for the bins
labels = ['1900-1949', '1950-1999', '2000-2015']

# Bin the yr_built column in df1
df1.loc[:, 'yr_built_bins'] = pd.cut(df1['yr_built'], bins=bins, labels=labels, right=False)

# Convert the binned column into dummy variables
dummy_variables = pd.get_dummies(df1['yr_built_bins'], prefix='year', drop_first=True)

# Concatenate the dummy variables with the original DataFrame df1
df1 = pd.concat([df1, dummy_variables], axis=1)

# Drop the original yr_built and binned columns if needed
df1.drop(['yr_built', 'yr_built_bins'], axis=1, inplace=True)

# Display the modified DataFrame df1
print(df1.head())


# # 

# In[21]:


bedrooms_bins = [0, 3, 6, 9, 12]  # Example: 1-2 bedrooms, 3-5 bedrooms, 6-8 bedrooms, 9-11 bedrooms
bedrooms_labels = ['1-2', '3-5', '6-8', '9-11']

# Bin the bedrooms column
df1['bedrooms_bins'] = pd.cut(df1['bedrooms'], bins=bedrooms_bins, labels=bedrooms_labels, right=False)

# Convert the binned column into dummy variables
dummy_variables_bedrooms = pd.get_dummies(df1['bedrooms_bins'], prefix='bedrooms', drop_first=True)

# Concatenate the dummy variables with the original DataFrame df1
df1 = pd.concat([df1, dummy_variables_bedrooms], axis=1)

# Drop the original bedrooms and binned columns if needed
df1.drop(['bedrooms', 'bedrooms_bins'], axis=1, inplace=True)


# In[22]:


bathrooms_bins = [0, 1, 2, 3, 4, 5, 6, 7, 8.5]  # Example: 0-1, 1-2, 2-3, 3-4, 4-5, 5-6, 6-7, 7-8, 8+
bathrooms_labels = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8']

# Bin the bathrooms column
df1['bathrooms_bins'] = pd.cut(df1['bathrooms'], bins=bathrooms_bins, labels=bathrooms_labels, right=False)

# Convert the binned column into dummy variables
dummy_variables_bathrooms = pd.get_dummies(df1['bathrooms_bins'], prefix='bathrooms')

# Concatenate the dummy variables with the original DataFrame df1
df1 = pd.concat([df1, dummy_variables_bathrooms], axis=1)

# Drop the original bathrooms and binned columns if needed
df1.drop(['bathrooms', 'bathrooms_bins'], axis=1, inplace=True)


# In[23]:


grade_bins = [2, 7, 13]  # Example: Grades 3-6, Grades 7-13
grade_labels = ['3-6', '7-13']

# Bin the grade column
df1['grade_bins'] = pd.cut(df1['grade'], bins=grade_bins, labels=grade_labels, right=False)

# Convert the binned column into dummy variables
dummy_variables_grade = pd.get_dummies(df1['grade_bins'], prefix='grade', drop_first=True)

# Concatenate the dummy variables with the original DataFrame df1
df1 = pd.concat([df1, dummy_variables_grade], axis=1)

# Drop the original grade and binned columns if needed
df1.drop(['grade', 'grade_bins'], axis=1, inplace=True)


# ## Iteration 1: Base Model 
# 

# In[24]:


outcome = 'price'
x_cols = ['sqft_living', 'sqft_lot', 'sqft_living15', 'sqft_lot15']
predictors = '+'.join(x_cols)
formula = outcome + '~' + predictors
model = ols(formula=formula, data=df1).fit()
model.summary()


# ## Base Model (Iteration 1) Take Aways
# 
# It is clear from the above Regretion Results that more work is needed to make this a usable model. The R Squared is only at 0.502 and the Prob F-statistic is showing as zero. 
# It is also indicating there may be strong multicollinearity in this.
# 

# 

# ## Identifying Multicollinearity

# In[25]:


data_pred = df1.iloc[:,1:20]
data_pred.head()


# In[26]:


data_pred.corr()


# In[27]:


abs(data_pred.corr()) > 0.75


# In[28]:


df=data_pred.corr().abs().stack().reset_index().sort_values(0, ascending=False)

# zip the variable name columns (Which were only named level_0 and level_1 by default) in a new column named "pairs"
df['pairs'] = list(zip(df.level_0, df.level_1))

# set index to pairs
df.set_index(['pairs'], inplace = True)

#d rop level columns
df.drop(columns=['level_1', 'level_0'], inplace = True)

# rename correlation column as cc rather than 0
df.columns = ['cc']

# drop duplicates. This could be dangerous if you have variables perfectly correlated with variables other than themselves.
# for the sake of exercise, kept it in.
df.drop_duplicates(inplace=True)


# In[29]:


df[(df.cc>.75) & (df.cc <1)]


# In[30]:


sns.heatmap(data_pred.corr(), center=0);


# ## Identifying Multicollinearity Take Aways
# While the above is showing several examples of multicollinearity, they are all originally from the same columns so it would be expected the data would correllate. So for the moment I wont remove these as it would remove some of the data from the original columns before they were grouped and then fixed with dummies
# 
# 

# ## Log Transformation
# Transforming non-normal features in regression modeling is essential to meet the assumption of normality, improve model performance by making relationships more linear. Overall, transforming non-normal features is a crucial preprocessing step that enhances the validity and accuracy of regression models.

# In[31]:


pd.plotting.scatter_matrix(df1[x_cols], figsize=(10,12));


# ## Transforming Non-Normal Features

# In[32]:


non_normal = ['sqft_living', 'sqft_lot', 'sqft_living15', 'sqft_lot']
for feat in non_normal:
    df1[feat] = df1[feat].map(lambda x: np.log(x))
pd.plotting.scatter_matrix(df1[x_cols], figsize=(10,12));


# ## Iteration 2: Model After Transforming Non-Normal Features

# In[36]:


outcome = 'price'
x_cols = ['sqft_living', 'sqft_lot', 'sqft_living15', 'sqft_lot15']
predictors = '+'.join(x_cols)
formula = outcome + '~' + predictors
model = ols(formula=formula, data=df1).fit()
model.summary()


# ## Scaling & Normalisation

# In[37]:


df1[['sqft_living', 'sqft_lot', 'sqft_living15', 'sqft_lot15']].hist(figsize  = [6, 6]); 


# In[39]:


data_log = pd.DataFrame([])
data_log['logliv'] = np.log(data_pred['sqft_living'])
data_log['loglot'] = np.log(data_pred['sqft_lot'])
data_log['logliv15'] = np.log(data_pred['sqft_living15'])
data_log['loglot15'] = np.log(data_pred['sqft_lot15'])
data_log.hist(figsize  = [6, 6]);


# There is a clear difference after the scaling, the data has improved significantly. The ditribution is now a very good shape and the skewness is now gone.

# ## Iteration 2: Model After Transforming Non-Normal Features

# In[44]:


outcome = 'price'
x_cols = ['sqft_living', 'sqft_lot', 'sqft_living15', 'sqft_lot15']
predictors = '+'.join(x_cols)
formula = outcome + '~' + predictors
model = ols(formula=formula, data=df1).fit()
model.summary()


# The R-Squared Value has gone down since the Log transformation however we can see the sqft_lot p value has gone down & there is no longer strong correlation that would be affecting the model.

# In[42]:





# In[ ]:




