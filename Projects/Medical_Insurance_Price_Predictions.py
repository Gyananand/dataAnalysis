#!/usr/bin/env python
# coding: utf-8

# # Medical_Insurance_Price_Prediction
# 

# ### Data Preprocessing

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# In[4]:


df = pd.read_csv('insurance.csv')
df.head()


# In[5]:


df.isnull().sum()


# In[6]:


df.info()


# In[7]:


df.describe()


# ### Exploratory Data Analysis (EDA)

# In[9]:


import matplotlib.pyplot as plt

features = ['sex', 'smoker', 'region']

plt.subplots(figsize=(20, 10))
for i, col in enumerate(features):
	plt.subplot(1, 3, i + 1)

	x = df[col].value_counts()
	plt.pie(x.values,
			labels=x.index,
			autopct='%1.1f%%')

plt.show()


# In[10]:


sns.boxplot(df['age'])


# In[11]:


sns.boxplot(df['bmi'])


# In[12]:


# Calculate the first, second, third quartile of the 'bmi' column
Q1=df['bmi'].quantile(0.25)
Q2=df['bmi'].quantile(0.50)
Q3=df['bmi'].quantile(0.75)

# Calculate the Interquartile Range (IQR) as the difference between Q3 and Q1
IQR=Q3-Q1 

# Define the lower bound and upper bound for outliers (any value this is considered an outlier)
lower_bound = Q1-1.5*IQR
upper_bound = Q3+1.5*IQR
print(lower_bound)
print(upper_bound)


# In[13]:


# Filter out outliers based on the lower and upper bounds
df = df[(df['bmi'] >= lower_bound) & (df['bmi'] <= upper_bound)]

# Check the shape of the new DataFrame to see how many rows remain
print(df.shape)

# Check again for outliers
sns.boxplot(df['bmi'])


# In[14]:


from sklearn.preprocessing import LabelEncoder

# Label encoding for 'sex' and 'smoker'
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])  # female=0, male=1
df['smoker'] = le.fit_transform(df['smoker'])  # no=0, yes=1

# One-hot encoding for 'region'
df = pd.get_dummies(df, columns=['region'], drop_first=True)

# df = df.replace({True: 1, False: 0})
df.head()


# In[15]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize = (11,7.5))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()


# In[16]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['age', 'bmi', 'children']] = scaler.fit_transform(df[['age', 'bmi', 'children']])


# ### Split Data into Features and Target

# In[18]:


x = df.drop('charges', axis = 1)
x.head()


# In[19]:


y = df['charges']
y.head()


# In[20]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# ### Model Building

# In[22]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

# Define a dictionary with the models you want to try
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Support Vector Regressor": SVR(),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

results = {}

# Loop over the models
for name, model in models.items():
    # Train the model
    model.fit(x_train, y_train)
    
    # Predictions for test set
    y_pred_test = model.predict(x_test)
    
    # Predictions for train set (to calculate train accuracy)
    y_pred_train = model.predict(x_train)
    
    # Calculate performance metrics
    mae = mean_absolute_error(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)  # Train accuracy
    r2_test = r2_score(y_test, y_pred_test)  # Test accuracy
    
    # Cross-Validation Score
    cv_score = np.mean(cross_val_score(model, x, y, cv=5, scoring='r2'))  # Average R² score from CV
    
    # Store the results
    results[name] = {
        "Model": name,
        "Train Accuracy (R²)": r2_train,
        "Test Accuracy (R²)": r2_test,
        "MAE": mae,
        "MSE": mse,
        "CV Score (R²)": cv_score
    }

# Print out the performance of each model
for model_name, metrics in results.items():
    print(f"{model_name} Performance:")
    print(f"  MAE: {metrics['MAE']}")
    print(f"  MSE: {metrics['MSE']}")
    print(f"  Train Accuracy: {metrics['Train Accuracy (R²)']}")
    print(f"  Test Accuracy: {metrics['Test Accuracy (R²)']}")
    print(f"  CV Score (R²): {metrics['CV Score (R²)']}\n")


# In[23]:


model.score(x_test,y_test)


# In[26]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# Example data (replace with your actual dataset)
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
gbr = GradientBoostingRegressor()

# Define the hyperparameters to tune
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Setup RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=gbr, 
                                   param_distributions=param_grid, 
                                   n_iter=50,  # Number of combinations to try
                                   cv=5,       # Cross-validation folds
                                   verbose=2,  
                                   n_jobs=-1,  # Use all available cores
                                   random_state=42)

# Fit the RandomizedSearchCV on the training data
random_search.fit(x_train, y_train)

# Get the best parameters and the best estimator
print("Best Parameters:", random_search.best_params_)
best_gbr = random_search.best_estimator_

# Evaluate the performance on the test set
y_pred = best_gbr.predict(x_test)

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")


# ### Model Evaluation

# In[28]:


# Calculate metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))


# In[30]:


# Assuming 'model' is your trained model

# Collecting user input
age = int(input("Enter age: "))
sex = int(input("Enter sex (0 for female, 1 for male): "))
bmi = float(input("Enter BMI: "))
children = int(input("Enter number of children: "))
smoker = int(input("Are you a smoker? (0 for No, 1 for Yes): "))

# For region, create a simple selection process
print("Select region:")
print("1. Northwest")
print("2. Southeast")
print("3. Southwest")

region = int(input("Enter the number corresponding to the region: "))

# Initialize region features
region_northwest = 0
region_southeast = 0
region_southwest = 0

# Assign the value to the appropriate region based on user input
if region == 1:
    region_northwest = 1
elif region == 2:
    region_southeast = 1
elif region == 3:
    region_southwest = 1

# Prepare input for prediction
input_data = [[age, sex, bmi, children, smoker, region_northwest, region_southeast, region_southwest]]

# Make prediction
prediction = model.predict(input_data)

# Output the predicted charges
print(f"Predicted medical insurance charges: Rs{prediction[0]}")


# In[32]:


from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# 'subsample': 1.0, 'n_estimators': 200, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'log2', 'max_depth': 3, 'learning_rate': 0.05}
# Example data (replace with your actual data)
X, y = make_regression(n_samples=100, n_features=4, noise=0.2, random_state=42)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree model
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)

# Train Random Forest model
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_dt = dt_model.predict(X_test)  # Decision Tree predictions
y_pred_rf = rf_model.predict(X_test)  # Random Forest predictions

# Calculate R² score
r2_dt = r2_score(y_test, y_pred_dt)
r2_rf = r2_score(y_test, y_pred_rf)

# Print the R² scores
print(f"Decision Tree R² Score: {r2_dt}")
print(f"Random Forest R² Score: {r2_rf}")

# Plot the comparison of R² scores
models = ['Decision Tree', 'Random Forest']
r2_scores = [r2_dt, r2_rf]

plt.bar(models, r2_scores, color=['blue', 'green'])
plt.ylabel("R² Score")
plt.title("Comparison of Model Performance")
plt.show()


# In[ ]:




