#!/usr/bin/env python
# coding: utf-8

# <div align="center" style="background: lightgreen; color: Black; padding: 10px; border-radius: 70%;">
#   <h1 style="font-size: 40px; font-weight: bold; font-family: 'Times New Roman'; margin: 0; padding: 0;">Revenue Prediction with the help of User Segmentation Analysis</h1>
# </div> 

# <div align="center" style="background: lightyellow; color: black; padding: 10px; border-top-left-radius: 20px; border-top-right-radius: 0px; border-bottom-left-radius: 0px; border-bottom-right-radius: 20px;">
#   <h1 style="font-size: 40px; font-weight: bold; font-family: 'Times New Roman'; margin: 0; padding: 0;">1. Problem Statement</h1>
# </div>
# 
# <right>
# <h1 style="background-color:lightyellow; color:black; font-size:15px; font-family:Roman; border-radius: 10px;">
#          
# 1. Predictive Modeling for Revenue Generation: Develop a predictive model to forecast whether a user will generate revenue (make a purchase) during an online session based on features such as session duration, page engagement, traffic source, and visitor type. The goal is to enhance the platform's ability to identify potential customers and tailor the user experience to increase conversion rates.
# 
# 1. User Segmentation and Behavior Analysis:Conduct a comprehensive analysis of user behavior on the online purchasing platform by segmenting users based on their characteristics (e.g., visitor type, region, operating system). Identify distinct patterns and preferences within each segment and explore how these factors influence the likelihood of making a purchase. The objective is to provide actionable insights for targeted marketing and user experience improvements. </h1>
# </div> 
# 

# In[1]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import style
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('online_shoppers_intention.csv')


# In[3]:


df.head(5)


# In[4]:


df.shape


# In[5]:


df.columns


# <right>
# <h1 style="background-color:lightyellow; color:black; font-size:15px; font-family:Roman; border-radius: 20px;"> It is useful to first understand the data that you’re dealing with along with the data types of each column. The data types may dictate how you transform and engineer features

# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


print(df.apply(lambda col: col.unique())) 


# In[9]:


df.nunique()


# In[10]:


df.describe()


# # Data Visualization

# In[11]:


Revenue_1 = df.groupby("Revenue")
Revenue_1 = Revenue_1.size()
Revenue_1


# In[12]:


plt.pie(Revenue_1.values , labels = ("False", "True"), autopct='%1.1f%%',colors=['#28CDC3','#FCE858'], radius = 1, textprops = {"fontsize" : 16}) 
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Distribution of Revenue',  c= "blue")
plt.show()


# <right>
# <h1 style="background-color:lightyellow; color:black; font-size:16px; font-family:Roman; border-radius: 20px;"> First lets change the catogorical data into numarical data so that data can be easaly train.

# In[13]:


# Import the LabelEncoder
from sklearn.preprocessing import LabelEncoder

# Encode categorical features
label_encoder = LabelEncoder()
df['Month'] = label_encoder.fit_transform(df['Month'])
df['VisitorType'] = label_encoder.fit_transform(df['VisitorType'])
df['Weekend'] = df['Weekend'].astype(int)
df['Revenue'] = df['Revenue'].astype(int)


# In[14]:


X = df.drop('Revenue', axis=1)
y = df['Revenue']


# In[15]:


# Set a custom color palette
custom_palette = ['#28CDC3', '#FCE858']

# Visualize the distribution of the target variable with a custom color palette
plt.figure(figsize=(8, 5))
sns.countplot(x='Revenue', data=df, palette=custom_palette)
plt.title('Distribution of Revenue')
plt.show()


# In[16]:


# Set a different color palette for the correlation matrix
correlation_palette = 'viridis'

# Visualize the correlation matrix of numeric features with a different color palette
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap=correlation_palette, fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


# In[17]:


# Visualize the distribution of categorical features with different colors
categorical_features = ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend']

# Define a color palette with additional colors #FCE858 and #28CDC3
colors = ['#FCE858', '#28CDC3', *sns.color_palette('pastel')]

for feature in categorical_features:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=feature, hue='Revenue', data=df, palette=colors)
    plt.title(f'Distribution of {feature} by Revenue')
    plt.show()


# In[18]:


custom_palette_behavior = ['#28CDC3','#FCE858']

plt.figure(figsize=(10, 6))
sns.scatterplot(x='BounceRates', y='ExitRates', hue='Revenue', data=df, palette=custom_palette_behavior)
plt.title('Behavior Analysis: Bounce Rates vs Exit Rates')
plt.xlabel('Bounce Rates')
plt.ylabel('Exit Rates')
plt.show()


# In[19]:


different_palette = ['#28CDC3','#FCE858']

plt.figure(figsize=(10, 6))
sns.countplot(x='VisitorType', hue='Revenue', data=df, palette=different_palette)
plt.title('User Segmentation based on Visitor Type')
plt.show()


# # Outlires Handling

# In[20]:


import matplotlib.pyplot as plt
import seaborn as sns

# Select the numerical columns from your dataset
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Plot boxplots for each numeric column
plt.figure(figsize=(6, 12))  # Adjust figure size for better readability
for i, column in enumerate(numeric_columns, 1):
    plt.subplot(len(numeric_columns)//2 + 1, 2, i)  # 2 columns of subplots
    sns.boxplot(df[column])
    plt.title(f'Box plot of {column}')
    
plt.tight_layout()  # For better spacing between plots
plt.show()


# In[21]:


# Outlier detection using the IQR method
def detect_outliers_iqr(df):
    outliers = pd.DataFrame()
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[col] = ((df[col] < lower_bound) | (df[col] > upper_bound))
    
    return outliers

# Apply the function to the dataset
outliers = detect_outliers_iqr(df)


# In[22]:


# Display columns with outliers
outliers_summary = outliers.sum()
print("\nColumns with outliers:\n", outliers_summary[outliers_summary > 0])

# Optionally: Remove rows with outliers (can also apply different strategies, such as capping)
df_cleaned = df[~outliers.any(axis=1)]

# Summary of original vs cleaned data
print("\nOriginal data shape:", df.shape)
print("Cleaned data shape (without outliers):", df_cleaned.shape)


# <right>
# <h1 style="background-color:lightyellow; color:black; font-size:16px; font-family:Roman; border-radius: 20px;"> Outlire visualzation to understand the distribution better.

# <div align="center" style="background: lightyellow; color: black; padding: 10px; border-top-left-radius: 20px; border-top-right-radius: 0px; border-bottom-left-radius: 0px; border-bottom-right-radius: 20px;">
#   <h1 style="font-size: 40px; font-weight: bold; font-family: 'Times New Roman'; margin: 0; padding: 0;">  Preparing the Data for Model Traning
#  </h1>
# </div> 

# In[23]:


print("Unique classes in y (Revenue):")
print(y.unique())


# In[25]:


get_ipython().system('pip install imbalanced-learn')


# In[26]:


get_ipython().run_line_magic('pip', 'install imbalanced-learn')


# In[27]:


from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]


# In[28]:


# Importing SMOTE from imbalanced-learn library
from imblearn.over_sampling import SMOTE

# Assuming df_cleaned is the DataFrame after handling outliers
X = df_cleaned.drop('Revenue', axis=1)  # 'Revenue' is your target column
y = df_cleaned['Revenue']

# Splitting data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Applying SMOTE to balance the classes in the training set
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Check the number of samples in each class after applying SMOTE
print("Before SMOTE, class distribution in training set:")
print(y_train.value_counts())
print("\nAfter SMOTE, class distribution in resampled training set:")
print(y_train_smote.value_counts())


# In[29]:


from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on the balanced training data and transform it
X_train_scaled = scaler.fit_transform(X_train_smote)

# Transform the test data (scale using the same parameters from training)
X_test_scaled = scaler.transform(X_test)


# In[30]:


from sklearn.linear_model import LogisticRegression

# Initialize and fit the Logistic Regression model
lr_model = LogisticRegression(penalty='l2', C=1.0, random_state=42)
lr_model.fit(X_train_scaled, y_train_smote)


# In[31]:


from sklearn.tree import DecisionTreeClassifier

# Initialize and fit the Decision Tree model
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train_scaled, y_train_smote)


# In[32]:


from sklearn.ensemble import RandomForestClassifier
# Initialize and fit the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train_smote)


# In[33]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Evaluate Logistic Regression
y_pred_lr = lr_model.predict(X_test_scaled)
print("Logistic Regression Evaluation:")
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
print("Accuracy:", accuracy_score(y_test, y_pred_lr))

# Evaluate Decision Tree
y_pred_dt = dt_model.predict(X_test_scaled)
print("Decision Tree Evaluation:")
print(confusion_matrix(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))
print("Accuracy:", accuracy_score(y_test, y_pred_dt))

# Evaluate Random Forest
y_pred_rf = rf_model.predict(X_test_scaled)
print("Random Forest Evaluation:")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("Accuracy:", accuracy_score(y_test, y_pred_rf))


# ##### Hyperparameter Tuning for Decision Tree

# In[37]:


from sklearn.model_selection import GridSearchCV


# In[38]:


# Define the parameter grid for Decision Tree
param_grid_dt = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the Decision Tree model
dt = DecisionTreeClassifier(random_state=42)

# Set up GridSearchCV
grid_search_dt = GridSearchCV(estimator=dt, param_grid=param_grid_dt, 
                              cv=5, scoring='accuracy', verbose=2, n_jobs=-1)

# Fit the model to the training data
grid_search_dt.fit(X_train_smote, y_train_smote)

# Print the best parameters and score
print("Best Parameters for Decision Tree:", grid_search_dt.best_params_)
print("Best Score for Decision Tree:", grid_search_dt.best_score_)


# In[39]:


# Importing necessary libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Define the parameter distribution for Decision Tree
param_dist_dt = {
    'criterion': ['gini', 'entropy'],  # Splitting criteria
    'max_depth': [10, 20, 30, None],  # Possible tree depths
    'min_samples_split': [2, 5, 10],  # Minimum samples to split an internal node
    'min_samples_leaf': [1, 2, 4]     # Minimum samples at a leaf node
}

# Initialize the Decision Tree model
dt = DecisionTreeClassifier(random_state=42)

# Use RandomizedSearchCV for hyperparameter tuning
random_search_dt = RandomizedSearchCV(
    estimator=dt,
    param_distributions=param_dist_dt,  # Parameter grid
    n_iter=50,                          # Number of random parameter combinations to try
    cv=5,                               # Cross-validation folds
    scoring='accuracy',                 # Evaluation metric
    verbose=2,                          # Verbosity level
    random_state=42,                    # Seed for reproducibility
    n_jobs=2                            # Number of parallel jobs (adjust based on system capability)
)

# Fit the model on the SMOTE-resampled training data
random_search_dt.fit(X_train_smote, y_train_smote)

# Print the best parameters and score
print("Best Parameters for Decision Tree:", random_search_dt.best_params_)
print("Best Score for Decision Tree:", random_search_dt.best_score_)


# In[1]:


pip install lazypredict


# In[3]:


pip install --upgrade lazypredict


# In[4]:


pip install scikit-learn==1.0.2


# In[3]:



from lazypredict.Supervised import LazyClassifier, LazyRegressor


# In[ ]:




