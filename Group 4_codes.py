# -*- coding: utf-8 -*-
"""
Created on Thu May 16 20:39:13 2024

@author: vp00511
"""
####---------------------------
### Vrushali Decision tree
###_--------------------------
#DECISION TREE CLASSIFIER - 1 ALGORITH

# For suppressing the warnings
import warnings
warnings.simplefilter(action='ignore')

# Import the package
import pandas as pd

# Replace 'file_path' with the path to your CSV file
file_path =r'C:\Users\vp00511\OneDrive - University of Surrey\Desktop\Machine learning\Group\FEWS_NET_Staple_Food_Price_Data.csv'

# Read the CSV file into a DataFrame
Raw_DataFrame = pd.read_csv(file_path)
# Display/check the attribute names of the dataframe
print('\n\n\n-----------------------------------------------------------------------')
print('Attribute Names of the Dataframe')
print('-----------------------------------------------------------------------')
print(Raw_DataFrame.columns)
print('----------------------------------------------------------------------- \n\n\n')

# Display the top 5 observations of the DataFrame to verify that the data has been read correctly
print('----------------------------------------------------------------------------')
print('Top 5 observations of the DataFrame')
print('----------------------------------------------------------------------------')
print(Raw_DataFrame.head())
print('---------------------------------------------------------------------------- \n\n\n')

# Display the last 5 observations of the DataFrame to verify that the data has been read correctly
print('----------------------------------------------------------------------------')
print('Last 5 observations of the DataFrame')
print('----------------------------------------------------------------------------')
print(Raw_DataFrame.tail())
print('---------------------------------------------------------------------------- \n\n\n')
# Display the types and information about data
print('----------------------------------------------------------------------------')
print('Types and Information about DataFrame')
print('----------------------------------------------------------------------------')
print(Raw_DataFrame.info())
print('---------------------------------------------------------------------------- \n\n\n')

# Missing Values Calculation
ms = Raw_DataFrame.isnull().sum()
# Calculate the percentage of missing values in each column
ms_percentage = (Raw_DataFrame.isnull().sum()/(len(Raw_DataFrame)))*100
# Combine the missing value information into one dataframe 
Missing_Data_Info = pd.DataFrame({'Total Missings': ms, 'Percentage': ms_percentage})
# Print them the missing value information on screen
print('----------------------------------------')
print('Missing Data Information')
print('----------------------------------------')
print(Missing_Data_Info)
print('----------------------------------------\n\n\n')
# ----------------------------------------
# Missing value handling
# ----------------------------------------

# creating a copy of dataframe
Missing_Value_Handled_DataFrame_1 = Raw_DataFrame.copy()

#replacing the missing value of variable ' value' using the median
Missing_Value_Handled_DataFrame_1['value'].fillna(Raw_DataFrame['value'].median(), inplace=True)
# ---------------------------------------------------
# Repeat - Missing value identification
# ---------------------------------------------------

# Missing Values Calculation
ms2 = Missing_Value_Handled_DataFrame_1.isnull().sum()
# Calculate the percentage of missing values in each column
ms_percentage2 = (Missing_Value_Handled_DataFrame_1.isnull().sum()/(len(Missing_Value_Handled_DataFrame_1)))*100
# Combine the missing value information into one dataframe 
Missing_Data_Info2 = pd.DataFrame({'Total Missings': ms2, 'Percentage': ms_percentage2})

# Print them the missing value information on screen
print('-----------------------------------------------------')
print('Missing Data Information - After Dropping Attribute')
print('-----------------------------------------------------')
print(Missing_Data_Info2)
print('---------------------------------------------\n\n\n')
# ----------------------------
# Data Encoding - One Hot Encoding
# ----------------------------

# Shape of the Data
DataFrame_Shape = Missing_Value_Handled_DataFrame_1.shape

# Number of row
DataFrame_Row = DataFrame_Shape[0]

# Number of column
DataFrame_Col = DataFrame_Shape[1]

# Separate the features only for data encoding
Raw_DataFrame_Features_Only = Missing_Value_Handled_DataFrame_1.drop(['market'], axis = 1)

# Perform one-hot encoding, OHE = one-hot encoding
OHE_DataFrame_Features_Only = pd.get_dummies(Raw_DataFrame_Features_Only)
# Convert True/False to 1/0
Numeric_DataFrame = OHE_DataFrame_Features_Only.astype(int)

# Concatenate encoded features with target column - final numeric dataframe after encoding
Prepared_DataFrame = pd.concat([Numeric_DataFrame, Raw_DataFrame['market']], axis=1)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#Understanding outliers Identify numeric columns
numeric_cols = Prepared_DataFrame.select_dtypes(include=np.number).columns
#Calculate summary statistics
summary_stats = Prepared_DataFrame[numeric_cols].describe()

#Visualize data distribution
for col in numeric_cols:
    plt.figure(figsize=(8, 6))
    sns.histplot(Prepared_DataFrame[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

import pandas as pd
from scipy import stats
def remove_outliers(df, threshold=3):
    print("Threshold value:", threshold) 
    
    # Select numerical columns
    numerical_cols = Prepared_DataFrame.select_dtypes(include=['number']).columns
    
    # Check if there are numerical columns
    if len(numerical_cols) == 0:
        print("No numerical columns found in the DataFrame.")
        return Prepared_DataFrame
    
    # Calculate z-scores for numerical columns
    z_scores = stats.zscore(df[numerical_cols])
    abs_z_scores = abs(z_scores)
    
    # Check if there are any outliers based on the threshold
    filtered_entries = (abs_z_scores < threshold).all(axis=1)
    
    # Check if any outliers were found
    if filtered_entries.sum() == 0:
        print("No outliers were found based on the specified threshold.")
        return Prepared_DataFrame
    # Remove outliers and return the filtered DataFrame
    return Prepared_DataFrame[filtered_entries]


Prepared_DataFrame_no_outliers = remove_outliers(Prepared_DataFrame, threshold=3)

# Check if the resulting DataFrame is empty
if Prepared_DataFrame_no_outliers.empty:
    print("The DataFrame is empty after removing outliers.")
else:
    print("Outliers were successfully removed. DataFrame size:", Prepared_DataFrame_no_outliers.shape)


# -------------------
# Data Statistics
# -------------------

# statistics summary of data belonging to numerical datatype such as int, float
Data_Stat = Prepared_DataFrame.describe().T
print('-------------------------------------------------------------------------------')
print('Data Summary')
print('-------------------------------------------------------------------------------')
print(Data_Stat)
print('-------------------------------------------------------------------------------\n\n\n')


# provides a statistics summary of all data, include object, category etc
Data_Stat_All = Prepared_DataFrame.describe(include='all').T
    
# EDA Univariate Analysis
# ----------------------------------------
# Plotting the histogram of attributes
# ----------------------------------------

# Import the package
import matplotlib.pyplot as plt

# Get the list of column names
columns = Prepared_DataFrame.iloc[:, 0:3].columns.tolist() + Prepared_DataFrame.iloc[:,289 :300].columns.tolist()

# Set the number of rows and columns for subplots
num_rows = 5
num_cols = 5
# Define column ranges
column_ranges = [(300, 289), (0, 3)]

# Create a new figure and set its size
plt.figure(figsize=(15, 15))

# Iterate over each column and plot a histogram
for i, column in enumerate(columns, 1):
 plt.subplot(num_rows, num_cols, i)
 Prepared_DataFrame[column].hist()
 plt.title(column)
 
# ========================================
# Class Distribution
# ========================================

print("-----------------------------------------------")
print("Class Distribution")
print("-----------------------------------------------")
class_distribution = Prepared_DataFrame['market'].value_counts()
print("Class - Addis Ababa, Merkato -> " + "{:.3f}".format((class_distribution['Addis Ababa, Merkato']/DataFrame_Row)*100)+ " %")
print("Class - Bahir Dar -> " + "{:.3f}".format((class_distribution['Bahir Dar']/DataFrame_Row)*100) + " % \n\n\n")
print("Class - Warder -> " + "{:.3f}".format((class_distribution['Warder']/DataFrame_Row)*100) + " % \n\n\n")
print("Class - Chereti -> " + "{:.3f}".format((class_distribution['Chereti']/DataFrame_Row)*100) + " % \n\n\n")
print("Class - Degehabour -> " + "{:.3f}".format((class_distribution['Degehabour']/DataFrame_Row)*100) + " % \n\n\n")
print("Class - Dessie -> " + "{:.3f}".format((class_distribution['Dessie']/DataFrame_Row)*100) + " % \n\n\n")
print("Class - Dire Dawa, Kezira -> " + "{:.3f}".format((class_distribution['Dire Dawa, Kezira']/DataFrame_Row)*100) + " % \n\n\n")
print("Class - Gambella -> " + "{:.3f}".format((class_distribution['Gambella']/DataFrame_Row)*100) + " % \n\n\n")
print("Class - Gode -> " + "{:.3f}".format((class_distribution['Gode']/DataFrame_Row)*100) + " % \n\n\n")
print("Class - Jinka -> " + "{:.3f}".format((class_distribution['Jinka']/DataFrame_Row)*100) + " % \n\n\n")

#Dividing  the data set into training, testing and validation

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Separate features and target variable from the DataFrame
Features_Only = Prepared_DataFrame.drop(['market'], axis=1) 
Labels_Only = Prepared_DataFrame['market']

# Split the data into 70% training, 20% testing, and 10% validation sets
X_train, X_temp, y_train, y_temp = train_test_split(Features_Only, Labels_Only, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

# Initialize decision tree classifier
Decision_Tree_Classifier = DecisionTreeClassifier()

# Train the decision tree classifier on the training data
Decision_Tree_Classifier.fit(X_train, y_train)

# Make predictions on the testing set
Predicted_Labels = Decision_Tree_Classifier.predict(X_test)

# Calculate accuracy on the testing set
Accuracy = accuracy_score(y_test, Predicted_Labels)

# Output accuracy
print(f"Accuracy on Testing Set: {Accuracy*(100):.3f}")

# Make predictions on the validation set
Predicted_Labels_val = Decision_Tree_Classifier.predict(X_val)

# Calculate accuracy on the validation set
Accuracy_val = accuracy_score(y_val, Predicted_Labels_val)

# Output accuracy
print(f"Accuracy on Validation Set: {Accuracy_val*(100):.3f}")

# ============================================
# Decision Tree Classification
# ============================================

# Import the package
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report

# Separate features and target variable from the DataFrame
Features_Only = Prepared_DataFrame_no_outliers.drop(['market'],axis=1) # Assuming 'target_column' isthe name of your target variable
Labels_Only = Prepared_DataFrame_no_outliers['market']
 

# Convert Features_Only to an array-like object if it's a DataFrame
import numpy as np
Features_Only_array = Features_Only.values if isinstance(Features_Only, pd.DataFrame) else np.array(Features_Only)
print("----------------------------------------------------------------------------------------------------")
print("All Attributes Selected & Classified with Decision Tree")

# Define the number of folds for cross-validation
num_folds = 5

# Initialize a KFold object
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Initialize a list to store the accuracy scores for each fold
Accuracy_Scores = []
# Initialize a list to store the error scores for each fold
Error_Scores = []
# Initialize a list to store the sensitivity scores for each fold
Sensitivity_Scores = []
# Initialize a list to store the specificity scores for each fold
Specificity_Scores = []
# Initialize a list to store the F1 scores for each fold
F1_Scores = []


print("----------------------------------------------------------------------------------------------------")
print("\t\tFold \t Accuracy \t\t Error \t\t Sensitivity \t Specificity \t\t F1 Score")
print("----------------------------------------------------------------------------------------------------")

# Iterate through each fold
for f, (train_index, test_index) in enumerate(kf.split(Features_Only_array)):
    Training_Features, Testing_Features = Features_Only_array[train_index], Features_Only_array[test_index]
    Training_Labels, Testing_Labels = Labels_Only[train_index], Labels_Only[test_index]

    # Initialize decision tree classifier
    Decision_Tree_Classifier = DecisionTreeClassifier()

    # train the decision tree classifier
    Decision_Tree_Classifier.fit(Training_Features, Training_Labels)

    # Make predictions
    Predicted_Labels = Decision_Tree_Classifier.predict(Testing_Features)

    # Calculate confusion matrix
    Confusion_Matrix = confusion_matrix(Testing_Labels, Predicted_Labels)

    # Calculate accuracy for this fold
    Accuracy = accuracy_score(Testing_Labels, Predicted_Labels)

    # Append the accuracy score to the list
    Accuracy_Scores.append(Accuracy)

    # Calculate error for this fold
    Error = 1 - Accuracy
    # Append the error score to the list
    Error_Scores.append(Error)

    # Calculate sensitivity and specificity
    True_Negatives = Confusion_Matrix[0, 0]
    False_Positives = Confusion_Matrix[0, 1]
    False_Negatives = Confusion_Matrix[1, 0]
    True_Positives = Confusion_Matrix[1, 1]
    Sensitivity = True_Positives / (True_Positives + False_Negatives)

    # Append the sensitivity score to the list
    Sensitivity_Scores.append(Sensitivity)
    Specificity = True_Negatives / (True_Negatives + False_Positives)

    # Append the specificity score to the list
    Specificity_Scores.append(Specificity)

    # Calculate Precision and Recall from the confusion matrix
    Precision = True_Positives / (True_Positives + False_Positives)
    Recall = True_Positives / (True_Positives + False_Negatives)

    # Calculate F1 score
    f1 = 2 * (Precision * Recall) / (Precision + Recall)

    # Append the f1 score to the list
    F1_Scores.append(f1)

    print(" \t\t" + str(f) + " \t\t {:.3f}".format(Accuracy*100) + " \t\t {:.3f}".format(Error*100) + " \t\t {:.3f}".format(Sensitivity*100) + " \t\t {:.3f}".format(Specificity*100) + " \t\t {:.3f}".format(f1*100))

# Printing separation line to indicate 5-fold execution completion
print("----------------------------------------------------------------------------------------------------")
# Calculate the average accuracy across all folds
Average_Accuracy = sum(Accuracy_Scores) / len(Accuracy_Scores)
# Calculate the average error across all folds
Average_Error = sum(Error_Scores) / len(Error_Scores)
# Calculate the average sensitivity across all folds
Average_Sensitivity = sum(Sensitivity_Scores) / len(Sensitivity_Scores)
# Calculate the average specificity across all folds
Average_Specificity = sum(Specificity_Scores) / len(Specificity_Scores)
# Calculate the average f1 across all folds
Average_F1 = sum(F1_Scores) / len(F1_Scores)
# Printing the average performance scores
print("Average ->" + " \t\t {:.3f}".format(Average_Accuracy*100) + " \t\t {:.3f}".format(Average_Error*100) + " \t\t {:.3f}".format(Average_Sensitivity*100) + " \t\t{:.3f}".format(Average_Specificity*100) + " \t\t {:.3f}".format(Average_F1*100))
print("----------------------------------------------------------------------------------------------------")

#===============================================
# Decision Tree Classification Performance Improvement
# ===================================================

# Separate features and target variable from the DataFrame
Features_Only_Round_2 = Prepared_DataFrame_no_outliers.drop(['currency_ETB','unit_type_Weight','unit_type_Volume', 'unit_type_Item','unit_kg','unit_ea','unit_day','unit_L','market'],axis=1)
Labels_Only_Round_2 = Prepared_DataFrame_no_outliers['market']
print("\n\n\n----------------------------------------------------------------------------------------------------")
print("Features Selected & Classified with Decision Tree")

# ------------------------------------------
# Redo Decision Tree Classification
# ------------------------------------------
# Define the number of folds for cross-validation
num_folds = 5
# Initialize a KFold object
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Initialize lists to store the scores for each fold
Accuracy_Scores = []
Error_Scores = []
Sensitivity_Scores = []
Specificity_Scores = []
F1_Scores = []

#Convert Features_Only_Round_2 to an array-like object if it's a DataFrame
Features_Only_Round_2_array = Features_Only_Round_2.values if isinstance(Features_Only_Round_2, pd.DataFrame) else np.array(Features_Only_Round_2)

# Iterate through each fold
for f, (train_index, test_index) in enumerate(kf.split(Features_Only_Round_2_array)):
    Training_Features = Features_Only_Round_2_array[train_index]
    Testing_Features = Features_Only_Round_2_array[test_index]
    Training_Labels = Labels_Only_Round_2.loc[train_index]
    Testing_Labels = Labels_Only_Round_2.loc[test_index]
    
    # Initialize decision tree classifier
    Decision_Tree_Classifier = DecisionTreeClassifier(random_state=42)    

    # Train the decision tree classifier
    Decision_Tree_Classifier.fit(Training_Features, Training_Labels)
    # Make predictions
    Predicted_Labels = Decision_Tree_Classifier.predict(Testing_Features)

    # Calculate confusion matrix
    Confusion_Matrix = confusion_matrix(Testing_Labels, Predicted_Labels)

    # Calculate accuracy for this fold
    Accuracy = accuracy_score(Testing_Labels, Predicted_Labels)

    # Append the accuracy score to the list
    Accuracy_Scores.append(Accuracy)

    # Calculate error for this fold
    Error = 1 - Accuracy
    # Append the error score to the list
    Error_Scores.append(Error)

    # Calculate sensitivity and specificity
    True_Negatives = Confusion_Matrix[0, 0]
    False_Positives = Confusion_Matrix[0, 1]
    False_Negatives = Confusion_Matrix[1, 0]
    True_Positives = Confusion_Matrix[1, 1]
    Sensitivity = True_Positives / (True_Positives + False_Negatives)

    # Append the sensitivity score to the list
    Sensitivity_Scores.append(Sensitivity)
    Specificity = True_Negatives / (True_Negatives + False_Positives)

    # Append the specificity score to the list
    Specificity_Scores.append(Specificity)

    # Calculate Precision and Recall from the confusion matrix
    Precision = True_Positives / (True_Positives + False_Positives)
    Recall = True_Positives / (True_Positives + False_Negatives)

    # Calculate F1 score
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    # Append the F1 score to the list
    F1_Scores.append(F1)

    print("\t\t" + str(f) + " \t\t {:.3f}".format(Accuracy*100) + " \t\t {:.3f}".format(Error*100) + " \t\t {:.3f}".format(Sensitivity*100) + " \t\t {:.3f}".format(Specificity*100) + " \t\t {:.3f}".format(F1*100))

# Printing separation line to indicate 5-fold execution completion
print("----------------------------------------------------------------------------------------------------")

# Calculate the average scores across all folds
Average_Accuracy = sum(Accuracy_Scores) / len(Accuracy_Scores)
Average_Error = sum(Error_Scores) / len(Error_Scores)
Average_Sensitivity = sum(Sensitivity_Scores) / len(Sensitivity_Scores)
Average_Specificity = sum(Specificity_Scores) / len(Specificity_Scores)
Average_F1 = sum(F1_Scores) / len(F1_Scores)

# Printing the average performance scores
print("Average ->" + " \t\t {:.3f}".format(Average_Accuracy*100) + " \t\t {:.3f}".format(Average_Error*100) + " \t\t {:.3f}".format(Average_Sensitivity*100) + " \t\t{:.3f}".format(Average_Specificity*100) + " \t\t {:.3f}".format(Average_F1*100))
print("-------------------------------------------")

# ============================================
# Decision Tree Classification Performance Improvement
# ============================================

# ----------------------------------------------------------------------------------------
# Feature Extraction based on PCA
#----------------------------------------------------------------------------------------


# Import the package
from sklearn.decomposition import PCA

# First, separate the features (X) from the target variable (if any)
Data_without_target = Prepared_DataFrame_no_outliers.drop('market', axis=1) 

# Initialize PCA with the desired number of components
pca = PCA(n_components=10)
# Fit PCA to the data and transform the features
Extracted_PC_features = pca.fit_transform(Data_without_target)

# Create a new DataFrame with the transformed features
New_PC_features_in_dataframe = pd.DataFrame(data=Extracted_PC_features, columns=[f'PC{i+1}' for i in range(10)])

# Combine the principal components DataFrame with the target column (if required)
New_PC_features_with_target = pd.concat([New_PC_features_in_dataframe,Prepared_DataFrame_no_outliers['market']], axis=1)

 # Separate features and target variable from the DataFrame
Features_Only_Round_3 = New_PC_features_with_target.drop(['market'],axis=1)
Labels_Only_Round_3 = New_PC_features_with_target['market']
print("\n\n\n----------------------------------------------------------------------------------------------------")
print("Features Extracted & Classified with Decision Tree")

# ------------------------------------------
# Redo Decision Tree Classification
# ------------------------------------------

# Define the number of folds for cross-validation
num_folds = 5
# Initialize a KFold object
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
# Initialize a list to store the accuracy scores for each fold
Accuracy_Scores = []
# Initialize a list to store the error scores for each fold
Error_Scores = []
# Initialize a list to store the sensitivity scores for each fold
Sensitivity_Scores = []
# Initialize a list to store the specificity scores for each fold
Specificity_Scores = []
# Initialize a list to store the F1 scores for each fold
F1_Scores = []

print("\n\n\n----------------------------------------------------------------------------------------------------")
print("Features Extracted & Classified with Decision Tree")
print("----------------------------------------------------------------------------------------------------")
print("\t\tFold \t Accuracy \t\t Error \t\t Sensitivity \t Specificity \t\t F1 Score")
print("----------------------------------------------------------------------------------------------------")

#Convert Features_Only_Round_2 to an array-like object if it's a DataFrame
Features_Only_Round_3_array = Features_Only_Round_3.values if isinstance(Features_Only_Round_3, pd.DataFrame) else np.array(Features_Only_Round_3)
# Iterate through each fold
for f, (train_index, test_index) in enumerate(kf.split(Features_Only_Round_3_array)):
    # Split the data into training and testing sets for this fold
    Training_Features = Features_Only_Round_3_array[train_index]
    Testing_Features = Features_Only_Round_3_array[test_index]
    Training_Labels = Labels_Only_Round_3.loc[train_index]
    Testing_Labels = Labels_Only_Round_3.loc[test_index]
    
    # Initialize decision tree classifier
    Decision_Tree_Classifier = DecisionTreeClassifier()

    # Train the decision tree classifier
    Decision_Tree_Classifier.fit(Training_Features, Training_Labels)

    # Make predictions
    Predicted_Labels = Decision_Tree_Classifier.predict(Testing_Features)

    # Calculate confusion matrix
    Confusion_Matrix = confusion_matrix(Testing_Labels, Predicted_Labels)
    
    # Calculate accuracy for this fold
    Accuracy = accuracy_score(Testing_Labels, Predicted_Labels)
    # Append the accuracy score to the list
    Accuracy_Scores.append(Accuracy)    
    
    # Calculate error for this fold
    Error = 1 - Accuracy
    # Append the error score to the list
    Error_Scores.append(Error)
    
    # Calculate sensitivity and specificity
    True_Negatives = Confusion_Matrix[0, 0]
    False_Positives = Confusion_Matrix[0, 1]
    False_Negatives = Confusion_Matrix[1, 0]
    True_Positives = Confusion_Matrix[1, 1]

    Sensitivity = True_Positives / (True_Positives + False_Negatives)
    # Append the sensitivity score to the list
    Sensitivity_Scores.append(Sensitivity)

    Specificity = True_Negatives / (True_Negatives + False_Positives)
    # Append the specificity score to the list
    Specificity_Scores.append(Specificity)

    # Calculate Precision and Recall from the confusion matrix
    Precision = True_Positives / (True_Positives + False_Positives)
    Recall = True_Positives / (True_Positives + False_Negatives)

    # Calculate F1 score
    f1 = 2 * (Precision * Recall) / (Precision + Recall)
    # Append the f1 score to the list
    F1_Scores.append(f1)

    # Print the performance metrics for this fold
    print("\t\t" + str(f) + " \t\t {:.3f}".format(Accuracy*100) + " \t\t {:.3f}".format(Error*100) + " \t\t {:.3f}".format(Sensitivity*100) + " \t\t {:.3f}".format(Specificity*100) + " \t\t {:.3f}".format(f1*100))

# Printing separation line to indicate 5-fold execution completion
print("----------------------------------------------------------------------------------------------------")
# Calculate the average accuracy across all folds
Average_Accuracy = sum(Accuracy_Scores) / len(Accuracy_Scores)
# Calculate the average error across all folds
Average_Error = sum(Error_Scores) / len(Error_Scores)
# Calculate the average sensitivity across all folds
Average_Sensitivity = sum(Sensitivity_Scores) / len(Sensitivity_Scores)
# Calculate the average specificity across all folds
Average_Specificity = sum(Specificity_Scores) / len(Specificity_Scores)
# Calculate the average f1 across all folds
Average_F1 = sum(F1_Scores) / len(F1_Scores)
# Printing the average performance scores
print("Average ->" + " \t\t {:.3f}".format(Average_Accuracy*100) + " \t\t {:.3f}".format(Average_Error*100) + " \t\t {:.3f}".format(Average_Sensitivity*100) + " \t\t{:.3f}".format(Average_Specificity*100) + " \t\t {:.3f}".format(Average_F1*100))
print("----------------------------------------------------------------------------------------------------")

import pandas as pd
import matplotlib.pyplot as plt


# Filter data for Addis Ababa Market
Awash_data = Prepared_DataFrame_no_outliers[Raw_DataFrame['market'] == 'Awash']

product_Awash = Awash_data[['product_Beans (Haricot)', 'product_Camels (Local Quality)', 'product_Casual Labor (unskilled, daily, without food)', 'product_Diesel', 'product_Firewood', 'product_Gasoline', 'product_Goats (Local Quality)', 'product_Horse beans', 'product_Maize Grain (White)', 'product_Mixed Teff', 'product_Oxen (Local Quality)', 'product_Refined Vegetable Oil', 'product_Refined sugar', 'product_Rice (Milled)', 'product_Sheep (Local Quality)', 'product_Sorghum (Red)', 'product_Sorghum (White)', 'product_Sorghum (Yellow)', 'product_Wheat Flour', 'product_Wheat Grain']].sum()
# Print the counts for each product
print("Product counts with respect to Awash market:")
print(product_Awash)

Bahir_Dar_data = Prepared_DataFrame_no_outliers[Prepared_DataFrame_no_outliers['market'] == 'Bahir Dar']
product_Bahir_Dar = Bahir_Dar_data[['product_Beans (Haricot)', 'product_Camels (Local Quality)', 'product_Casual Labor (unskilled, daily, without food)', 'product_Diesel', 'product_Firewood', 'product_Gasoline', 'product_Goats (Local Quality)', 'product_Horse beans', 'product_Maize Grain (White)', 'product_Mixed Teff', 'product_Oxen (Local Quality)', 'product_Refined Vegetable Oil', 'product_Refined sugar', 'product_Rice (Milled)', 'product_Sheep (Local Quality)', 'product_Sorghum (Red)', 'product_Sorghum (White)', 'product_Sorghum (Yellow)', 'product_Wheat Flour', 'product_Wheat Grain']].sum()
print("Product with respect to Bahir Dar:")
print(product_Bahir_Dar)


Beddenno_data = Prepared_DataFrame_no_outliers[Prepared_DataFrame_no_outliers['market'] == 'Beddenno']
product_Beddenno = Beddenno_data[['product_Beans (Haricot)', 'product_Camels (Local Quality)', 'product_Casual Labor (unskilled, daily, without food)', 'product_Diesel', 'product_Firewood', 'product_Gasoline', 'product_Goats (Local Quality)', 'product_Horse beans', 'product_Maize Grain (White)', 'product_Mixed Teff', 'product_Oxen (Local Quality)', 'product_Refined Vegetable Oil', 'product_Refined sugar', 'product_Rice (Milled)', 'product_Sheep (Local Quality)', 'product_Sorghum (Red)', 'product_Sorghum (White)', 'product_Sorghum (Yellow)', 'product_Wheat Flour', 'product_Wheat Grain']].sum()
print("Product with respect to Bahir Dar:")
print(product_Beddenno)

Chereti_data = Prepared_DataFrame_no_outliers[Prepared_DataFrame_no_outliers['market'] == 'Chereti']
product_chereti = Chereti_data[['product_Beans (Haricot)', 'product_Camels (Local Quality)', 'product_Casual Labor (unskilled, daily, without food)', 'product_Diesel', 'product_Firewood', 'product_Gasoline', 'product_Goats (Local Quality)', 'product_Horse beans', 'product_Maize Grain (White)', 'product_Mixed Teff', 'product_Oxen (Local Quality)', 'product_Refined Vegetable Oil', 'product_Refined sugar', 'product_Rice (Milled)', 'product_Sheep (Local Quality)', 'product_Sorghum (Red)', 'product_Sorghum (White)', 'product_Sorghum (Yellow)', 'product_Wheat Flour', 'product_Wheat Grain']].sum()
print("Product with respect to Bahir Dar:")
print(product_chereti)


Degehabour_data = Prepared_DataFrame_no_outliers[Prepared_DataFrame_no_outliers['market'] == 'Degehabour']
product_Degehabour = Degehabour_data[['product_Beans (Haricot)', 'product_Camels (Local Quality)', 'product_Casual Labor (unskilled, daily, without food)', 'product_Diesel', 'product_Firewood', 'product_Gasoline', 'product_Goats (Local Quality)', 'product_Horse beans', 'product_Maize Grain (White)', 'product_Mixed Teff', 'product_Oxen (Local Quality)', 'product_Refined Vegetable Oil', 'product_Refined sugar', 'product_Rice (Milled)', 'product_Sheep (Local Quality)', 'product_Sorghum (Red)', 'product_Sorghum (White)', 'product_Sorghum (Yellow)', 'product_Wheat Flour', 'product_Wheat Grain']].sum()
print("Product with respect to Degehabour:")
print(product_Degehabour)

Dessie_data = Prepared_DataFrame_no_outliers[Prepared_DataFrame_no_outliers['market'] == 'Dessie']
product_Dessie = Dessie_data[['product_Beans (Haricot)', 'product_Camels (Local Quality)', 'product_Casual Labor (unskilled, daily, without food)', 'product_Diesel', 'product_Firewood', 'product_Gasoline', 'product_Goats (Local Quality)', 'product_Horse beans', 'product_Maize Grain (White)', 'product_Mixed Teff', 'product_Oxen (Local Quality)', 'product_Refined Vegetable Oil', 'product_Refined sugar', 'product_Rice (Milled)', 'product_Sheep (Local Quality)', 'product_Sorghum (Red)', 'product_Sorghum (White)', 'product_Sorghum (Yellow)', 'product_Wheat Flour', 'product_Wheat Grain']].sum()
print("Product with respect to Dessie:")
print(product_Dessie)

Dire_Dawa_Kezira_data = Prepared_DataFrame_no_outliers[Prepared_DataFrame_no_outliers['market'] == 'Dire Dawa Kezira']
product_Dire_Dawa = Dire_Dawa_Kezira_data[['product_Beans (Haricot)', 'product_Camels (Local Quality)', 'product_Casual Labor (unskilled, daily, without food)', 'product_Diesel', 'product_Firewood', 'product_Gasoline', 'product_Goats (Local Quality)', 'product_Horse beans', 'product_Maize Grain (White)', 'product_Mixed Teff', 'product_Oxen (Local Quality)', 'product_Refined Vegetable Oil', 'product_Refined sugar', 'product_Rice (Milled)', 'product_Sheep (Local Quality)', 'product_Sorghum (Red)', 'product_Sorghum (White)', 'product_Sorghum (Yellow)', 'product_Wheat Flour', 'product_Wheat Grain']].sum()
print("Product with respect to Dire Dawa, Kezira:")
print(product_Dire_Dawa)

Gambella_data = Prepared_DataFrame_no_outliers[Prepared_DataFrame_no_outliers['market'] == 'Gambella']
product_Gambella = Gambella_data[['product_Beans (Haricot)', 'product_Camels (Local Quality)', 'product_Casual Labor (unskilled, daily, without food)', 'product_Diesel', 'product_Firewood', 'product_Gasoline', 'product_Goats (Local Quality)', 'product_Horse beans', 'product_Maize Grain (White)', 'product_Mixed Teff', 'product_Oxen (Local Quality)', 'product_Refined Vegetable Oil', 'product_Refined sugar', 'product_Rice (Milled)', 'product_Sheep (Local Quality)', 'product_Sorghum (Red)', 'product_Sorghum (White)', 'product_Sorghum (Yellow)', 'product_Wheat Flour', 'product_Wheat Grain']].sum()
print("Product with respect to Gambella:")
print(product_Gambella)

Gode_data = Prepared_DataFrame_no_outliers[Prepared_DataFrame_no_outliers['market'] == 'Gode']
product_Gode = Gode_data[['product_Beans (Haricot)', 'product_Camels (Local Quality)', 'product_Casual Labor (unskilled, daily, without food)', 'product_Diesel', 'product_Firewood', 'product_Gasoline', 'product_Goats (Local Quality)', 'product_Horse beans', 'product_Maize Grain (White)', 'product_Mixed Teff', 'product_Oxen (Local Quality)', 'product_Refined Vegetable Oil', 'product_Refined sugar', 'product_Rice (Milled)', 'product_Sheep (Local Quality)', 'product_Sorghum (Red)', 'product_Sorghum (White)', 'product_Sorghum (Yellow)', 'product_Wheat Flour', 'product_Wheat Grain']].sum()
print("Product with respect to Gode:")
print(product_Gode)

Jinka_data = Prepared_DataFrame_no_outliers[Prepared_DataFrame_no_outliers['market'] == 'Jinka']
product_Jinka = Jinka_data[['product_Beans (Haricot)', 'product_Camels (Local Quality)', 'product_Casual Labor (unskilled, daily, without food)', 'product_Diesel', 'product_Firewood', 'product_Gasoline', 'product_Goats (Local Quality)', 'product_Horse beans', 'product_Maize Grain (White)', 'product_Mixed Teff', 'product_Oxen (Local Quality)', 'product_Refined Vegetable Oil', 'product_Refined sugar', 'product_Rice (Milled)', 'product_Sheep (Local Quality)', 'product_Sorghum (Red)', 'product_Sorghum (White)', 'product_Sorghum (Yellow)', 'product_Wheat Flour', 'product_Wheat Grain']].sum()
print("Product with respect to Jinka:")
print(product_Jinka)

Logia_data = Prepared_DataFrame_no_outliers[Prepared_DataFrame['market'] == 'Logia']
product_Logia = Logia_data[['product_Beans (Haricot)', 'product_Camels (Local Quality)', 'product_Casual Labor (unskilled, daily, without food)', 'product_Diesel', 'product_Firewood', 'product_Gasoline', 'product_Goats (Local Quality)', 'product_Horse beans', 'product_Maize Grain (White)', 'product_Mixed Teff', 'product_Oxen (Local Quality)', 'product_Refined Vegetable Oil', 'product_Refined sugar', 'product_Rice (Milled)', 'product_Sheep (Local Quality)', 'product_Sorghum (Red)', 'product_Sorghum (White)', 'product_Sorghum (Yellow)', 'product_Wheat Flour', 'product_Wheat Grain']].sum()
print("Product with respect to Logia:")
print(product_Logia)

Mekele_data = Prepared_DataFrame_no_outliers[Prepared_DataFrame_no_outliers['market'] == 'Mekele']
product_Mekele = Mekele_data[['product_Beans (Haricot)', 'product_Camels (Local Quality)', 'product_Casual Labor (unskilled, daily, without food)', 'product_Diesel', 'product_Firewood', 'product_Gasoline', 'product_Goats (Local Quality)', 'product_Horse beans', 'product_Maize Grain (White)', 'product_Mixed Teff', 'product_Oxen (Local Quality)', 'product_Refined Vegetable Oil', 'product_Refined sugar', 'product_Rice (Milled)', 'product_Sheep (Local Quality)', 'product_Sorghum (Red)', 'product_Sorghum (White)', 'product_Sorghum (Yellow)', 'product_Wheat Flour', 'product_Wheat Grain']].sum()
print("Product with respect to Mekele:")
print(product_Mekele)

Nazareth_Adama_data = Prepared_DataFrame_no_outliers[Prepared_DataFrame_no_outliers['market'] == 'Nazareth Adama']
product_Nazareth = Nazareth_Adama_data[['product_Beans (Haricot)', 'product_Camels (Local Quality)', 'product_Casual Labor (unskilled, daily, without food)', 'product_Diesel', 'product_Firewood', 'product_Gasoline', 'product_Goats (Local Quality)', 'product_Horse beans', 'product_Maize Grain (White)', 'product_Mixed Teff', 'product_Oxen (Local Quality)', 'product_Refined Vegetable Oil', 'product_Refined sugar', 'product_Rice (Milled)', 'product_Sheep (Local Quality)', 'product_Sorghum (Red)', 'product_Sorghum (White)', 'product_Sorghum (Yellow)', 'product_Wheat Flour', 'product_Wheat Grain']].sum()
print("Product with respect to Nazareth Adama:")
print(product_Nazareth)

Nekemte_data = Prepared_DataFrame_no_outliers[Prepared_DataFrame_no_outliers['market'] == 'Nekemte']
product_Nekemte = Nekemte_data[['product_Beans (Haricot)', 'product_Camels (Local Quality)', 'product_Casual Labor (unskilled, daily, without food)', 'product_Diesel', 'product_Firewood', 'product_Gasoline', 'product_Goats (Local Quality)', 'product_Horse beans', 'product_Maize Grain (White)', 'product_Mixed Teff', 'product_Oxen (Local Quality)', 'product_Refined Vegetable Oil', 'product_Refined sugar', 'product_Rice (Milled)', 'product_Sheep (Local Quality)', 'product_Sorghum (Red)', 'product_Sorghum (White)', 'product_Sorghum (Yellow)', 'product_Wheat Flour', 'product_Wheat Grain']].sum()
print("Product with respect to Nekemte:")
print(product_Nekemte)


Sekota_data = Prepared_DataFrame_no_outliers[Prepared_DataFrame_no_outliers['market'] == 'Sekota']
product_Sekota = Sekota_data[['product_Beans (Haricot)', 'product_Camels (Local Quality)', 'product_Casual Labor (unskilled, daily, without food)', 'product_Diesel', 'product_Firewood', 'product_Gasoline', 'product_Goats (Local Quality)', 'product_Horse beans', 'product_Maize Grain (White)', 'product_Mixed Teff', 'product_Oxen (Local Quality)', 'product_Refined Vegetable Oil', 'product_Refined sugar', 'product_Rice (Milled)', 'product_Sheep (Local Quality)', 'product_Sorghum (Red)', 'product_Sorghum (White)', 'product_Sorghum (Yellow)', 'product_Wheat Flour', 'product_Wheat Grain']].sum()
print("Product with respect to Sekota:")
print(product_Sekota)

Shashemene_data = Prepared_DataFrame_no_outliers[Prepared_DataFrame_no_outliers['market'] == 'Shashemene']
product_Shashemene = Shashemene_data[['product_Beans (Haricot)', 'product_Camels (Local Quality)', 'product_Casual Labor (unskilled, daily, without food)', 'product_Diesel', 'product_Firewood', 'product_Gasoline', 'product_Goats (Local Quality)', 'product_Horse beans', 'product_Maize Grain (White)', 'product_Mixed Teff', 'product_Oxen (Local Quality)', 'product_Refined Vegetable Oil', 'product_Refined sugar', 'product_Rice (Milled)', 'product_Sheep (Local Quality)', 'product_Sorghum (Red)', 'product_Sorghum (White)', 'product_Sorghum (Yellow)', 'product_Wheat Flour', 'product_Wheat Grain']].sum()
print("Product with respect to Shashemene:")
print(product_Shashemene)

Shire_data = Prepared_DataFrame_no_outliers[Prepared_DataFrame_no_outliers['market'] == 'Shire']
product_Shire = Shire_data[['product_Beans (Haricot)', 'product_Camels (Local Quality)', 'product_Casual Labor (unskilled, daily, without food)', 'product_Diesel', 'product_Firewood', 'product_Gasoline', 'product_Goats (Local Quality)', 'product_Horse beans', 'product_Maize Grain (White)', 'product_Mixed Teff', 'product_Oxen (Local Quality)', 'product_Refined Vegetable Oil', 'product_Refined sugar', 'product_Rice (Milled)', 'product_Sheep (Local Quality)', 'product_Sorghum (Red)', 'product_Sorghum (White)', 'product_Sorghum (Yellow)', 'product_Wheat Flour', 'product_Wheat Grain']].sum()
print("Product with respect to Shire:")
print(product_Shire)

Sikela_data = Prepared_DataFrame_no_outliers[Prepared_DataFrame_no_outliers['market'] == 'Sikela']
product_Sikela = Sikela_data[['product_Beans (Haricot)', 'product_Camels (Local Quality)', 'product_Casual Labor (unskilled, daily, without food)', 'product_Diesel', 'product_Firewood', 'product_Gasoline', 'product_Goats (Local Quality)', 'product_Horse beans', 'product_Maize Grain (White)', 'product_Mixed Teff', 'product_Oxen (Local Quality)', 'product_Refined Vegetable Oil', 'product_Refined sugar', 'product_Rice (Milled)', 'product_Sheep (Local Quality)', 'product_Sorghum (Red)', 'product_Sorghum (White)', 'product_Sorghum (Yellow)', 'product_Wheat Flour', 'product_Wheat Grain']].sum()
print("Product with respect to Sikela:")
print(product_Sikela)

Sodo_data = Prepared_DataFrame_no_outliers[Prepared_DataFrame_no_outliers['market'] == 'Sodo']
product_Sodo = Sodo_data[['product_Beans (Haricot)', 'product_Camels (Local Quality)', 'product_Casual Labor (unskilled, daily, without food)', 'product_Diesel', 'product_Firewood', 'product_Gasoline', 'product_Goats (Local Quality)', 'product_Horse beans', 'product_Maize Grain (White)', 'product_Mixed Teff', 'product_Oxen (Local Quality)', 'product_Refined Vegetable Oil', 'product_Refined sugar', 'product_Rice (Milled)', 'product_Sheep (Local Quality)', 'product_Sorghum (Red)', 'product_Sorghum (White)', 'product_Sorghum (Yellow)', 'product_Wheat Flour', 'product_Wheat Grain']].sum()
print("Product with respect to Sodo:")
print(product_Sodo)

Warder_data = Prepared_DataFrame_no_outliers[Prepared_DataFrame_no_outliers['market'] == 'Warder']
product_Warder = Warder_data[['product_Beans (Haricot)', 'product_Camels (Local Quality)', 'product_Casual Labor (unskilled, daily, without food)', 'product_Diesel', 'product_Firewood', 'product_Gasoline', 'product_Goats (Local Quality)', 'product_Horse beans', 'product_Maize Grain (White)', 'product_Mixed Teff', 'product_Oxen (Local Quality)', 'product_Refined Vegetable Oil', 'product_Refined sugar', 'product_Rice (Milled)', 'product_Sheep (Local Quality)', 'product_Sorghum (Red)', 'product_Sorghum (White)', 'product_Sorghum (Yellow)', 'product_Wheat Flour', 'product_Wheat Grain']].sum()
print("Product with respect to Warder:")
print(product_Warder)

Yabelo_data = Prepared_DataFrame_no_outliers[Prepared_DataFrame_no_outliers['market'] == 'Yabelo']
product_Yabelo = Yabelo_data[['product_Beans (Haricot)', 'product_Camels (Local Quality)', 'product_Casual Labor (unskilled, daily, without food)', 'product_Diesel', 'product_Firewood', 'product_Gasoline', 'product_Goats (Local Quality)', 'product_Horse beans', 'product_Maize Grain (White)', 'product_Mixed Teff', 'product_Oxen (Local Quality)', 'product_Refined Vegetable Oil', 'product_Refined sugar', 'product_Rice (Milled)', 'product_Sheep (Local Quality)', 'product_Sorghum (Red)', 'product_Sorghum (White)', 'product_Sorghum (Yellow)', 'product_Wheat Flour', 'product_Wheat Grain']].sum()
print("Product with respect to Yabelo:")
print(product_Yabelo)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.countplot(data=Missing_Value_Handled_DataFrame_1, x='product_source')
plt.title('Distribution of Product Sources')
plt.xlabel('Product Source')
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()

imported_products = Missing_Value_Handled_DataFrame_1[Missing_Value_Handled_DataFrame_1['product_source'] == 'Import']['product'].unique()
print("Imported Products:", imported_products)

#filter the data set for the 3 product
required_columns = ['latitude','value','product_Diesel','product_Gasoline','product_source_Local','product_source_Import','market', 'longitude', 'product_Refined Vegetable Oil', 'product_Refined sugar', 'product_Rice (Milled)']

# Filter the DataFrame and select only the required columns
filtered_data = Prepared_DataFrame_no_outliers[required_columns]

# Create scatter plots
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.scatterplot(data=filtered_data, x='longitude', y='latitude', palette='viridis')
plt.title('Imported products')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Filter data for sugar product between specific latitude and longitude ranges
filtered_data_latitude_longitude = filtered_data[( filtered_data
                               ['latitude'] >= 6) & (filtered_data['latitude'] <= 10) & 
                               (filtered_data['longitude'] >= 36) & (filtered_data['longitude'] <= 42)]


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Decision tree classifer on all the imported product
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# To understand all the 3 different product with market create a ew data set
# Filter the DataFrame for 'product_Refined sugar' data
sugar_data = filtered_data_latitude_longitude[filtered_data_latitude_longitude['product_Refined sugar'] == 1]

# Plot scatter plot
# Plot box plot
plt.figure(figsize=(10, 6))
plt.boxplot([sugar_data[sugar_data['market'] == market]['value'] for market in sugar_data['market'].unique()],
            labels=sugar_data['market'].unique())
plt.title('Box Plot of Product Refined Sugar Value across Market')
plt.xlabel('Market')
plt.ylabel('Value')
plt.grid(True)
plt.show()

#decision tree classifier for sugar:
    
X = sugar_data[['value']]  # Feature column
y = sugar_data['market']     # Target column

# One-hot encode categorical variables
X = pd.get_dummies(X)

# Step 2: Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train Decision Tree Classifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Step 4: Predict on test set
y_pred = classifier.predict(X_test)

# Step 5: Evaluate the classifier
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
  
rice_data = filtered_data_latitude_longitude[filtered_data_latitude_longitude['product_Rice (Milled)'] == 1]

import matplotlib.pyplot as plt

# Plot box plot
plt.figure(figsize=(10, 6))
plt.boxplot([rice_data[rice_data['market'] == market]['value'] for market in rice_data['market'].unique()],
            labels=sugar_data['market'].unique())
plt.title('Box Plot of Product rice Value across Market')
plt.xlabel('Market')
plt.ylabel('Value')
plt.grid(True)
plt.show()

#decsion tree for rice


X = rice_data[['value']]  # Feature column
y = rice_data['market']     # Target column

# One-hot encode categorical variables
X = pd.get_dummies(X)

# Step 2: Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train Decision Tree Classifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Step 4: Predict on test set
y_pred = classifier.predict(X_test)

# Step 5: Evaluate the classifier
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

oil_data = filtered_data_latitude_longitude[filtered_data_latitude_longitude['product_Refined Vegetable Oil'] == 1]

import matplotlib.pyplot as plt


# Plot box plot
plt.figure(figsize=(10, 6))
plt.boxplot([oil_data[oil_data['market'] == market]['value'] for market in oil_data['market'].unique()],
            labels=oil_data['market'].unique())
plt.title('Box Plot of Product oil Value across Market')
plt.xlabel('Market')
plt.ylabel('Value')
plt.grid(True)
plt.show()

#Decision tree for oil

X = oil_data[['value']]  # Feature column
y = oil_data['market']     # Target column

# One-hot encode categorical variables
X = pd.get_dummies(X)

# Step 2: Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train Decision Tree Classifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Step 4: Predict on test set
y_pred = classifier.predict(X_test)

# Step 5: Evaluate the classifier
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


diesel_data = filtered_data_latitude_longitude[filtered_data_latitude_longitude['product_Diesel'] == 1]

import matplotlib.pyplot as plt

# Plot box plot
plt.figure(figsize=(10, 6))
plt.boxplot([diesel_data[diesel_data['market'] == market]['value'] for market in diesel_data['market'].unique()],
            labels=diesel_data['market'].unique())
plt.title('Box Plot of Product Diesel Value across Market')
plt.xlabel('Market')
plt.ylabel('Value')
plt.grid(True)
plt.show()

#Decision tree for diesel:
    
X = diesel_data[['value']]  # Feature column
y = diesel_data['market']     # Target column

    # One-hot encode categorical variables
X = pd.get_dummies(X)

    # Step 2: Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Train Decision Tree Classifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

    # Step 4: Predict on test set
y_pred = classifier.predict(X_test)

    # Step 5: Evaluate the classifier
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))    

Gasoline_data = filtered_data_latitude_longitude[filtered_data_latitude_longitude['product_Gasoline'] == 1]

import matplotlib.pyplot as plt

# Plot box plot
plt.figure(figsize=(10, 6))
plt.boxplot([Gasoline_data[Gasoline_data['market'] == market]['value'] for market in Gasoline_data['market'].unique()],
            labels=Gasoline_data['market'].unique())
plt.title('Box Plot of Product Gasoline Value across Market')
plt.xlabel('Market')
plt.ylabel('Value')
plt.grid(True)
plt.show()

#deicision tree classifier for gasoline:
    #For gasoline
    
X = Gasoline_data[['value']]  # Feature column
y = Gasoline_data['market']     # Target column

    # One-hot encode categorical variables
X = pd.get_dummies(X)

    # Step 2: Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Train Decision Tree Classifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

    # Step 4: Predict on test set
y_pred = classifier.predict(X_test)

    # Step 5: Evaluate the classifier
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
    

#Box plot of all imported data  wrt to market and value
plt.figure(figsize=(10, 6))
sns.boxplot(data=pd.concat([rice_data.assign(Product='Rice'),
                            diesel_data.assign(Product='Diesel'),
                            Gasoline_data.assign(Product='Gasoline'),
                            oil_data.assign(Product='Oil'),
                            sugar_data.assign(Product='Sugar')]),
            x='market', y='value', hue='Product')
plt.title('Distribution of Value across Different Markets for Rice, Oil, and Sugar')
plt.xlabel('Market')
plt.ylabel('Value')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()

#decision tree classifier for all columns in data set
X = filtered_data_latitude_longitude[['value']]  # Feature column
y = filtered_data_latitude_longitude['market']     # Target column

    # One-hot encode categorical variables
X = pd.get_dummies(X)

    # Step 2: Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Train Decision Tree Classifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

    # Step 4: Predict on test set
y_pred = classifier.predict(X_test)

    # Step 5: Evaluate the classifier
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Filter the DataFrame and select only the required columns
Local_data = Prepared_DataFrame_no_outliers.drop(['product_Diesel','product_Gasoline','product_Refined Vegetable Oil', 'product_Refined sugar', 'product_Rice (Milled)'], axis=1)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.scatterplot(data=Local_data, x='longitude', y='latitude', palette='viridis')
plt.title('Imported products')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Filter data for local products product between specific latitude and longitude ranges
local_data_filtered = Local_data[( Local_data
                               ['latitude'] >= 6) & (filtered_data['latitude'] <= 10) & 
                               (Local_data['longitude'] >= 36) & (Local_data['longitude'] <= 42)]

#decision tree for local data filtered:
    #decision tree classifier for all columns in data set
X = local_data_filtered[['value']]  # Feature column
y = local_data_filtered['market']     # Target column

        # One-hot encode categorical variables
X = pd.get_dummies(X)

        # Step 2: Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Step 3: Train Decision Tree Classifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

        # Step 4: Predict on test set
y_pred = classifier.predict(X_test)

        # Step 5: Evaluate the classifier
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#Created data set on local products
Bean_data = local_data_filtered[local_data_filtered['product_Beans (Haricot)'] == 1]
Camels_data= local_data_filtered[local_data_filtered['product_Camels (Local Quality)']==1]
Casual_labour_data= local_data_filtered[local_data_filtered['product_Casual Labor (unskilled, daily, without food)']==1]
firewood_data = local_data_filtered[local_data_filtered['product_Firewood'] == 1]
Goat_data= local_data_filtered[local_data_filtered['product_Goats (Local Quality)']==1]
Horse_data= local_data_filtered[local_data_filtered['product_Horse beans']==1]
Maize_Grain_data= local_data_filtered[local_data_filtered['product_Maize Grain (White)']==1]
Teff_data= local_data_filtered[local_data_filtered['product_Mixed Teff']==1]
Oxen_data= local_data_filtered[local_data_filtered['product_Oxen (Local Quality)']==1]
Sheep_data= local_data_filtered[local_data_filtered['product_Sheep (Local Quality)']==1]
Sourghum_Red_data= local_data_filtered[local_data_filtered['product_Sorghum (Red)']==1]
Sourghum_white_data= local_data_filtered[local_data_filtered['product_Sorghum (White)']==1]
Sourghum_yellow_data= local_data_filtered[local_data_filtered['product_Sorghum (Yellow)']==1]
Wheatflour_data= local_data_filtered[local_data_filtered['product_Wheat Flour']==1]
Wheatg_data= local_data_filtered[local_data_filtered['product_Wheat Grain']==1]

# Filter out values above 1000
Bean_data = Bean_data[Bean_data['value'] <= 1000]
Casual_labour_data = Casual_labour_data[Casual_labour_data['value'] <= 1000]
firewood_data = firewood_data[firewood_data['value'] <= 1000]
Goat_data = Goat_data[Goat_data['value'] <= 1000]
Horse_data = Horse_data[Horse_data['value'] <= 1000]
Maize_Grain_data = Maize_Grain_data[Maize_Grain_data['value'] <= 1000]
Teff_data = Teff_data[Teff_data['value'] <= 1000]
Sheep_data = Sheep_data[Sheep_data['value'] <= 1000]
Sourghum_Red_data = Sourghum_Red_data[Sourghum_Red_data['value'] <= 1000]
Sourghum_white_data = Sourghum_white_data[Sourghum_white_data['value'] <= 1000]
Sourghum_yellow_data = Sourghum_yellow_data[Sourghum_yellow_data['value'] <= 1000]
Wheatflour_data = Wheatflour_data[Wheatflour_data['value'] <= 1000]
Wheatg_data = Wheatg_data[Wheatg_data['value'] <= 1000]

# Now plot the box plot with the modified data
plt.figure(figsize=(10, 6))
sns.boxplot(data=pd.concat([Bean_data.assign(Product='Beans'),
                            firewood_data.assign(Product='Firewood'),
                            Goat_data.assign(Product='Goats'),
                            Horse_data.assign(Product='Horse'),
                            Maize_Grain_data.assign(Product='Maize Grain'),
                            Teff_data.assign(Product='Teff'),
                            Sheep_data.assign(Product='Sheep'),
                            Sourghum_Red_data.assign(Product='Sorghum Red'),
                            Sourghum_white_data.assign(Product='Sorghum White'),
                            Sourghum_yellow_data.assign(Product='Sorghum Yellow'),
                            Wheatflour_data.assign(Product='Wheat Flour'),
                            Wheatg_data.assign(Product='Wheat Grain')]),
            x='market', y='value', hue='Product')
plt.title('Distribution of Value across Different Products in Various Markets')
plt.xlabel('Market')
plt.ylabel('Value')
plt.xticks(rotation=45)  
plt.ylim(0, 200) 
plt.tight_layout()
plt.show()
####---------------------------
####Vrushali Decision tree end
###-----------------------


