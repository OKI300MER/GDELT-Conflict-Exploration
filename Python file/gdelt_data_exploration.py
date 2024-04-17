# Import libraries
import numpy as np
import pandas as pd
import json

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import folium
from folium import plugins
from folium.plugins import HeatMap
from folium.plugins import HeatMapWithTime


import scipy.stats as stats
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway
import statsmodels.api as sm
from scipy.stats import pearsonr

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mutual_info_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
gdelt_data = pd.read_csv(r'C:\Users\shric\Desktop\Dai\assignments\CAPSTONE\GDELT-Conflict-Exploration\data\gdelt_conflict_1_0.csv')

# Drop rows with missing values
gdelt_data.dropna(inplace=True)

def preprocess_data(data):
    # Perform necessary data preprocessing steps
    # Filter data for the desired year range and convert 'Year' to datetime format
    data['Year'] = pd.to_datetime(data['Year'], format='%Y')
    data = data[(data['Year'] >= '1990') & (data['Year'] <= '2001')]
    return data

def encode_categorical_features(data):
    # Encode categorical features
    encoder = LabelEncoder()
    data['CountryCode'] = encoder.fit_transform(data['CountryCode'])
    return data

def train_logistic_regression(X_train, y_train):
    # Train the logistic regression model
    log_reg_model = LogisticRegression()
    log_reg_model.fit(X_train, y_train)
    return log_reg_model

def evaluate_model(model, X_test, y_test):
    # Make predictions on the testing data
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def plot_predictions(X_test, predicted_probabilities, predicted_labels, country_names):
    # Plot the predicted probabilities or class labels against the countries
    plt.figure(figsize=(10, 6))
    plt.scatter(predicted_probabilities, X_test.index, c=predicted_labels, cmap='coolwarm')
    plt.xlabel('Predicted Probability of Event Happening')
    plt.ylabel('Country')
    plt.title('Logistic Regression Predictions')
    plt.colorbar(label='Predicted Class (0: No Event, 1: Event)')
    plt.show()

# Preprocess data
gdelt_data = preprocess_data(gdelt_data)

# Encode categorical features
gdelt_data = encode_categorical_features(gdelt_data)

# Define features and target variable
features = ['EventCode', 'SumNumMentions', 'GoldsteinScale', 'AvgAvgTone']
X = gdelt_data[features]
y = gdelt_data['Target']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
log_reg_model = train_logistic_regression(X_train, y_train)

# Evaluate the model
evaluate_model(log_reg_model, X_test, y_test)

# Extract the predicted probabilities or class labels for the test data
predicted_probabilities = log_reg_model.predict_proba(X_test)[:, 1]  # Probability of positive class (event happening)
predicted_labels = log_reg_model.predict(X_test)  # Predicted class labels

# Plot predictions
plot_predictions(X_test, predicted_probabilities, predicted_labels, encoded_countries.columns)