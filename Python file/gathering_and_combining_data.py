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

# Filter data for the desired year range and convert 'Year' to datetime format
gdelt_filtered = gdelt_data[(gdelt_data['Year'] >= 1990) & (gdelt_data['Year'] <= 2001)]
gdelt_filtered['Year'] = pd.to_datetime(gdelt_filtered['Year'], format='%Y')

# Categorize event descriptions based on frequency
event_descr_counts = gdelt_filtered["EventDescr"].value_counts()
quantiles = [0, 0.25, 0.5, 0.75, 1]
quantile_thresholds = event_descr_counts.quantile(quantiles)

def categorize_event(event_count):
    if event_count <= quantile_thresholds.iloc[1]:
        return "Low Frequency"
    elif event_count <= quantile_thresholds.iloc[2]:
        return "Medium-Low Frequency"
    elif event_count <= quantile_thresholds.iloc[3]:
        return "Medium-High Frequency"
    else:
        return "High Frequency"

gdelt_filtered['EventDescr_Category'] = gdelt_filtered['EventDescr'].map(event_descr_counts.apply(categorize_event))
# Correlation Matrix
numeric_columns = gdelt_filtered.select_dtypes(include=['float64', 'int64'])
gdelt_matrix = numeric_columns.corr()

# Plot Correlation Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(gdelt_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
# Excluded Event Descriptions
excluded_event_descr = [
    "Expel or deport individuals",
    "Confiscate property",
    "Coerce, not specified below",
    "Sexually assault",
    "Seize or damage property, not specified below",
    "Impose curfew",
    "Impose state of emergency or martial law",
    "Violate ceasefire",
    "Use as human shield"
]

# Filter data to exclude specific event descriptions
filtered_event_freq = event_descr_counts[~event_descr_counts.index.isin(excluded_event_descr)]

# Plot Filtered Event Frequencies
plt.figure(figsize=(12, 8))
filtered_event_freq.plot(kind='bar')
plt.title('Average Frequency of Events (Excluding Specific Types)')
plt.xlabel('Event Description')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.show()

'''
Frequency of types of events:

Null Hypothesis (H0): The type of event has no correlation with how many times it occurs.

Alternative Hypothesis (H1): Different types of events occur more often than others.
'''

# Select Top 20 Event Descriptions after excluding specific ones
top_20_filtered_event_freq = filtered_event_freq.head(20)

# Plot Top 20 Event Descriptions
plt.figure(figsize=(12, 8))
top_20_filtered_event_freq.plot(kind='bar')
plt.title('Average Frequency of Top 20 Event Descriptions (Excluding Specific Types)')
plt.xlabel('Event Description')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.show()

contingency_table1 = pd.crosstab(gdelt_filtered['EventDescr'], gdelt_filtered['EventDescr_Category'])

# Perform the Chi-Square Test
chi2, p_value, _, _ = chi2_contingency(contingency_table1)

# Interpret the results
alpha = 0.05
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p_value}")

if p_value < alpha:
    print("Reject the null hypothesis. There is a significant association between the type of event and its frequency.")
else:
    print("Fail to reject the null hypothesis. There is no significant association between the type of event and its frequency.")

contingency_table1

'''
Effect of Event Type on Goldstein Scale:

Null Hypothesis (H0): The type of event does not affect the Goldstein Scale score.

Alternative Hypothesis (H1): Different types of events have different average Goldstein Scale scores.
'''

# Mean Goldstein Scale by Top 50 Countries
country_goldstein = gdelt_filtered.groupby('CountryName')['GoldsteinScale'].mean().nlargest(50)

plt.figure(figsize=(12, 8))
plt.scatter(country_goldstein.index, country_goldstein.values, color='blue', alpha=0.5)
plt.title('Mean Goldstein Scale by Top 50 Countries')
plt.xlabel('Country')
plt.ylabel('Mean Goldstein Scale')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()
# Drop rows with missing values in the GoldsteinScale column
gdelt_filtered.dropna(subset=['GoldsteinScale'], inplace=True)

# Group the data by the type of event and calculate the average Goldstein Scale score for each group
event_goldstein_groups = gdelt_filtered.groupby('EventDescr')['GoldsteinScale'].apply(list)

# Perform the ANOVA test
f_statistic, p_value = f_oneway(*event_goldstein_groups)

# Interpret the results
alpha = 0.05
print(f"F-statistic: {f_statistic}")
print(f"P-value: {p_value}")

if p_value < alpha:
    print("Reject the null hypothesis. There are significant differences in the mean Goldstein Scale scores among the different types of events.")
else:
    print("Fail to reject the null hypothesis. There are no significant differences in the mean Goldstein Scale scores among the different types of events.")

'''
Relationship between Event Frequency and Time:

Null Hypothesis (H0): There is no significant change in the frequency of events over time.

Alternative Hypothesis (H1): The frequency of events has changed significantly over time.
'''

# Number of Events Over Time
events_by_year = gdelt_filtered.groupby('Year')['SumEvents'].sum()

plt.figure(figsize=(12, 8))
events_by_year.plot(kind='line', marker='o', color='blue')
plt.title('Number of Events Over Time')
plt.xlabel('Year')
plt.ylabel('Total Number of Events')
plt.grid(True)
plt.show()
X = sm.add_constant(gdelt_filtered.index)  # Index of the DataFrame as the time variable
y = gdelt_filtered['SumEvents']  # Frequency of events

# Fit linear regression model
model = sm.OLS(y, X).fit()

# Print summary statistics of the model
print(model.summary())

# Plot the data points
plt.figure(figsize=(12, 8))
plt.scatter(gdelt_filtered.index, gdelt_filtered['SumEvents'], color='blue', label='Data')

# Plot the regression line
plt.plot(gdelt_filtered.index, model.predict(X), color='red', label='Regression Line')

# Add labels and title
plt.title('Frequency of Events Over Time')
plt.xlabel('Time')
plt.ylabel('Frequency of Events')

# Add legend
plt.legend()

# Show plot
plt.show()
# Prepare the data
X = sm.add_constant(gdelt_filtered.index)  # Index of the DataFrame as the time variable
y = gdelt_filtered['SumEvents']  # Frequency of events

# Fit linear regression model
model = sm.OLS(y, X).fit()

# Perform hypothesis test on slope coefficient
slope_p_value = model.pvalues[1]  # Index 1 corresponds to the coefficient for the time variable

# Set significance level
alpha = 0.05

# Print results
if slope_p_value < alpha:
    print("Reject the null hypothesis. There is a significant change in the frequency of events over time.")
else:
    print("Fail to reject the null hypothesis. There is no significant change in the frequency of events over time.")

'''
Relationship between Number of Mentions and Number of Events:

Null Hypothesis (H0): There is no signifigance in the number of mentions and the number of events.

Alternative Hypothesis (H1): The number of events goes up with more mentions.
'''

# Selecting features (independent variables) and target (dependent variable)
X = gdelt_filtered[['SumEvents']]  # Features
y = gdelt_filtered['SumNumMentions']  # Target variable

# Creating and training the linear regression model
model = LinearRegression()
model.fit(X, y)

# Making predictions
y_pred = model.predict(X)

# Plotting
plt.figure(figsize=(12, 8))
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, y_pred, color='red', label='Linear Regression')
plt.title('Total Number of Mentions vs Total Number of Events')
plt.xlabel('Total Number of Events')
plt.ylabel('Total Number of Mentions')
plt.legend()
plt.grid(True)
plt.show()

# Print the coefficients of the model
print("Coefficients:", model.coef_)
# Calculate Pearson correlation coefficient and p-value
correlation_coefficient, p_value = pearsonr(gdelt_filtered['SumNumMentions'], gdelt_filtered['SumEvents'])

print("Pearson correlation coefficient:", correlation_coefficient)
print("P-value:", p_value)

# Compare p-value to significance level
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis. The correlation is statistically significant.")
else:
    print("Fail to reject the null hypothesis. The correlation is not statistically significant.")

'''
Comparison of Event Frequency between Countries:

Null Hypothesis (H0): There is no significant difference in the frequency of events between countries.

Alternative Hypothesis (H1): There are significant differences in the frequency of events between countries.
'''

# Total Number of Events by Top 50 Locations
events_by_location = gdelt_filtered.groupby('CountryName')['SumEvents'].sum().nlargest(50)

plt.figure(figsize=(12, 8))
events_by_location.sort_values(ascending=False).plot(kind='bar', color='blue')
plt.title('Total Number of Events by Top 50 Locations')
plt.xlabel('Location')
plt.ylabel('Total Number of Events')
plt.xticks(rotation=90)
plt.grid(axis='y')
plt.show()
# Group the data by country and calculate the total number of events for each country
events_by_country = gdelt_filtered.groupby('CountryName')['SumEvents'].apply(list)

# Perform the ANOVA test
f_statistic, p_value = stats.f_oneway(*events_by_country)

# Set significance level
alpha = 0.05

# Interpret the results
print(f"F-statistic: {f_statistic}")
print(f"P-value: {p_value}")

if p_value < alpha:
    print("Reject the null hypothesis. There are significant differences in the frequency of events between countries.")
else:
    print("Fail to reject the null hypothesis. There are no significant differences in the frequency of events between countries.")

'''
Impact of Average Tone on Event Frequency:

Null Hypothesis (H0): There is no relationship between the average tone and the frequency of events.

Alternative Hypothesis (H1): The average tone is associated with variations in the frequency of events.
'''

# Plot the scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=gdelt_filtered['AvgAvgTone'], y=gdelt_filtered['SumEvents'], color='blue', alpha=0.5)

# Plot the regression line
sns.regplot(x=gdelt_filtered['AvgAvgTone'], y=gdelt_filtered['SumEvents'], scatter=False, color='red')

# Add labels and title
plt.title('Relationship between Average Tone and Frequency of Events')
plt.xlabel('Average Tone')
plt.ylabel('Frequency of Events')

# Show plot
plt.grid(True)
plt.show()
# Calculate the residuals
residuals = gdelt_filtered['SumEvents'] - model.predict(X)

# Plot the residuals against the independent variable (Average Tone)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=gdelt_filtered['AvgAvgTone'], y=residuals, color='blue', alpha=0.5)

# Add a horizontal line at y=0 to show the mean
plt.axhline(y=0, color='red', linestyle='--')

# Add labels and title
plt.title('Residual Plot')
plt.xlabel('Average Tone')
plt.ylabel('Residuals (Observed - Predicted)')

# Show plot
plt.grid(True)
plt.show()
# Prepare the data
X = sm.add_constant(gdelt_filtered['AvgAvgTone'])  # Independent variable: average tone
y = gdelt_filtered['SumEvents']  # Dependent variable: frequency of events

# Fit linear regression model
model = sm.OLS(y, X).fit()

# Print model summary
print(model.summary())
# Selecting features (independent variables) and target (dependent variable)
X = gdelt_filtered[['SumEvents', 'SumNumMentions']]  # Features
y = gdelt_filtered['AvgAvgTone']  # Target variable

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions on the testing set
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Print the coefficients of the model
print("Coefficients:", model.coef_)

# Print other relevant statistics or perform hypothesis testing if needed
# Top 100 Events: Average Tone vs Total Number of Events
top_100_events = gdelt_filtered.groupby('EventDescr')['SumEvents'].sum().nlargest(100).index
top_100_data = gdelt_filtered[gdelt_filtered['EventDescr'].isin(top_100_events)]

plt.figure(figsize=(12, 8))
plt.scatter(top_100_data['AvgAvgTone'], top_100_data['SumEvents'], color='blue', alpha=0.5)
plt.title('Average Tone vs Total Number of Events for Top 100 Events')
plt.xlabel('Average Tone')
plt.ylabel('Total Number of Events')
plt.grid(True)
plt.show()

# Total Number of Mentions by Top 20 Countries
mentions_by_country = gdelt_filtered.groupby('CountryName')['SumNumMentions'].sum().nlargest(20)

plt.figure(figsize=(12, 8))
mentions_by_country.plot(kind='bar', color='blue')
plt.title('Total Number of Mentions by Top 20 Countries')
plt.xlabel('Country')
plt.ylabel('Total Number of Mentions')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()
# Pairplot of Numeric Columns
sns.pairplot(numeric_columns)
plt.show()
gdelt_filtered.describe()
gdelt_filtered.describe(include='object')

'''
Association between Event Type and Location:

Null Hypothesis (H0): There is no association between the type of event and the location.

Alternative Hypothesis (H1): The type of event is associated with specific locations.


Predictive Power of Event Attributes:

Null Hypothesis (H0): The event attributes (e.g., type, location, tone) do not significantly contribute to predicting event frequency.

Alternative Hypothesis (H1): The event attributes have predictive power for event frequency.
'''

# Create a contingency table
contingency_table2 = pd.crosstab(gdelt_filtered['EventDescr'], gdelt_filtered['CountryName'])

# Perform the Chi-Square Test
chi2, p_value, _, _ = chi2_contingency(contingency_table2)

# Interpret the results
alpha = 0.05
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p_value}")

if p_value < alpha:
    print("Reject the null hypothesis. There is a significant association between the type of event and the location.")
else:
    print("Fail to reject the null hypothesis. There is no significant association between the type of event and the location.")
contingency_table2.head()
extracted_columns = gdelt_filtered[['CountryName', 'SumEvents', 'SumNumMentions']]
extracted_columns.dropna(inplace=True)
dummy_columns = pd.get_dummies(extracted_columns, columns=['CountryName'], drop_first=True)
dummy_columns.head()
X = dummy_columns.drop(columns=['SumEvents'])
y = dummy_columns['SumEvents']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)
model = LinearRegression()
model.fit(X_train, y_train)
# Making predictions on the testing data
y_pred = model.predict(X_test)

# Evaluating the model (for example, using mean squared error)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
target_range = gdelt_filtered['SumEvents'].max() - gdelt_filtered['SumEvents'].min()
print("Range of SumEvents:", target_range)
# Plot the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.title('Actual vs. Predicted SumEvents')
plt.xlabel('Actual SumEvents')
plt.ylabel('Predicted SumEvents')
plt.grid(True)
plt.show()
residuals = y_test - y_pred

# Plot the residual plot
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, color='blue', alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Predicted SumEvents')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()
encoded_countries = pd.get_dummies(gdelt_filtered['CountryCode'])

# Concatenate the encoded countries with the original DataFrame
gdelt_encoded = pd.concat([gdelt_filtered, encoded_countries], axis=1)

columns_to_drop = ['CountryName', 'Year']

# Drop the original 'CountryName' column as it's no longer needed
gdelt_encoded.drop(columns=columns_to_drop, inplace=True)
gdelt_encoded
features = ['EventCode', 'SumNumMentions', 'GoldsteinScale', 'AvgAvgTone']
X = gdelt_encoded[features]

# Extracting the rows for the top 50 countries
top_50_countries = gdelt_filtered['CountryCode'].value_counts().nlargest(50).index
X_top_50_countries = gdelt_encoded[gdelt_encoded['CountryCode'].isin(top_50_countries)]
X_top_50_countries['Target'] = X_top_50_countries['CountryCode'].apply(lambda x: 1 if x in top_50_countries else 0)
# Create the target variable
y = X_top_50_countries['Target']
X_train_numeric = X_train.select_dtypes(include=[np.number])
X_test_numeric = X_test.select_dtypes(include=[np.number])

pd.set_option('display.max_rows', None)

# Train the logistic regression model
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train, y_train)
# Make predictions on the testing data
y_pred = log_reg_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
from sklearn.metrics import confusion_matrix


# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
# Extract the predicted probabilities or class labels for the test data
predicted_probabilities = log_reg_model.predict_proba(X_test)[:, 1]  # Probability of positive class (event happening)
predicted_labels = log_reg_model.predict(X_test)  # Predicted class labels

# Plot the predicted probabilities or class labels against the countries
plt.figure(figsize=(10, 6))
plt.scatter(predicted_probabilities, X_test.index, c=predicted_labels, cmap='coolwarm')
plt.xlabel('Predicted Probability of Event Happening')
plt.ylabel('Country')
plt.title('Logistic Regression Predictions')
plt.colorbar(label='Predicted Class (0: No Event, 1: Event)')
plt.show()


'''have to fix this, it will not convert the results back to names'''

# # Reset the indices of X_test
# X_test_reset_index = X_test.reset_index(drop=True)

# # Get the country names from the columns of encoded_countries
# country_names = encoded_countries.columns

# # Create an array of country names corresponding to the indices in X_test_reset_index
# predicted_countries = [country_names[idx] for idx in X_test_reset_index.index]

# # Plot the predicted probabilities or class labels against the countries
# plt.figure(figsize=(10, 6))
# plt.scatter(predicted_probabilities, predicted_countries, c=predicted_labels, cmap='coolwarm')
# plt.xlabel('Predicted Probability of Event Happening')
# plt.ylabel('Country')
# plt.title('Logistic Regression Predictions')
# plt.colorbar(label='Predicted Class (0: No Event, 1: Event)')
# plt.show()
