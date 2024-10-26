# road_accident_severity.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the dataset
data = pd.read_csv('road_accidents.csv')

# Display the first few rows of the dataset (for debugging)
print(data.head())

# Define the dependent variable (target) and independent variables (features)
# Assuming 'severity' is the column for accident severity
# Replace these with actual feature names from your dataset
X = data[['weather_condition', 'road_condition', 'time_of_day', 'vehicle_count']]  # Independent variables
y = data['severity']  # Dependent variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Save the model for future use
joblib.dump(model, 'road_accident_severity_model.joblib')

# Example prediction with hypothetical data
# Assuming we have the following hypothetical independent variables
hypothetical_data = pd.DataFrame({
    'weather_condition': [1],  # Replace with actual values
    'road_condition': [2],      # Replace with actual values
    'time_of_day': [1],        # Replace with actual values
    'vehicle_count': [10]      # Replace with actual values
})

predicted_severity = model.predict(hypothetical_data)
print(f'Predicted Accident Severity: {predicted_severity[0]}')
