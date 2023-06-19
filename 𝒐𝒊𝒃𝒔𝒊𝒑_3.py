import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('car_data.csv')

# Select relevant features for the model
features = ['symboling', 'wheelbase', 'carlength', 'carwidth', 'carheight',
            'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio',
            'horsepower', 'peakrpm', 'citympg', 'highwaympg']

target = 'price'

X = data[features]
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using mean squared error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Example prediction for a car
sample_car = [[3, 88.6, 168.8, 64.1, 48.8, 2548, 130, 3.47, 2.68, 9, 111, 5000, 21, 27]]
predicted_price = model.predict(sample_car)
print(f"Predicted Price: {predicted_price[0]}")
