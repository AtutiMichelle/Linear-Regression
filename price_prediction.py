import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the dataset from the CSV file
df = pd.read_csv('Nairobi_Office_Price_Ex.csv')

# Extract SIZE (feature) and PRICE (target)
X = df['SIZE'].values
y = df['PRICE'].values

# Step 2: Define the Mean Squared Error (MSE) function
def mean_squared_error(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    return mse

# Step 3: Define the Gradient Descent function
def gradient_descent(X, y, m, c, learning_rate):
    N = len(X)
    y_pred = m * X + c  # Predicted values
    dm = (-2 / N) * np.sum(X * (y - y_pred))  # Gradient for slope (m)
    dc = (-2 / N) * np.sum(y - y_pred)  # Gradient for intercept (c)
    m = m - learning_rate * dm  # Update slope
    c = c - learning_rate * dc  # Update intercept
    return m, c

# Step 4: Train the model using Gradient Descent and track errors
def train_model(X, y, learning_rate=0.0001, epochs=10):
    m, c = np.random.randn(), np.random.randn()  # Initialize m and c
    errors = []  # To store the MSE for each epoch

    print(f"Initial m: {m}, Initial c: {c}")

    for epoch in range(epochs):
        y_pred = m * X + c  # Prediction
        error = mean_squared_error(y, y_pred)  # Calculate error
        errors.append(error)  # Store the error
        print(f"Epoch {epoch + 1}: MSE = {error:.4f}")
        m, c = gradient_descent(X, y, m, c, learning_rate)  # Update m and c

    print(f"Trained m: {m}, Trained c: {c}")
    return m, c, errors

# Step 5: Train the model for 10 epochs
m, c, errors = train_model(X, y, learning_rate=0.0001, epochs=10)

# Step 6: Predict office price for size = 100 sq. ft.
size = 100
predicted_price = m * size + c
print(f"Predicted Price for size {size} sq. ft.: {predicted_price:.2f}")

# Step 7: Plot the line of best fit
plt.figure(figsize=(10, 5))

# Subplot 1: Line of Best Fit
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, m * X + c, color='red', label='Line of Best Fit')
plt.xlabel('Office Size (sq. ft.)')
plt.ylabel('Office Price')
plt.title('Office Size vs Price - Linear Regression')
plt.legend()

# Subplot 2: MSE over Epochs
plt.subplot(1, 2, 2)
plt.plot(range(1, 11), errors, marker='o', color='purple')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('MSE over Epochs')
plt.grid(True)

plt.tight_layout()
plt.show()
