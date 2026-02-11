import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Study hours 
X = np.array([1, 2, 3, 4, 5,  7, 8]).reshape(-1,1)

# Marks scored
y = np.array([35, 40, 50, 55, 65, 70,  85])

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict marks
prediction = model.predict([[9]])
print("Predicted Marks for 9 hours:", prediction[0])

# Plot graph
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Student Marks Predictio as LinearRegression")
plt.show()
