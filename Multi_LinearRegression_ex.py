import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#  (Study Hours, Attendance)
X = np.array([
    [2, 60],
    [3, 65],
    [4, 70],
    [5, 75],
    [6, 80],
    [7, 85],
    [8, 90]
])

# Marks
y = np.array([40, 50, 55, 65, 70, 80, 85])

# Train model
model = LinearRegression()
model.fit(X, y)

# Prediction 
prediction = model.predict([[9, 85]])
print("Predicted Marks:", prediction[0])

# Graph 
plt.scatter(X[:,0], y)
plt.plot(X[:,0], model.predict(X))
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Multiple Linear Regression (Hours vs Marks)")
plt.show()
