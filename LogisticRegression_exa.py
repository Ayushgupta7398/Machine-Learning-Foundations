import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Dataset
x = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
y = np.array([0, 0, 0, 1, 1, 1, 1, 1])

# Train model
model = LogisticRegression()
model.fit(x, y)

prediction = model.predict([[9]])
print("Predicted Marks for 9 hours:", prediction[0])



# Graph
plt.scatter(x, y)
plt.plot(x, model.predict(x))
plt.xlabel("Study Hours")
plt.ylabel("Pass (1) / Fail (0)")
plt.title("Logistic Regression - Student Pass Prediction")
plt.show()
