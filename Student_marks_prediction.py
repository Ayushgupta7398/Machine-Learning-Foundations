import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
    'Hours_Studied': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Marks_Scored': [35, 42, 48, 55, 62, 70, 78, 85, 92]
}
df = pd.DataFrame(data)

# Input And output 
X = df[['Hours_Studied']]  
y = df['Marks_Scored']

#  Split the data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Initialize and Train  Model
model = LinearRegression()
model.fit(X_train, y_train)

#  Predict
y_pred = model.predict(X_test)

#  Evaluate Model
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R-squared Score: {r2_score(y_test, y_pred):.2f}")


hours = [[8.5]]
predicted_mark = model.predict(hours)
print(f"Predicted mark for 8.5 hours of study: {predicted_mark[0]:.2f}")

# Graphs
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.title('Hours vs Marks')
plt.xlabel('Hours Studied')
plt.ylabel('Marks Scored')
plt.legend()

plt.show()
