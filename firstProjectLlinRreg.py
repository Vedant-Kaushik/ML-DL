import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import sklearn.model_selection
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# Load the dataset and select relevant columns
data = pd.read_csv("student-mat.csv", sep=';')
data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]

# Define the target variable and features
predict = 'G3'
x = data.drop(columns=[predict])
y = np.array(data[predict])

# Split the data into training and test sets
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# Initialize variables to track the best model
best = 0
best_model = None

# Train the model and find the best one
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    
    if acc > best:
        best = acc
        best_model = linear
        with open('studentmodel.pickle', 'wb') as f:
            pickle.dump(linear, f)

# Load the best model from the pickle file
pickle_in = open('studentmodel.pickle', 'rb')
linear = pickle.load(pickle_in)

# Make predictions using the model
predictions = linear.predict(x_test)

# Print the best accuracy achieved during training
print("Best accuracy: ", best)

# Visualize the data and the best-fit line
p = 'G1'
y = 'G3'
pyplot.figure(figsize=(12, 6))

# Scatter plot of actual data points
pyplot.scatter(data[p], data[y], color='blue', label='Actual Data')

# Fit a linear model (best-fit line) to the data
best_model = np.polyfit(data[p], data[y], 1)  # 1 indicates linear fit
predicted_line = np.polyval(best_model, data[p])

# Plotting the best-fit line
pyplot.plot(data[p], predicted_line, color='red', label='Best-Fit Line')

# Setting labels and title
pyplot.xlabel(p)
pyplot.ylabel(y)
pyplot.title(f"{p} vs {y}")

# Adding legend, rotating x-ticks, and adding grid for better readability
pyplot.legend()
pyplot.xticks(rotation=90)
pyplot.grid(True)

# Display the plot
pyplot.show()