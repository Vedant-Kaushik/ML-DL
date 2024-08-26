import pandas as pd
from sklearn import linear_model
import pickle
import numpy as np
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import sklearn.model_selection
import matplotlib.pyplot as plt
from matplotlib import style

# Load the training and testing data
train_data = pd.read_csv("train.csv", sep=",")
test_data = pd.read_csv("test.csv", sep=",")

# Define the numerical columns
numerical_columns = [
    'Id',
    'MSSubClass',  # Often treated as numerical for modeling
    'LotFrontage',
    'LotArea',
    'YearBuilt',
    'YearRemodAdd',
    'MasVnrArea',
    'TotalBsmtSF',
    '1stFlrSF',
    '2ndFlrSF',
    'GrLivArea',
    'BsmtFullBath',
    'BsmtHalfBath',
    'FullBath',
    'HalfBath',
    'BedroomAbvGr',
    'KitchenAbvGr',
    'TotRmsAbvGrd',
    'Fireplaces',
    'GarageCars',
    'GarageArea',
    'WoodDeckSF',
    'OpenPorchSF',
    'EnclosedPorch',
    'ScreenPorch',
    'PoolArea',
    'MiscVal',
    'MoSold',
    'YrSold'
]

# Initialize the LabelEncoder for encoding categorical data
le = preprocessing.LabelEncoder()

# Define the categorical columns
categorical_columns = [
    'MSZoning',
    'Street',
    'Alley',
    'LotShape',
    'LandContour',
    'Utilities',
    'LotConfig',
    'LandSlope',
    'Neighborhood',
    'Condition1',
    'Condition2',
    'BldgType',
    'HouseStyle',
    'RoofStyle',
    'RoofMatl',
    'Exterior1st',
    'Exterior2nd',
    'MasVnrType',
    'ExterQual',
    'ExterCond',
    'Foundation',
    'BsmtQual',
    'BsmtCond',
    'BsmtExposure',
    'BsmtFinType1',
    'BsmtFinType2',
    'Heating',
    'HeatingQC',
    'CentralAir',
    'Electrical',
    'KitchenQual',
    'Functional',
    'FireplaceQu',
    'GarageType',
    'GarageFinish',
    'GarageQual',
    'GarageCond',
    'PavedDrive',
    'PoolQC',
    'Fence',
    'MiscFeature',
    'SaleType',
    'SaleCondition'
]

# Encode categorical columns for both train and test data
for col in categorical_columns:
    train_data[col] = le.fit_transform(list(train_data[col]))   
for col in categorical_columns:
    test_data[col] = le.fit_transform(list(test_data[col]))

# Combine numerical and encoded categorical columns
train_data = train_data[numerical_columns + categorical_columns + ['SalePrice']]
test_data = test_data[numerical_columns + categorical_columns]

# Impute missing values for numerical columns
num_imputer = SimpleImputer(strategy='mean')
train_data = pd.DataFrame(num_imputer.fit_transform(train_data), columns=train_data.columns)
test_data = pd.DataFrame(num_imputer.fit_transform(test_data), columns=test_data.columns)

# Define the target variable
predict = "SalePrice"
x_train_cv = train_data.drop(columns=[predict])
y_train_cv = np.array(train_data[predict])

# Split the data into training and cross-validation sets
x_train, x_cv, y_train, y_cv = sklearn.model_selection.train_test_split(x_train_cv, y_train_cv, test_size=0.1)

# Initialize variables to track the best model
best = 0
best_model = None

# Train the model and find the best one
for _ in range(10):
    x_train, x_cv, y_train, y_cv = sklearn.model_selection.train_test_split(x_train_cv, y_train_cv, test_size=0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_cv, y_cv)
    
    if acc > best:
        best = acc
        best_model = linear
        with open('houseprice.pickle', 'wb') as f:
            pickle.dump(linear, f)

# Load the best model from the pickle file
with open("houseprice.pickle", "rb") as f:
    best_model = pickle.load(f)

x_test=test_data
# Make predictions using the model
predictions = best_model.predict(x_test)
test_data['SalePrice'] = predictions

filtered_data = test_data[['Id', 'SalePrice']].astype(int)
# Save the predictions to a CSV file
filtered_data.to_csv("test_final.csv", index=False)

# Load the final data for visualization
final_data = pd.read_csv("test_final.csv", sep=',')

# Print the best accuracy
print(f"best Accuracy: {best:.4f}")

# Plotting the bar graph
p = 'Id'
style.use('ggplot')
fig, ax = plt.subplots(figsize=(12, 6))
plt.xticks(rotation=90)

# Adjust bar width as needed
bar_width = 0.8
bar_positions = range(len(final_data[p]))

# Plotting the bars
ax.scatter(final_data[p], final_data['SalePrice'], color='blue', edgecolor='black')
best_model = np.polyfit(final_data[p], final_data['SalePrice'], deg=60) 
predicted_line = np.polyval(best_model, final_data[p])

# Plotting the best-fit line
plt.plot(final_data[p], predicted_line, color='red', label='Best-Fit Line')

# Setting labels and title
ax.set_xlabel(p, fontsize=14)
ax.set_ylabel('Sale Price', fontsize=14)
ax.set_title('Predicted Sale Price vs ID', fontsize=16)

# Adding grid for better readability
ax.yaxis.grid(True, linestyle='--', linewidth=0.7)

# Adjusting x-axis tick rotation and hiding top and right spines
plt.xticks(rotation=45)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Tight layout for better spacing
plt.tight_layout() 
plt.show()