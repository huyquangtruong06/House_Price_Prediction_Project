#House Price Prediction Project
'''
Step 1 : Project Defitition 
    Goal : Predict the sales price for each house.
'''
#Import Libraries
import pandas as pd
import numpy as np

data = pd.read_csv("train.csv", index_col="Id")

'''
Step 2 : Feature Selection
    Choose features to train ML Model
    Need to use 'Feature Engineering' to identify Features needed
'''
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
'''
Step 3 : Spliting the dataset
    data : dataset
    X : data[features]
    Y : target variable (SalePrice)
    X,Y -> X_train, Y_train, X_Test, Y_Test
'''
X = data[features]
Y = data["SalePrice"]

from sklearn.model_selection import train_test_split

X_train, X_Test, Y_train, Y_Test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=1)
'''
Step 4 : Training Machine Learning Model
'''
from sklearn.tree import DecisionTreeRegressor

dt_model = DecisionTreeRegressor(random_state=1)
#Fit training data into model
dt_model.fit(X_train, Y_train)
y_predict = dt_model.predict(X_Test.head(5))
result = pd.DataFrame({'y' : Y_Test.head(5), 'y_predict' : y_predict})

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
rf_model = RandomForestRegressor(random_state=1)
#Fit training data into model
rf_model.fit(X_train, Y_train)
rf_val_predict = rf_model.predict(X_Test)
result = pd.DataFrame({'y' : Y_Test, 'y_predict' : rf_val_predict})


### Predict with a new input
pre = rf_model.predict([[6969,2021,1000,800,4,5,8]])
print(pre)

# Model Evaluation
'''
Step 5 : Model Evaluation
    Đánh giá độ chính xác của mô hình bằng các metrics:
    - Mean Absolute Error (MAE)
    - Mean Squared Error (MSE)
    - Root Mean Squared Error (RMSE)
    - R² Score
'''

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Dự đoán trên tập kiểm tra
rf_predictions = rf_model.predict(X_Test)

# Tính các metrics
mae = mean_absolute_error(Y_Test, rf_predictions)
mse = mean_squared_error(Y_Test, rf_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(Y_Test, rf_predictions)

# In kết quả đánh giá
print("Random Forest Regressor Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")