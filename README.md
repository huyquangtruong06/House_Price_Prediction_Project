# House Price Prediction Project

## Project Description
This project focuses on building a machine learning model to predict house prices based on various house features. The model uses a Random Forest Regressor algorithm to provide accurate price predictions.

## Project Structure
house-price-prediction/
├── House_Price_Prediction_Project.py # Main Python script
└── train.csv # Dataset file

## Files Description
1. `House_Price_Prediction_Project.py`: Main Python script containing:
   - Data preprocessing
   - Feature selection
   - Model training
   - Model evaluation
2. `train.csv`: Dataset containing house features and corresponding sale prices

## Key Features
- Implements machine learning pipeline from data loading to prediction
- Uses multiple evaluation metrics for model performance
- Ready for extension with new features and models

## Implementation Steps
1. **Data Preparation**:
   - Load data from CSV file
   - Select relevant features:
     ```python
     features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 
                'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
     ```

2. **Data Splitting**:
   - Split data into training (80%) and testing (20%) sets

3. **Model Training**:
   - Train Random Forest Regressor model
   - Make predictions on new inputs

4. **Model Evaluation**:
   - Calculate evaluation metrics:
     - Mean Absolute Error (MAE)
     - Mean Squared Error (MSE)
     - Root Mean Squared Error (RMSE)
     - R² Score

## Requirements
- Python 3.x
- Required packages: pip install pandas numpy scikit-learn

## How to Run
1. Clone/download the project files
2. Ensure `train.csv` is in the same directory
3. Run the Python script: python House_Price_Prediction_Project.py

## Sample Output
The model provides predictions along with evaluation metrics:Random Forest Regressor Evaluation:
- Mean Absolute Error (MAE): XXXX.XX
- Mean Squared Error (MSE): XXXX.XX
- Root Mean Squared Error (RMSE): XXXX.XX
- R² Score: X.XXXX

## Future Improvements
- Experiment with additional features
- Perform hyperparameter tuning
- Test alternative algorithms (XGBoost, Neural Networks)
- Develop a web interface for predictions

## Author
[Huy Quang Truong]

