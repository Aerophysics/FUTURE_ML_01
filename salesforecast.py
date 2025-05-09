Question='''

Task 1

Sales Forecasting for Retail Business


🔹 Skills Gained: Time series forecasting, regression, trend analysis.

🔹 Tools: Python (Prophet, Scikit-learn, Pandas), Matplotlib.

🔹 Dataset: Sales data (daily, weekly, or monthly sales numbers).

🔹 Deliverable: A forecasting model that predicts future sales trends,
with visualizations of forecast accuracy and seasonality.


'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import kagglehub
from datetime import datetime
import warnings
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# Download dataset from Kaggle
path = kagglehub.dataset_download("kyanyoga/sample-sales-data")
df = pd.read_csv(f"{path}/sales_data_sample.csv", encoding='latin1')

# Display DataFrame info
print("\nDataFrame Info:")
print(df.info())
print("\nFirst few rows:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())

# Data Preprocessing
# Convert date column to datetime
df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
df = df.sort_values('ORDERDATE')

# Aggregate sales by date
daily_sales = df.groupby('ORDERDATE')['SALES'].sum().reset_index()
daily_sales.columns = ['ds', 'y']  # Prophet requires columns to be named 'ds' and 'y'
print(daily_sales.head())

# Create and train the Prophet model
# Prophet is a forecasting model developed by Facebook that works well with time series data
# We create it with the following parameters:
# - yearly_seasonality=True: Captures yearly patterns (e.g. holiday sales spikes)
# - weekly_seasonality=True: Captures weekly patterns (e.g. weekend vs weekday differences) 
# - daily_seasonality=True: Captures daily patterns (e.g. morning vs evening sales)
# - changepoint_prior_scale=0.05: Controls flexibility of trend, higher values = more flexible
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True, 
    daily_seasonality=True,
    changepoint_prior_scale=0.05
)

# Fit the model to our daily sales data
# The data must have columns 'ds' (dates) and 'y' (target values)
model.fit(daily_sales)

# Make future predictions
future_dates = model.make_future_dataframe(periods=90)  # Predict next 90 days
forecast = model.predict(future_dates)

# Visualize the results
plt.figure(figsize=(15, 10))

# Plot 1: Forecast
plt.subplot(2, 1, 1)
plt.plot(daily_sales['ds'], daily_sales['y'], label='Actual Sales')
plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='red')
plt.fill_between(forecast['ds'], 
                 forecast['yhat_lower'], 
                 forecast['yhat_upper'], 
                 color='red', alpha=0.1)
plt.title('Sales Forecast')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()

# Plot 2: Components
plt.subplot(2, 1, 2)
model.plot_components(forecast)
plt.tight_layout()
plt.savefig('sales_forecast.png')
plt.close()

# Calculate forecast accuracy metrics
# Get predictions for historical data
historical_forecast = forecast[forecast['ds'].isin(daily_sales['ds'])]
actual = daily_sales['y']
predicted = historical_forecast['yhat']

# Calculate metrics
mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
r2 = r2_score(actual, predicted)

print("\nForecast Accuracy Metrics:")
print(f"Mean Absolute Error: ${mae:,.2f}")
print(f"Root Mean Square Error: ${rmse:,.2f}")
print(f"R-squared Score: {r2:.4f}")

# Create sales categories for confusion matrix
def create_sales_categories(sales_data, num_categories=5):
    """Convert continuous sales data into categories"""
    percentiles = np.linspace(0, 100, num_categories + 1)
    thresholds = np.percentile(sales_data, percentiles)
    return np.digitize(sales_data, thresholds[:-1]).astype(int)  # Convert to integer categories

# Create categories for actual and predicted values
actual_categories = create_sales_categories(actual)
predicted_categories = create_sales_categories(predicted)

# Create confusion matrix
cm = confusion_matrix(actual_categories, predicted_categories)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f'Category {i+1}' for i in range(5)],
            yticklabels=[f'Category {i+1}' for i in range(5)])
plt.title('Confusion Matrix for Sales Categories')
plt.xlabel('Predicted Category')
plt.ylabel('Actual Category')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

# Calculate category-wise accuracy
category_accuracy = cm.diagonal().sum() / cm.sum()
print(f"\nCategory-wise Accuracy: {category_accuracy:.2%}")

# Save forecast results to CSV
forecast_results = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
forecast_results.columns = ['Date', 'Forecast', 'Lower_Bound', 'Upper_Bound']
forecast_results.to_csv('sales_forecast_results.csv', index=False)

print("\nForecast results have been saved to 'sales_forecast_results.csv'")
print("Visualization has been saved to 'sales_forecast.png'")
print("Confusion matrix has been saved to 'confusion_matrix.png'")
