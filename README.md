# Data Science Projects: Sales Forecasting, Stock Prediction, and Customer Support Chatbot

This repository contains three different data science projects that use machine learning to solve real-world problems. Let's break down each project and explain how they work!

## 📊 Project 1: Sales Forecasting

### What Does This Project Do?
This project helps businesses predict their future sales by looking at their past sales data. Think of it like predicting the weather - we use past weather patterns to predict future weather. Similarly, we use past sales data to predict future sales!

### How Does It Work?
1. **Data Collection**: 
   - We get sales data from a real business
   - The data includes daily sales numbers

2. **Data Processing**:
   - We clean up the data (like removing any mistakes)
   - We organize the data by date
   - We calculate daily total sales

3. **Making Predictions**:
   - We use a special tool called "Prophet" (created by Facebook)
   - It looks at patterns in the sales data
   - It finds trends (like if sales are going up or down)
   - It finds seasonal patterns (like if sales are higher during holidays)

4. **Results**:
   - We get predictions for future sales
   - We can see how accurate our predictions are
   - We create visual graphs to show the predictions
   - We save all results in files for easy viewing

### Files Created:
- `sales_forecast.png`: Shows the sales predictions
- `sales_forecast_results.csv`: Contains all the numbers
- `confusion_matrix.png`: Shows how accurate our predictions are

## 📈 Project 2: Stock Price Prediction

### What Does This Project Do?
This project helps predict whether a stock's price will go up or down in the future. It's like having a crystal ball for the stock market!

### How Does It Work?
1. **Data Collection**:
   - We get stock price data from Apple (AAPL)
   - The data includes daily prices, trading volume, etc.

2. **Data Processing**:
   - We calculate important numbers like:
     - Daily returns (how much the price changed)
     - Moving averages (average price over time)
     - Volatility (how much the price jumps around)

3. **Making Predictions**:
   - We use a special type of computer program called LSTM (Long Short-Term Memory)
   - It's really good at learning patterns over time
   - It looks at past prices to predict future prices

4. **Results**:
   - We get predictions for future stock prices
   - We can see how accurate our predictions are
   - We create visual graphs to show the predictions
   - We save all results in files for easy viewing

### Files Created:
- `stock_confusion_matrix.png`: Shows how accurate our predictions are
- `stock_training_history.png`: Shows how the model learned
- `stock_predictions.csv`: Contains all the predictions

## 🤖 Project 3: Customer Support Chatbot

This project implements a customer support chatbot using both Naive Bayes and LSTM models for ticket classification. The chatbot is accessible through a Streamlit web interface.

### Features

- Dual model approach (Naive Bayes and LSTM) for ticket classification
- Real-time chat interface using Streamlit
- Support for various types of customer inquiries
- Beautiful and responsive UI
- Chat history tracking

### Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the model training script first:
```bash
python chatbot.py
```

3. Launch the Streamlit app:
```bash
streamlit run app.py
```

### Usage

1. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)
2. Type your customer support query in the text input field
3. The chatbot will process your query and provide a response based on the classification

### Project Structure

- `chatbot.py`: Script for training and saving the machine learning models
- `app.py`: Streamlit web interface for the chatbot
- `requirements.txt`: List of Python dependencies
- `README.md`: Project documentation

### Model Information

The chatbot uses two models for classification:
1. Naive Bayes: Fast and efficient for text classification
2. LSTM: Deep learning model for better understanding of context and sequence

The models are trained on a customer support ticket dataset and can classify queries into different categories such as:
- Technical Support
- Billing Issues
- Product Information
- Account Management
- General Inquiries

## 🛠️ How to Run These Projects

### Requirements:
1. Python 3.8 or higher
2. Required Python packages (install using `pip install -r requirements.txt`):
   - pandas
   - numpy
   - matplotlib
   - seaborn
   - scikit-learn
   - tensorflow
   - prophet
   - kagglehub

### Running the Projects:
1. **Sales Forecasting**:
   ```bash
   python salesforecast.py
   ```

2. **Stock Price Prediction**:
   ```bash
   python stockprice.py
   ```

3. **Customer Support Chatbot**:
   ```bash
   python chatbot.py
   ```

## 📊 Understanding the Results

### Confusion Matrix:
- A confusion matrix is like a scorecard
- It shows how many predictions were right and wrong
- The diagonal line shows correct predictions
- Other numbers show where the model made mistakes

### Accuracy and Precision:
- Accuracy: How often the model is right overall
- Precision: How often the model is right when it makes a specific prediction

### Training History:
- Shows how the model learned over time
- The line going up means the model is getting better
- If the line flattens, the model has learned as much as it can

## 🔍 Tips for Understanding the Code

1. **Comments**: 
   - Look for lines starting with # - these explain what the code does
   - They're like notes to help you understand the code

2. **Variable Names**:
   - Names like `sales_data` or `predictions` tell you what the data is
   - They make the code easier to understand

3. **Functions**:
   - These are like recipes that tell the computer what to do
   - Each function does one specific job

4. **Visualizations**:
   - The graphs help you see the results
   - They make it easier to understand the predictions

## 🎓 Learning More

If you want to learn more about these topics:
1. **Sales Forecasting**: Look up "time series analysis" and "Prophet"
2. **Stock Prediction**: Learn about "LSTM" and "financial analysis"
3. **Chatbots**: Study "natural language processing" and "machine learning"

Remember: These projects are just starting points! You can modify them to make them better or use them for different purposes. 