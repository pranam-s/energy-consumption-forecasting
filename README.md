# Energy Consumption Forecasting System

## Infosys Springboard AI Internship Final Project

This project is the culmination of my AI internship with Infosys Springboard. It demonstrates the application of various machine learning techniques to forecast energy consumption, showcasing the skills and knowledge gained during the internship.

### Project Context

As part of the Infosys Springboard AI internship, this final project aims to solve a real-world problem using artificial intelligence and machine learning. The Energy Consumption Forecasting System applies advanced data analysis and predictive modeling techniques to the critical area of energy management, aligning with the internship's focus on practical AI applications.

### Skills Demonstrated

- Data preprocessing and analysis
- Feature engineering
- Implementation of multiple machine learning models (LSTM, Random Forest, XGBoost)
- Model evaluation and comparison
- Creation of an interactive user interface for predictions

This project reflects the comprehensive training provided by Infosys Springboard, encompassing both theoretical knowledge and hands-on experience in AI and machine learning.

This project aims to predict electricity consumption in India based on various factors such as population, GDP, and different sources of electricity production. The system uses machine learning models to forecast future consumption, which can aid in better planning and management of energy resources.

## Project Overview

The Energy Consumption Forecasting System follows these main steps:

1. Data Loading and Preprocessing
2. Exploratory Data Analysis
3. Feature Engineering
4. Model Training (LSTM, Random Forest, XGBoost)
5. Model Evaluation
6. Interactive Prediction Interface

## Models Used

### 1. Long Short-Term Memory (LSTM)

LSTM is a type of recurrent neural network capable of learning long-term dependencies. In this project, we use a stacked LSTM model with dropout for regularization. The LSTM model is particularly suitable for this task due to its ability to capture temporal dependencies in time-series data.

Parameters:
- First LSTM layer: 100 units with ReLU activation
- Dropout: 0.2
- Second LSTM layer: 50 units with ReLU activation
- Dense layers: 25 units (ReLU) and 1 unit (output)
- Optimizer: Adam
- Loss function: Mean Squared Error

### 2. Random Forest

Random Forest is an ensemble learning method that operates by constructing multiple decision trees during training. It's known for its high accuracy and ability to handle large datasets with higher dimensionality.

Parameters:
- Number of estimators: 100
- Random state: 42

### 3. XGBoost

XGBoost (Extreme Gradient Boosting) is an optimized distributed gradient boosting library. It's known for its speed and performance, particularly in structured/tabular data.

Parameters:
- Number of estimators: 100
- Learning rate: 0.1
- Random state: 42

## How to Use

1. Clone this repository
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the main script: `python energy_consumption_forecast.py`
4. The script will load the data, train the models, and launch a Gradio interface for interactive predictions

## Data

The dataset used in this project includes historical data on electricity consumption and production from various sources, along with population and GDP data for India from 1990 to 2020.

## Results

The script will output performance metrics (Mean Absolute Error, Mean Squared Error, and R2 Score) for each model. You can compare these metrics to determine which model performs best on the test data.

## Interactive Prediction

The Gradio interface allows users to input various features and receive a prediction of total electricity consumption. This can be used to forecast future consumption based on projected values of the input features.
