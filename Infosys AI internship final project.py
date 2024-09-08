# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import xgboost as xgb
import gradio as gr

# Set random seed for reproducibility
np.random.seed(42)

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the data from a CSV file.
    
    Args:
    file_path (str): Path to the CSV file.
    
    Returns:
    pd.DataFrame: Preprocessed DataFrame.
    """
    # Load the data
    df = pd.read_csv(file_path)
    
    # Filter for India and select relevant columns
    df = df[df['country'] == 'India']
    selected_columns = [
        'year', 'population', 'gdp', 'biofuel_electricity', 'coal_electricity',
        'gas_electricity', 'hydro_electricity', 'nuclear_electricity', 'oil_electricity',
        'other_renewable_electricity', 'solar_electricity', 'wind_electricity',
        'biofuel_elec_per_capita', 'coal_elec_per_capita', 'gas_elec_per_capita',
        'hydro_elec_per_capita', 'nuclear_elec_per_capita', 'oil_elec_per_capita',
        'other_renewables_elec_per_capita', 'solar_elec_per_capita', 'wind_elec_per_capita'
    ]
    df = df[selected_columns]
    
    # Filter for years >= 1990
    df = df[df['year'] >= 1990]
    
    # Fill missing values
    df['biofuel_elec_per_capita'] = df['biofuel_elec_per_capita'].fillna(0)
    df['biofuel_electricity'] = df['biofuel_electricity'].fillna(0)
    if 'population' in df.columns:
        df['population'] = df['population'].fillna(df['population'].mean())
    else:
        print("Warning: 'population' column is missing from the dataset.")
    
    # Create new features
    df['total_electricity_production'] = df[[col for col in df.columns if col.endswith('_electricity')]].sum(axis=1)
    df['total_electricity_consumption'] = df[[col for col in df.columns if col.endswith('_per_capita')]].sum(axis=1)
    
    return df.reset_index(drop=True)

def plot_electricity_consumption(df):
    """
    Plot the total electricity consumption over time.
    
    Args:
    df (pd.DataFrame): Preprocessed DataFrame.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df['year'], df['total_electricity_consumption'])
    plt.title('Total Electricity Consumption in India (1990-2020)')
    plt.xlabel('Year')
    plt.ylabel('Total Electricity Consumed (TWh)')
    plt.grid(True)
    plt.show()

def prepare_data_for_modeling(df):
    """
    Prepare the data for modeling by splitting into features and target,
    and performing train-test split.
    
    Args:
    df (pd.DataFrame): Preprocessed DataFrame.
    
    Returns:
    tuple: X_train, X_test, y_train, y_test, scaler
    """
    features = df.drop(columns=['total_electricity_consumption'])
    target = df['total_electricity_consumption'].values.reshape(-1, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y

def build_lstm_model(input_shape):
    """
    Build and compile an LSTM model.
    
    Args:
    input_shape (tuple): Shape of the input data.
    
    Returns:
    tensorflow.keras.models.Sequential: Compiled LSTM model.
    """
    model = Sequential([
        LSTM(100, activation='relu', input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm_model(X_train, y_train):
    """
    Train the LSTM model.
    
    Args:
    X_train (np.array): Training features.
    y_train (np.array): Training target.
    
    Returns:
    tensorflow.keras.models.Sequential: Trained LSTM model.
    """
    X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    model = build_lstm_model((1, X_train.shape[1]))
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train_reshaped, y_train, epochs=100, batch_size=32, validation_split=0.2, 
              callbacks=[early_stopping], verbose=0)
    return model

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest model.
    
    Args:
    X_train (np.array): Training features.
    y_train (np.array): Training target.
    
    Returns:
    RandomForestRegressor: Trained Random Forest model.
    """
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

def train_xgboost(X_train, y_train):
    """
    Train an XGBoost model.
    
    Args:
    X_train (np.array): Training features.
    y_train (np.array): Training target.
    
    Returns:
    xgboost.XGBRegressor: Trained XGBoost model.
    """
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)
    return xgb_model

def evaluate_model(model, X_test, y_test, scaler_y, model_name):
    """
    Evaluate the model and print performance metrics.
    
    Args:
    model: Trained model (LSTM, Random Forest, or XGBoost).
    X_test (np.array): Test features.
    y_test (np.array): Test target.
    model_name (str): Name of the model being evaluated.
    """
    if model_name == 'LSTM':
        X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        y_pred_scaled = model.predict(X_test_reshaped).flatten()
    else:
        y_pred_scaled = model.predict(X_test)
    
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler_y.inverse_transform(y_test).flatten()
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"{model_name} Model Performance:")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R2 Score: {r2:.2f}\n")

def predict_energy_consumption(model, scaler_X, scaler_y, input_data):
    """
    Predict energy consumption using the trained model.
    
    Args:
    model: Trained model (LSTM, Random Forest, or XGBoost).
    scaler (StandardScaler): Fitted scaler for feature normalization.
    input_data (dict): Dictionary containing input features.
    
    Returns:
    float: Predicted total electricity consumption.
    """
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler_X.transform(input_df)
    
    if isinstance(model, Sequential):
        input_reshaped = input_scaled.reshape((1, 1, input_scaled.shape[1]))
        prediction_scaled = model.predict(input_reshaped)
    else:
        prediction_scaled = model.predict(input_scaled.reshape(1, -1))
    
    prediction = scaler_y.inverse_transform(prediction_scaled)
    return prediction[0][0]

def main():
    # Load and preprocess data
    df = load_and_preprocess_data('energy_data.csv')
    
    # Plot electricity consumption
    plot_electricity_consumption(df)
    
    # Prepare data for modeling
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = prepare_data_for_modeling(df)
    
    # Train and evaluate LSTM model
    lstm_model = train_lstm_model(X_train, y_train)
    evaluate_model(lstm_model, X_test, y_test, scaler_y, 'LSTM')
    
    # Train and evaluate Random Forest model
    rf_model = train_random_forest(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test, scaler_y, 'Random Forest')
    
    # Train and evaluate XGBoost model
    xgb_model = train_xgboost(X_train, y_train)
    evaluate_model(xgb_model, X_test, y_test, scaler_y, 'XGBoost')    
    
    # Create Gradio interface
    input_components = [
        gr.Number(label="Year"),
        gr.Number(label="Population"),
        gr.Number(label="GDP"),
        gr.Number(label="Biofuel Electricity"),
        gr.Number(label="Coal Electricity"),
        gr.Number(label="Gas Electricity"),
        gr.Number(label="Hydro Electricity"),
        gr.Number(label="Nuclear Electricity"),
        gr.Number(label="Oil Electricity"),
        gr.Number(label="Other Renewable Electricity"),
        gr.Number(label="Solar Electricity"),
        gr.Number(label="Wind Electricity"),
        gr.Number(label="Biofuel Electricity Per Capita"),
        gr.Number(label="Coal Electricity Per Capita"),
        gr.Number(label="Gas Electricity Per Capita"),
        gr.Number(label="Hydro Electricity Per Capita"),
        gr.Number(label="Nuclear Electricity Per Capita"),
        gr.Number(label="Oil Electricity Per Capita"),
        gr.Number(label="Other Renewables Electricity Per Capita"),
        gr.Number(label="Solar Electricity Per Capita"),
        gr.Number(label="Wind Electricity Per Capita")
    ]
    
    output = gr.Number(label="Predicted Total Electricity Consumption")
    
    description = """
    This Energy Consumption Forecasting System predicts electricity consumption based on various factors.
    Input the required information, and the model will provide a forecast of total electricity consumption.
    """
    
    gr.Interface(
        fn=lambda *args: predict_energy_consumption(lstm_model, scaler_X, scaler_y, dict(zip(df.columns[:-1], args))),
        inputs=input_components,
        outputs=output,
        title="Energy Consumption Forecasting System",
        description=description
    ).launch()

if __name__ == "__main__":
    main()