import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import os

# --- NEU: Seaborn-Stil für Plots verwenden ---
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['axes.formatter.use_locale'] = True

def berechne_indikatoren(df):
    """Berechnet technische Indikatoren und entfernt Zeilen mit NaN-Werten."""
    df_copy = df.copy()
    # RSI
    delta = df_copy['Close'].diff()
    gewinn = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    verlust = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gewinn / verlust
    df_copy['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df_copy['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df_copy['Close'].ewm(span=26, adjust=False).mean()
    df_copy['MACD'] = exp1 - exp2
    df_copy['Signal_Line'] = df_copy['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bänder
    df_copy['SMA_20'] = df_copy['Close'].rolling(window=20).mean()
    df_copy['Upper_Bollinger'] = df_copy['SMA_20'] + (df_copy['Close'].rolling(window=20).std() * 2)
    df_copy['Lower_Bollinger'] = df_copy['SMA_20'] - (df_copy['Close'].rolling(window=20).std() * 2)
    
    df_copy.dropna(inplace=True)
    return df_copy

# --- 1. Daten laden ---
print("Lade Daten von Yahoo Finance...")
tickers = ['TSLA', 'NVDA']
start_date = '2010-01-01'
data = yf.download(tickers, start=start_date, group_by='ticker')
print("Daten erfolgreich geladen.")

# --- 2. Datenvorbereitung (mit Indikatoren UND Ticker-ID) ---
print("Bereite Daten für das LSTM-Modell vor...")
sequence_length = 60
training_split_ratio = 0.8
num_tickers = len(tickers)
ticker_map = {ticker: i for i, ticker in enumerate(tickers)}

X_train, y_train = [], []
X_test, y_test = [], []
scalers = {}
feature_columns = []
processed_data_store = {}

for i, ticker in enumerate(tickers):
    stock_df = data[ticker].copy()
    stock_df.dropna(inplace=True)
    stock_df = berechne_indikatoren(stock_df)
    
    if i == 0:
        tech_indicators_cols = ['Close', 'Volume', 'RSI', 'MACD', 'Signal_Line', 'Upper_Bollinger', 'Lower_Bollinger']
        feature_columns = tech_indicators_cols + tickers
    
    # Technische Indikatoren als Basis-Features
    features_df = stock_df[tech_indicators_cols].copy()
    
    # Ticker-ID als One-Hot-Encoding hinzufügen
    for t in tickers:
        features_df[t] = 1 if t == ticker else 0

    # Daten in Trainings- und Testsets aufteilen (zeitlich)
    training_data_len = int(np.ceil(len(features_df) * training_split_ratio))
    
    # Scaler NUR auf den Trainingsdaten anpassen
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(features_df.iloc[:training_data_len])
    scalers[ticker] = scaler
    
    # Gesamte Feature-Matrix skalieren
    scaled_features = scaler.transform(features_df)
    
    # Trainingssequenzen erstellen
    train_data = scaled_features[:training_data_len]
    for j in range(sequence_length, len(train_data)):
        X_train.append(train_data[j-sequence_length:j, :])
        y_train.append(train_data[j, 0]) # Ziel ist immer noch der 'Close' Preis
        
    # Testsequenzen erstellen
    test_data = scaled_features[training_data_len - sequence_length:]
    for j in range(sequence_length, len(test_data)):
        X_test.append(test_data[j-sequence_length:j, :])
        y_test.append(test_data[j, 0])
        
    processed_data_store[ticker] = {
        'full_df': stock_df,
        'features_df': features_df,
        'training_len': training_data_len
    }

# In NumPy-Arrays umwandeln
X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)

# Reshape für LSTM: [samples, timesteps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], len(feature_columns)))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], len(feature_columns)))

print("Datenvorbereitung abgeschlossen.")
print(f"Anzahl der Features: {len(feature_columns)}")
print(f"Trainingsdaten Form: {X_train.shape}")
print(f"Testdaten Form: {X_test.shape}")

# --- 3. LSTM-Modell aufbauen ---
print("Baue das LSTM-Modell auf...")
model = Sequential([
    LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(units=32, return_sequences=False),
    Dropout(0.2),
    Dense(units=16),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# --- 4. Modell trainieren ---
print("Trainiere das Modell...")
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=30, # Etwas länger trainieren
    validation_data=(X_test, y_test)
)
print("Modell-Training abgeschlossen.")

# --- 5. Ergebnisse evaluieren und visualisieren ---
print("Evaluiere das Modell und erstelle Plots...")

if not os.path.exists('Simon/plots'):
    os.makedirs('Simon/plots')

# Trainings- und Validierungsverlust gemeinsam plotten
plt.figure(figsize=(14, 6))
plt.plot(history.history['loss'], label='Trainingsverlust', color='cornflowerblue')
plt.plot(history.history['val_loss'], label='Validierungsverlust', color='tomato')
plt.title('Trainings- & Validierungsverlust (Gemeinsam)')
plt.xlabel('Epochen')
plt.ylabel('Verlust (MSE)')
plt.legend()
plt.savefig('Simon/plots/training_validation_loss_combined.png')
plt.show()

# --- NEU: Trainingsverlust separat plotten ---
plt.figure(figsize=(14, 6))
plt.plot(history.history['loss'], label='Trainingsverlust', color='cornflowerblue')
plt.title('Trainingsverlust (Detailansicht)')
plt.xlabel('Epochen')
plt.ylabel('Verlust (MSE)')
plt.legend()
plt.savefig('Simon/plots/training_loss_detail.png')
plt.show()

# Vorhersagen machen
predictions_scaled = model.predict(X_test)
prediction_idx_start = 0

for ticker in tickers:
    proc_data = processed_data_store[ticker]
    full_df = proc_data['full_df']
    scaler = scalers[ticker]
    training_data_len = proc_data['training_len']
    test_set_len = len(full_df) - training_data_len
    
    prediction_idx_end = prediction_idx_start + test_set_len
    stock_predictions_scaled = predictions_scaled[prediction_idx_start:prediction_idx_end]
    
    # Korrekte Rücktransformation
    dummy_predictions = np.zeros((len(stock_predictions_scaled), len(feature_columns)))
    dummy_predictions[:, 0] = stock_predictions_scaled.flatten()
    stock_predictions = scaler.inverse_transform(dummy_predictions)[:, 0]
    
    prediction_idx_start = prediction_idx_end

    # Daten für den Plot vorbereiten
    train_df = full_df[:training_data_len]
    valid_df = full_df[training_data_len:].copy()
    valid_df['Predictions'] = stock_predictions

    # Wichtig: Metriken auf den tatsächlichen und vorhergesagten Werten berechnen
    y_actual = valid_df['Close'].values
    y_predicted = valid_df['Predictions'].values

    # NaN-Werte entfernen, falls welche durch die Zuweisung entstanden sind
    mask = ~np.isnan(y_predicted)
    y_actual = y_actual[mask]
    y_predicted = y_predicted[mask]

    mae = mean_absolute_error(y_actual, y_predicted)
    rmse = np.sqrt(mean_squared_error(y_actual, y_predicted))
    r2 = r2_score(y_actual, y_predicted)

    print(f"\n--- Evaluationsmetriken für {ticker} (Korrigiertes Modell) ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f} USD")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} USD")
    print(f"R-squared (R²): {r2:.4f}")
    print("-----------------------------------------------------------")

    # Ergebnisse plotten
    plt.figure(figsize=(15, 7))
    plt.title(f'Aktienkursprognose für {ticker} (Korrigiertes Modell)')
    plt.xlabel('Datum')
    plt.ylabel('Preis in USD')
    # --- NEUE FARBEN ---
    plt.plot(train_df.index, train_df['Close'], label='Trainingsdaten (Tatsächlich)', color='dodgerblue', alpha=0.8)
    plt.plot(valid_df.index, valid_df['Close'], color='darkorange', label='Testdaten (Tatsächlich)')
    plt.plot(valid_df.index, valid_df['Predictions'], color='red', linestyle='--', label='Vorhergesagter Preis')
    plt.legend()
    plt.savefig(f'Simon/plots/prediction_corrected_{ticker}.png')
    plt.show()

print("Skript vollständig ausgeführt.")
