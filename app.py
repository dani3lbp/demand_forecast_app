import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# Judul aplikasi
st.title("Forecasting Demand dengan XGBoost")
st.markdown("---")

# Sidebar untuk parameter interaktif
st.sidebar.header("Pengaturan Model")
file_upload = st.sidebar.file_uploader("Upload file Excel (opsional)", type="xlsx")
split_ratio = st.sidebar.slider("Train-Test Split Ratio", 0.6, 0.9, 0.8, 0.05)
forecast_weeks = st.sidebar.slider("Jumlah Minggu Forecast", 1, 52, 12)
lags = st.sidebar.slider("Jumlah Lag Features", 1, 12, 4)

# Load data
@st.cache_data
def load_data(file_path):
    if file_upload is not None:
        df = pd.read_excel(file_upload)
    else:
        df = pd.read_excel('try1.xlsx')  # Default file
    return df.sort_values('Week').reset_index(drop=True)

df = load_data('try1.xlsx')

# Preprocessing
st.subheader("Preprocessing Data")
Q1 = df['Historical_Sales_Data'].quantile(0.25)
Q3 = df['Historical_Sales_Data'].quantile(0.75)
IQR = Q3 - Q1
df_clean = df[(df['Historical_Sales_Data'] >= (Q1 - 1.5 * IQR)) & (df['Historical_Sales_Data'] <= (Q3 + 1.5 * IQR))].copy()

# Feature engineering
for i in range(1, lags + 1):
    df_clean[f'lag_{i}'] = df_clean['Historical_Sales_Data'].shift(i)
df_clean['rolling_mean'] = df_clean['Historical_Sales_Data'].rolling(window=lags).mean()
df_clean['rolling_std'] = df_clean['Historical_Sales_Data'].rolling(window=lags).std()
df_clean = df_clean.dropna()

feature_cols = [f'lag_{i}' for i in range(1, lags + 1)] + [
    'rolling_mean', 'rolling_std', 'Service_Level', 'Std_Dev', 'Z_Score', 'Lead_Time', 'Safety_Stock'
]

X = df_clean[feature_cols]
y = df_clean['Historical_Sales_Data']

# Split data
split_idx = int(split_ratio * len(df_clean))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
test_weeks = df_clean['Week'].iloc[split_idx:]

# Model training dengan tuning
if st.sidebar.button("Latih Model"):
    tscv = TimeSeriesSplit(n_splits=3)
    param_grid = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}
    grid_search = GridSearchCV(XGBRegressor(random_state=42, objective='reg:squarederror'), param_grid, cv=tscv, scoring='neg_mean_squared_error')
    with st.spinner("Melatih model..."):
        grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    st.sidebar.success(f"Best Params: {grid_search.best_params_}")

    # Prediksi test
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Metrik
    col1, col2 = st.columns(2)
    col1.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
    col2.metric("R-squared (R²)", f"{r2:.4f}")

    # Forecasting
    future_predictions = []
    last_features = X.iloc[-1].copy()
    last_week = df_clean['Week'].iloc[-1]
    for _ in range(forecast_weeks):
        pred = model.predict(last_features.values.reshape(1, -1))[0]
        future_predictions.append(pred)
        for i in range(lags, 1, -1):
            last_features[f'lag_{i}'] = last_features[f'lag_{i-1}']
        last_features[f'lag_1'] = pred
        last_features['rolling_mean'] = np.mean([last_features[f'lag_{j}'] for j in range(1, lags + 1)])
        last_features['rolling_std'] = np.std([last_features[f'lag_{j}'] for j in range(1, lags + 1)])
        last_week += 1

    future_weeks = np.arange(last_week - forecast_weeks + 1, last_week + 1)
    forecast_df = pd.DataFrame({'Week': future_weeks, 'Predicted_Sales': future_predictions})

    # Tampilkan tabel prediksi
    st.subheader("Prediksi 12 Minggu Mendatang")
    st.dataframe(forecast_df)

    # Download button
    csv_buffer = io.StringIO()
    forecast_df.to_csv(csv_buffer, index=False)
    st.download_button("Download Prediksi (CSV)", csv_buffer.getvalue(), "forecast_predictions.csv", "text/csv")

    # Visualisasi dengan Plotly
    st.subheader("Visualisasi")
    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], vertical_spacing=0.1, subplot_titles=('Forecasting Demand', 'Residuals'))
    
    # Plot utama
    fig.add_trace(go.Scatter(x=test_weeks, y=y_test, mode='lines', name='Data Historis Test', line=dict(color='blue', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=test_weeks, y=y_pred, mode='lines', name='Prediksi Test', line=dict(color='orange', width=2, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=future_weeks, y=future_predictions, mode='lines', name='Forecast', line=dict(color='green', width=2, dash='dash')), row=1, col=1)
    
    # Residuals
    residuals = y_test - y_pred
    fig.add_trace(go.Scatter(x=test_weeks, y=residuals, mode='markers', name='Residuals', marker=dict(color='red', size=6)), row=2, col=1)
    
    fig.update_layout(height=600, title_text='Forecasting Demand dengan XGBoost', showlegend=True, hovermode='x unified')
    fig.update_xaxes(title_text='Minggu', row=1, col=1)
    fig.update_yaxes(title_text='Penjualan Historis', row=1, col=1)
    fig.update_yaxes(title_text='Residuals', row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

# Deskripsi proyek untuk portfolio
with st.expander("Tentang Proyek Ini"):
    st.write("""
    Proyek ini mengimplementasikan model XGBoost untuk forecasting demand berdasarkan data historis penjualan. Fitur utama mencakup penanganan outliers, rekayasa fitur temporal, dan visualisasi interaktif. Cocok untuk demonstrasi skills data science dalam konteks supply chain.
    """)