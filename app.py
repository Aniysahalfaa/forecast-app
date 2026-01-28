"""
=============================================================================
PELINDO TPS SURABAYA - PETIKEMAS FORECASTING APPLICATION
Modern Streamlit App for Container Activity Forecasting using XGBoost
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import warnings
import base64
from pathlib import Path
warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="TPS Surabaya - Petikemas Forecasting",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# MODERN CSS STYLING
# =============================================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        padding: 0;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .logo-header {
        background: white;
        padding: 0.8rem 2rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin: -6rem -6rem 0 -6rem;
        position: relative;
        z-index: 10;
    }
    
    .logo-header img {
        height: 50px;
        width: auto;
    }
    
    .hero-section {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 25%, #546e7a 50%, #78909c 75%, #90a4ae 100%);
        padding: 8rem 2rem 7rem 2rem;
        margin: 0 -6rem 3rem -6rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(44,62,80,0.3);
        position: relative;
        overflow: hidden;
        min-height: 400px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .hero-section.with-background {
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-blend-mode: overlay;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(44,62,80,0.75) 0%, rgba(52,73,94,0.65) 25%, rgba(84,110,122,0.55) 50%, rgba(120,144,156,0.65) 75%, rgba(144,164,174,0.75) 100%);
        z-index: 1;
        mix-blend-mode: multiply;
    }
    
    .hero-section::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(ellipse at center, transparent 0%, rgba(44,62,80,0.4) 100%);
        z-index: 1;
    }
    
    .hero-content {
        position: relative;
        z-index: 2;
        width: 100%;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .hero-title {
        color: #FFFFFF !important;
        font-size: 3.2rem;
        font-weight: 800;
        font-family: 'Poppins', sans-serif;
        margin: 0 auto 1.5rem auto;
        text-shadow: 3px 3px 12px rgba(0,0,0,0.6);
        line-height: 1.2;
        letter-spacing: -0.5px;
        text-align: center;
        display: block;
        width: 100%;
    }
    
    .hero-subtitle {
        color: #e0e0e0;
        font-size: 1.3rem;
        font-weight: 400;
        font-family: 'Poppins', sans-serif;
        margin: 0 auto;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.5);
        max-width: 800px;
        text-align: center;
        line-height: 1.5;
        display: block;
        width: 100%;
        padding: 0 1rem;
    }
    
    .section-header {
        font-size: 2rem;
        font-weight: 700;
        font-family: 'Poppins', sans-serif;
        color: #2c3e50;
        margin: 3rem 0 1.5rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 4px solid;
        border-image: linear-gradient(90deg, #546e7a 0%, #90a4ae 100%) 1;
        display: inline-block;
    }
    
    .section-subheader {
        font-size: 1.1rem;
        color: #546E7A;
        margin-bottom: 2rem;
    }
    
    .modern-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(44,62,80,0.12);
        border: 1px solid rgba(144,164,174,0.3);
        border-top: 4px solid;
        border-image: linear-gradient(90deg, #546e7a 0%, #90a4ae 100%) 1;
        border-image-slice: 1 0 0 0;
        margin-bottom: 2rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .modern-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(44,62,80,0.18);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #eceff1 0%, #cfd8dc 50%, #b0bec5 100%);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border-left: 4px solid #546e7a;
        box-shadow: 0 4px 15px rgba(84,110,122,0.2);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        font-family: 'Poppins', sans-serif;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #546e7a;
        font-weight: 600;
        font-family: 'Poppins', sans-serif;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #2c3e50 0%, #546e7a 50%, #78909c 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        font-family: 'Poppins', sans-serif;
        padding: 0.8rem 2rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 15px rgba(84,110,122,0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #78909c 0%, #546e7a 50%, #2c3e50 100%);
        box-shadow: 0 6px 25px rgba(84,110,122,0.6);
        transform: translateY(-2px);
    }
    
    .stDownloadButton>button {
        background: linear-gradient(135deg, #546e7a 0%, #78909c 50%, #90a4ae 100%);
        box-shadow: 0 4px 15px rgba(120,144,156,0.4);
        font-family: 'Poppins', sans-serif;
    }
    
    .stDownloadButton>button:hover {
        background: linear-gradient(135deg, #90a4ae 0%, #78909c 50%, #546e7a 100%);
        box-shadow: 0 6px 25px rgba(120,144,156,0.5);
    }
    
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #2c3e50 0%, #546e7a 100%);
        border-radius: 10px;
    }
    
    .uploadedFile {
        border-radius: 12px;
        border: 2px dashed #546e7a;
        background: #eceff1;
    }
    
    .success-message {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        color: #2E7D32;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .info-message {
        background: linear-gradient(135deg, #eceff1 0%, #cfd8dc 100%);
        color: #2c3e50;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #546e7a;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .warning-message {
        background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%);
        color: #E65100;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #FF9800;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .streamlit-expanderHeader {
        background: #eceff1;
        border-radius: 8px;
        font-weight: 600;
        font-family: 'Poppins', sans-serif;
        color: #2c3e50;
    }
    
    hr {
        margin: 3rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, #546e7a 30%, #78909c 70%, transparent 100%);
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animate-fade-in {
        animation: fadeInUp 0.6s ease-out;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.5rem 0.5rem 0.5rem 0;
    }
    
    .badge-success {
        background: #E8F5E9;
        color: #2E7D32;
        border: 1px solid #4CAF50;
    }
    
    .badge-processing {
        background: #eceff1;
        color: #2c3e50;
        border: 1px solid #546e7a;
    }
    
    .badge-info {
        background: #cfd8dc;
        color: #2c3e50;
        border: 1px solid #78909c;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_logo_base64():
    """Try to load logo, return base64 or placeholder"""
    try:
        logo_path = Path("logo.png")
        if logo_path.exists():
            with open(logo_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
    except:
        pass
    return None

def get_background_base64():
    """Try to load background image, return base64 or None"""
    try:
        bg_path = Path("background.jpg")
        if bg_path.exists():
            with open(bg_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        bg_path = Path("background.png")
        if bg_path.exists():
            with open(bg_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
    except:
        pass
    return None

@st.cache_data
def preprocess_data(uploaded_file):
    """
    Preprocess Excel data with 3-level header structure
    Flexibly handles any number of year sheets (1+)
    """
    excel_file = pd.ExcelFile(uploaded_file)
    all_sheet_names = excel_file.sheet_names
    
    year_sheets = []
    for sheet in all_sheet_names:
        try:
            year = int(sheet)
            if 2000 <= year <= 2100:
                year_sheets.append(sheet)
        except ValueError:
            continue
    
    year_sheets = sorted(year_sheets)
    
    if len(year_sheets) == 0:
        raise ValueError("Tidak ada sheet dengan nama tahun yang valid (contoh: 2023, 2024, 2025)")
    
    all_data = []
    
    for year in year_sheets:
        df = pd.read_excel(uploaded_file, sheet_name=year, header=[0, 1, 2])
        
        new_columns = []
        for col in df.columns:
            level_0, level_1, level_2 = col[0], col[1], col[2]
            
            if level_1 == 'Week':
                new_columns.append('Week')
            elif level_1 == 'Tanggal':
                new_columns.append('Tanggal')
            elif level_1 == 'Hari':
                new_columns.append('Hari')
            elif level_1 == 'ket':
                new_columns.append(None)
            elif level_1 == 'Shift III':
                new_columns.append(f'{level_2}_S3' if 'Unnamed' not in str(level_2) else None)
            elif level_1 == 'Shift I':
                new_columns.append(f'{level_2}_S1' if 'Unnamed' not in str(level_2) else None)
            elif level_1 == 'Shift II':
                new_columns.append(f'{level_2}_S2' if 'Unnamed' not in str(level_2) else None)
            elif 'Actual' in level_1:
                new_columns.append(f'{level_2}_Actual_{year}' if 'Unnamed' not in str(level_2) else None)
            else:
                new_columns.append(f'{level_1}_{level_2}' if 'Unnamed' not in str(level_2) else None)
        
        df.columns = new_columns
        df = df.loc[:, df.columns.notnull()]
        df['Year'] = year
        
        df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
        numeric_cols = [col for col in df.columns if col not in ['Week', 'Tanggal', 'Hari', 'Year']]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(how='all', subset=numeric_cols)
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sort_values('Tanggal').reset_index(drop=True)
    
    actual_activities = ['Receiving', 'Delivery', 'Discharge', 'Loading', 'Export', 'Import']
    for activity in actual_activities:
        cols_to_merge = [col for col in combined_df.columns if f'{activity}_Actual_' in col]
        if cols_to_merge:
            combined_df[f'{activity}_Actual'] = combined_df[cols_to_merge].bfill(axis=1).iloc[:, 0]
            combined_df = combined_df.drop(columns=cols_to_merge)
    
    if 'Week' in combined_df.columns:
        combined_df = combined_df.drop(columns=['Week'])
    
    base_cols = ['Tanggal', 'Hari', 'Year']
    activity_order = ['Receiving', 'Delivery', 'Discharge', 'Loading', 'Relokasi', 'Behandle']
    
    s3_cols = [f'{act}_S3' for act in activity_order if f'{act}_S3' in combined_df.columns]
    s1_cols = [f'{act}_S1' for act in activity_order if f'{act}_S1' in combined_df.columns]
    s2_cols = [f'{act}_S2' for act in activity_order if f'{act}_S2' in combined_df.columns]
    actual_cols = [f'{act}_Actual' for act in actual_activities if f'{act}_Actual' in combined_df.columns]
    
    ordered_cols = base_cols + s3_cols + s1_cols + s2_cols + actual_cols
    combined_df = combined_df[ordered_cols]
    
    metadata = {
        'years_found': year_sheets,
        'latest_year': int(year_sheets[-1]),
        'earliest_year': int(year_sheets[0]),
        'total_years': len(year_sheets)
    }
    
    return combined_df, metadata

def create_features(df, target_col):
    """Create time series features for XGBoost"""
    df = df.copy()
    
    df['day_of_week'] = df['Tanggal'].dt.dayofweek
    df['month'] = df['Tanggal'].dt.month
    df['week_of_year'] = df['Tanggal'].dt.isocalendar().week
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    df['lag_1'] = df[target_col].shift(1)
    df['lag_7'] = df[target_col].shift(7)
    df['lag_14'] = df[target_col].shift(14)
    df['lag_30'] = df[target_col].shift(30)
    
    df['rolling_mean_7'] = df[target_col].rolling(7).mean()
    df['rolling_std_7'] = df[target_col].rolling(7).std()
    df['rolling_mean_30'] = df[target_col].rolling(30).mean()
    df['rolling_std_30'] = df[target_col].rolling(30).std()
    
    return df

def train_models(df, progress_container):
    """Train XGBoost models for all 24 target columns"""
    split_index = int(len(df) * 0.9)
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    
    actual_targets = ['Receiving_Actual', 'Delivery_Actual', 'Discharge_Actual',
                      'Loading_Actual', 'Export_Actual', 'Import_Actual']
    
    shift_targets = [
        'Receiving_S1', 'Delivery_S1', 'Discharge_S1', 'Loading_S1', 'Relokasi_S1', 'Behandle_S1',
        'Receiving_S2', 'Delivery_S2', 'Discharge_S2', 'Loading_S2', 'Relokasi_S2', 'Behandle_S2',
        'Receiving_S3', 'Delivery_S3', 'Discharge_S3', 'Loading_S3', 'Relokasi_S3', 'Behandle_S3'
    ]
    
    all_targets = actual_targets + shift_targets
    feature_cols = [
        'day_of_week', 'month', 'week_of_year', 'is_weekend',
        'lag_1', 'lag_7', 'lag_14', 'lag_30',
        'rolling_mean_7', 'rolling_std_7',
        'rolling_mean_30', 'rolling_std_30'
    ]
    
    all_models = {}
    all_metrics = {}
    total = len(all_targets)
    
    progress_bar = progress_container.progress(0)
    status_text = progress_container.empty()
    
    for i, target in enumerate(all_targets):
        status_text.markdown(f'<div class="info-message">🔄 Training model {i+1}/{total}: <strong>{target}</strong></div>', unsafe_allow_html=True)
        
        train_data = create_features(train_df, target).dropna()
        test_data = create_features(test_df, target).dropna()
        
        X_train = train_data[feature_cols]
        y_train = train_data[target]
        X_test = test_data[feature_cols]
        y_test = test_data[target]
        
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train, verbose=False)
        
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        all_models[target] = model
        all_metrics[target] = {'MAE': mae, 'RMSE': rmse}
        
        progress_bar.progress((i + 1) / total)
    
    status_text.markdown('<div class="success-message">✅ Training completed successfully!</div>', unsafe_allow_html=True)
    
    return all_models, all_metrics, train_df, test_df

def forecast_future(df, models, n_days=375, progress_container=None):
    """Generate recursive forecasts for n_days ahead"""
    actual_targets = ['Receiving_Actual', 'Delivery_Actual', 'Discharge_Actual',
                      'Loading_Actual', 'Export_Actual', 'Import_Actual']
    shift_targets = [
        'Receiving_S1', 'Delivery_S1', 'Discharge_S1', 'Loading_S1', 'Relokasi_S1', 'Behandle_S1',
        'Receiving_S2', 'Delivery_S2', 'Discharge_S2', 'Loading_S2', 'Relokasi_S2', 'Behandle_S2',
        'Receiving_S3', 'Delivery_S3', 'Discharge_S3', 'Loading_S3', 'Relokasi_S3', 'Behandle_S3'
    ]
    all_targets = actual_targets + shift_targets
    
    forecast_start = df['Tanggal'].max() + timedelta(days=1)
    forecast_dates = pd.date_range(start=forecast_start, periods=n_days, freq='D')
    
    forecast_df = df.tail(30).copy()
    forecast_results = {target: [] for target in all_targets}
    
    if progress_container:
        progress_bar = progress_container.progress(0)
        status_text = progress_container.empty()
    
    for i, date in enumerate(forecast_dates):
        if progress_container and (i + 1) % 50 == 0:
            status_text.markdown(f'<div class="info-message">📊 Forecasting day {i+1}/{n_days}</div>', unsafe_allow_html=True)
            progress_bar.progress((i + 1) / n_days)
        
        day_forecasts = {}
        
        for target in all_targets:
            if target in forecast_df.columns:
                recent_vals = forecast_df[target].tail(30).values
            else:
                recent_vals = np.zeros(30)
            
            features = {
                'day_of_week': date.dayofweek,
                'month': date.month,
                'week_of_year': date.isocalendar().week,
                'is_weekend': int(date.dayofweek >= 5),
                'lag_1': recent_vals[-1] if len(recent_vals) >= 1 else 0,
                'lag_7': recent_vals[-7] if len(recent_vals) >= 7 else 0,
                'lag_14': recent_vals[-14] if len(recent_vals) >= 14 else 0,
                'lag_30': recent_vals[-30] if len(recent_vals) >= 30 else 0,
                'rolling_mean_7': recent_vals[-7:].mean() if len(recent_vals) >= 7 else 0,
                'rolling_std_7': recent_vals[-7:].std() if len(recent_vals) >= 7 else 0,
                'rolling_mean_30': recent_vals.mean(),
                'rolling_std_30': recent_vals.std()
            }
            
            X_pred = pd.DataFrame([features])
            prediction = models[target].predict(X_pred)[0]
            prediction = max(0, prediction)
            
            day_forecasts[target] = prediction
            forecast_results[target].append(prediction)
        
        new_row = pd.DataFrame({'Tanggal': [date], **day_forecasts})
        forecast_df = pd.concat([forecast_df, new_row], ignore_index=True)
    
    if progress_container:
        status_text.markdown('<div class="success-message">✅ Forecasting completed!</div>', unsafe_allow_html=True)
        progress_bar.progress(1.0)
    
    return forecast_results, forecast_dates

def create_output_excel(forecast_results, forecast_dates, forecast_year):
    """Create formatted Excel output - dynamically uses the forecast year"""
    output_df = pd.DataFrame({
        'Week': ((forecast_dates - forecast_dates[0]).days // 7) + 1,
        'Tanggal': forecast_dates.strftime('%d/%m/%Y'),
        'Hari': forecast_dates.day_name()
    })
    
    activities = ['Receiving', 'Delivery', 'Discharge', 'Loading', 'Relokasi', 'Behandle']
    for activity in activities:
        col = f'{activity}_S3'
        output_df[f'Shift III_{activity}'] = np.round(forecast_results[col]).astype(int)
    
    for activity in activities:
        col = f'{activity}_S1'
        output_df[f'Shift I_{activity}'] = np.round(forecast_results[col]).astype(int)
    
    for activity in activities:
        col = f'{activity}_S2'
        output_df[f'Shift II_{activity}'] = np.round(forecast_results[col]).astype(int)
    
    actual_activities = ['Receiving', 'Delivery', 'Discharge', 'Loading', 'Export', 'Import']
    for activity in actual_activities:
        col = f'{activity}_Actual'
        output_df[f'Actual {forecast_year}_{activity}'] = np.round(forecast_results[col]).astype(int)
    
    output_df['ket'] = ''
    
    return output_df

def create_modern_visualization(df, forecast_results, forecast_dates, metrics):
    """Create modern interactive Plotly visualization"""
    actual_targets = ['Receiving_Actual', 'Delivery_Actual', 'Discharge_Actual',
                      'Loading_Actual', 'Export_Actual', 'Import_Actual']
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f"{t.replace('_Actual', '')}<br>MAE: {metrics[t]['MAE']:.1f} | RMSE: {metrics[t]['RMSE']:.1f}" 
                       for t in actual_targets],
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    colors = {
        'historical': '#1E88E5',
        'forecast': '#FF9800',
        'line': '#4CAF50'
    }
    
    # Convert forecast_dates to list of Timestamp objects for Plotly compatibility
    forecast_dates_list = forecast_dates.tolist()
    forecast_start_date = forecast_dates_list[0]
    
    for idx, target in enumerate(actual_targets):
        row = idx // 3 + 1
        col = idx % 3 + 1
        
        hist_data = df.tail(90)
        
        # Historical data
        fig.add_trace(
            go.Scatter(
                x=hist_data['Tanggal'].tolist(),
                y=hist_data[target],
                name='Historical',
                mode='lines',
                line=dict(color=colors['historical'], width=2),
                showlegend=(idx == 0)
            ),
            row=row, col=col
        )
        
        # Forecast data
        fig.add_trace(
            go.Scatter(
                x=forecast_dates_list,
                y=forecast_results[target],
                name='Forecast',
                mode='lines',
                line=dict(color=colors['forecast'], width=2, dash='dash'),
                showlegend=(idx == 0)
            ),
            row=row, col=col
        )
        
        # Add vertical line for forecast start using add_shape (compatible with all Plotly versions)
        fig.add_shape(
            type="line",
            x0=forecast_start_date,
            x1=forecast_start_date,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color=colors['line'], width=2, dash='dot'),
            row=row, col=col
        )
    
    fig.update_layout(
        title=dict(
            text='<b>Forecast Results: Container Activities (Next 375 Days)</b>',
            font=dict(size=20, color='#2c3e50'),
            x=0.5,
            xanchor='center'
        ),
        height=700,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Inter, sans-serif')
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E0E0E0')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E0E0E0')
    
    return fig

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    # LOGO HEADER (Kiri Atas - Kecil)
    logo_b64 = get_logo_base64()
    
    if logo_b64:
        st.markdown(f"""
            <div class="logo-header">
                <img src="data:image/png;base64,{logo_b64}">
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="logo-header">
                <div style="display: flex; align-items: center; gap: 0.8rem;">
                    <span style="font-size: 2rem;">🚢</span>
                    <div>
                        <div style="font-size: 1.2rem; font-weight: 700; color: #2c3e50; line-height: 1;">PELINDO</div>
                        <div style="font-size: 0.75rem; color: #546e7a; font-weight: 600; line-height: 1;">TPS SURABAYA</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # HERO SECTION with Background Image
    background_b64 = get_background_base64()

    if background_b64:
        st.markdown(f"""
            <div class="hero-section" style="background-image: url('data:image/jpeg;base64,{background_b64}');">
                <div class="hero-content">
                <center>
                    <h1 class="hero-title">Forecasting Aktivitas Petikemas</h1>
                    <p class="hero-subtitle">Prediksi Aktivitas Container Menggunakan Metode Machine Learning XGBoost</p>
                </center>
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="hero-section">
                <div class="hero-content">
                    <center>
                    <h1 class="hero-title">Forecasting Aktivitas Petikemas</h1>
                    <p class="hero-subtitle">Prediksi Aktivitas Container dengan Machine Learning XGBoost</p>
                </center>
                </div>
            </div>
        """, unsafe_allow_html=True)

    
    # SECTION 1: UPLOAD DATA
    st.markdown('<h2 class="section-header">📁 Upload Data</h2>', unsafe_allow_html=True)
    st.markdown('<p class="section-subheader">Upload file Excel yang berisi data aktivitas petikemas tahun 2023-2025</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_file = st.file_uploader(
            "Pilih file Excel",
            type=['xlsx', 'xls'],
            help="File harus memiliki minimal 1 sheet dengan nama tahun (contoh: 2023, 2024, 2025)"
        )
    
    if uploaded_file is not None:
        try:
            # Process data
            if 'data_processed' not in st.session_state:
                with st.spinner("🔄 Memproses data..."):
                    df, metadata = preprocess_data(uploaded_file)
                    st.session_state['df'] = df
                    st.session_state['metadata'] = metadata
                    st.session_state['data_processed'] = True
            
            df = st.session_state['df']
            metadata = st.session_state['metadata']
            
            # Calculate forecast year (latest year + 1)
            forecast_year = metadata['latest_year'] + 1
            
            st.markdown('<div class="success-message">✅ Data berhasil dimuat dan diproses!</div>', unsafe_allow_html=True)
            
            # Display metadata info
            st.markdown(f"""
                <div class="info-message">
                    📅 <strong>Data terdeteksi:</strong> {metadata['total_years']} tahun 
                    ({metadata['earliest_year']} - {metadata['latest_year']})<br>
                    🎯 <strong>Prediksi untuk:</strong> Tahun {forecast_year}
                </div>
            """, unsafe_allow_html=True)
            
            # Metrics
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-value">{len(df):,}</div>
                        <div class="metric-label">Total Records</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                date_range = f"{df['Tanggal'].min().strftime('%d/%m/%Y')}<br>to<br>{df['Tanggal'].max().strftime('%d/%m/%Y')}"
                st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-label" style="margin-bottom: 0.5rem;">Date Range</div>
                        <div style="font-size: 0.9rem; color: #2c3e50; font-weight: 600;">{date_range}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-value">{len(df.columns)}</div>
                        <div class="metric-label">Columns</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                    <div class="metric-container">
                        <div style="font-size: 1.8rem; font-weight: 700; color: #2c3e50; margin-bottom: 0.3rem;">{forecast_year}</div>
                        <div class="metric-label">Forecast Year</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Data Preview
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("👁️ Lihat Preview Data", expanded=False):
                st.dataframe(df.head(20), use_container_width=True, height=400)
            
            st.markdown("<hr>", unsafe_allow_html=True)
            
            # SECTION 2: TRAINING & FORECASTING
            st.markdown('<h2 class="section-header">🤖 Training & Forecasting</h2>', unsafe_allow_html=True)
            st.markdown('<p class="section-subheader">Sistem akan melatih 24 model XGBoost dan generate forecast 375 hari</p>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                start_button = st.button("🚀 Mulai Training & Forecasting", type="primary", use_container_width=True)
            
            if start_button:
                # Training Progress
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 🎯 Training Models")
                train_container = st.container()
                
                with train_container:
                    models, metrics, train_df, test_df = train_models(df, train_container)
                
                st.session_state['models'] = models
                st.session_state['metrics'] = metrics
                st.session_state['train_df'] = train_df
                st.session_state['test_df'] = test_df
                
                # Performance Metrics
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 📊 Model Performance")
                
                col1, col2 = st.columns(2)
                with col1:
                    avg_mae = np.mean([m['MAE'] for m in metrics.values()])
                    st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-value">{avg_mae:.2f}</div>
                            <div class="metric-label">Average MAE</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    avg_rmse = np.mean([m['RMSE'] for m in metrics.values()])
                    st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-value">{avg_rmse:.2f}</div>
                            <div class="metric-label">Average RMSE</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with st.expander("📈 Lihat Metrics Detail"):
                    metrics_df = pd.DataFrame(metrics).T
                    metrics_df = metrics_df.round(2)
                    st.dataframe(metrics_df, use_container_width=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Forecasting Progress
                st.markdown("### 🔮 Generating Forecast")
                forecast_container = st.container()
                
                with forecast_container:
                    forecast_results, forecast_dates = forecast_future(
                        df, models, n_days=375, progress_container=forecast_container
                    )
                
                st.session_state['forecast_results'] = forecast_results
                st.session_state['forecast_dates'] = forecast_dates
                
                # Create output with dynamic year
                output_df = create_output_excel(forecast_results, forecast_dates, forecast_year)
                st.session_state['output_df'] = output_df
                
                st.markdown("<hr>", unsafe_allow_html=True)
                
                # SECTION 3: VISUALIZATION
                st.markdown('<h2 class="section-header">📈 Visualisasi Hasil Forecast</h2>', unsafe_allow_html=True)
                st.markdown('<p class="section-subheader">Grafik interaktif menampilkan 90 hari historis dan 375 hari forecast</p>', unsafe_allow_html=True)
                
                fig = create_modern_visualization(df, forecast_results, forecast_dates, metrics)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Forecast Summary
                with st.expander("📋 Lihat Ringkasan Forecast"):
                    st.markdown("**Statistik Forecast (Actual Activities)**")
                    
                    actual_targets = ['Receiving_Actual', 'Delivery_Actual', 'Discharge_Actual',
                                     'Loading_Actual', 'Export_Actual', 'Import_Actual']
                    
                    summary_data = []
                    for target in actual_targets:
                        activity = target.replace('_Actual', '')
                        values = forecast_results[target]
                        summary_data.append({
                            'Activity': activity,
                            'Mean': f"{np.mean(values):.0f}",
                            'Std Dev': f"{np.std(values):.0f}",
                            'Min': f"{np.min(values):.0f}",
                            'Max': f"{np.max(values):.0f}",
                            'Total': f"{np.sum(values):.0f}"
                        })
                    
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True)
                
                st.markdown("<hr>", unsafe_allow_html=True)
                
                # SECTION 4: DOWNLOAD
                st.markdown('<h2 class="section-header">💾 Download Hasil</h2>', unsafe_allow_html=True)
                st.markdown('<p class="section-subheader">Download file Excel dengan format yang siap pakai</p>', unsafe_allow_html=True)
                
                # Convert to Excel
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    output_df.to_excel(writer, index=False, sheet_name=f'Forecast {forecast_year}')
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.download_button(
                        label=f"📥 Download Excel Forecast {forecast_year} (375 Hari)",
                        data=buffer.getvalue(),
                        file_name=f"forecast_petikemas_{forecast_year}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary",
                        use_container_width=True
                    )
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                    <div class="info-message">
                        ℹ️ <strong>File Info:</strong> Excel berisi {len(output_df)} baris data forecast 
                        (dari {forecast_dates[0].strftime('%d %B %Y')} sampai {forecast_dates[-1].strftime('%d %B %Y')})
                    </div>
                """, unsafe_allow_html=True)
        
        except Exception as e:
            st.markdown(f'<div class="warning-message">⚠️ Terjadi error: {str(e)}</div>', unsafe_allow_html=True)
            with st.expander("🔍 Lihat Detail Error"):
                st.exception(e)
    
    else:
        st.markdown("""
            <div class="info-message">
                📌 <strong>Petunjuk:</strong> Upload file Excel Anda untuk memulai proses forecasting.<br>
                💡 <strong>Format:</strong> File harus berisi minimal 1 sheet dengan nama tahun (contoh: 2023, 2024, 2025)<br>
                📊 <strong>Prediksi:</strong> Sistem otomatis memprediksi untuk tahun berikutnya
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # System Info
        st.markdown('<h2 class="section-header">ℹ️ Informasi Sistem</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div class="modern-card">
                    <h3 style="color: #2c3e50; margin-bottom: 1rem;">🎯 Spesifikasi Model</h3>
                    <ul style="line-height: 2;">
                        <li><strong>Algorithm:</strong> XGBoost Regressor</li>
                        <li><strong>Total Models:</strong> 24 models</li>
                        <li><strong>Forecast Period:</strong> 375 hari</li>
                        <li><strong>Training Split:</strong> 90% train, 10% test</li>
                        <li><strong>Features:</strong> 12 time series features</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="modern-card">
                    <h3 style="color: #2c3e50; margin-bottom: 1rem;">📊 Target Prediksi</h3>
                    <ul style="line-height: 2;">
                        <li><strong>Actual Activities:</strong> 6 aktivitas</li>
                        <li><strong>Shift 1:</strong> 6 aktivitas</li>
                        <li><strong>Shift 2:</strong> 6 aktivitas</li>
                        <li><strong>Shift 3:</strong> 6 aktivitas</li>
                        <li><strong>Total:</strong> 24 target variabel</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
            <div class="modern-card">
                <h3 style="color: #2c3e50; margin-bottom: 1rem;">🚀 Workflow Sistem</h3>
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                    <div style="text-align: center; padding: 1rem; flex: 1;">
                        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📁</div>
                        <strong>Upload Data</strong>
                        <div style="font-size: 0.85rem; color: #546E7A;">Excel 2023-2025</div>
                    </div>
                    <div style="font-size: 2rem; color: #546e7a;">→</div>
                    <div style="text-align: center; padding: 1rem; flex: 1;">
                        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">⚙️</div>
                        <strong>Preprocessing</strong>
                        <div style="font-size: 0.85rem; color: #546E7A;">Auto cleaning</div>
                    </div>
                    <div style="font-size: 2rem; color: #546e7a;">→</div>
                    <div style="text-align: center; padding: 1rem; flex: 1;">
                        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🤖</div>
                        <strong>Training</strong>
                        <div style="font-size: 0.85rem; color: #546E7A;">24 XGBoost models</div>
                    </div>
                    <div style="font-size: 2rem; color: #546e7a;">→</div>
                    <div style="text-align: center; padding: 1rem; flex: 1;">
                        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🔮</div>
                        <strong>Forecast</strong>
                        <div style="font-size: 0.85rem; color: #546E7A;">375 hari ke depan</div>
                    </div>
                    <div style="font-size: 2rem; color: #546e7a;">→</div>
                    <div style="text-align: center; padding: 1rem; flex: 1;">
                        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">💾</div>
                        <strong>Download</strong>
                        <div style="font-size: 0.85rem; color: #546E7A;">Excel hasil</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align: center; padding: 2rem; color: #546E7A; font-size: 0.9rem;">
            <strong>PELINDO Terminal Petikemas Surabaya</strong><br>
            Forecasting System with XGBoost Machine Learning<br>
            © 2026 - All Rights Reserved
        </div>
    """, unsafe_allow_html=True)

# =============================================================================
# RUN APPLICATION
# =============================================================================
if __name__ == "__main__":
    main()