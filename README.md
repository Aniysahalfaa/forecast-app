# Petikemas Activity Forecasting System

Modern web application for automatic time series forecasting of container terminal activities using XGBoost.

## Features

- **Automatic Data Processing**: Upload Excel file with 3-level headers, system handles preprocessing automatically
- **24 XGBoost Models**: Trains separate models for 6 Actual activities + 18 Shift activities
- **1-Year Forecast**: Generates recursive forecasts for 365 days ahead
- **Clean Interface**: Modern UI without clutter, focused on functionality
- **Excel Export**: Download formatted results matching original data structure

## System Architecture

```
Input: Excel (2023-2025 data)
   ↓
Preprocessing (3-level headers → standardized format)
   ↓
Training (24 XGBoost models, 90-10 split)
   ↓
Forecasting (Recursive, 365 days)
   ↓
Output: Excel file + Visualizations
```

## File Structure

```
petikemas-forecasting/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── .streamlit/            # Streamlit configuration (optional)
    └── config.toml
```

## Local Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Steps

1. **Clone or download project files**
```bash
mkdir petikemas-forecasting
cd petikemas-forecasting
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run application**
```bash
streamlit run app.py
```

4. **Access in browser**
```
Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

## Deployment on Streamlit Cloud (Free)

### Step 1: Prepare Repository

1. **Create GitHub account** (if you don't have one)
   - Go to https://github.com
   - Sign up for free

2. **Create new repository**
   - Click "New repository"
   - Name: `petikemas-forecasting`
   - Set as Public
   - Click "Create repository"

3. **Upload files**
   - Upload `app.py`
   - Upload `requirements.txt`
   - Upload `README.md`
   - Commit changes

### Step 2: Deploy on Streamlit Cloud

1. **Go to Streamlit Cloud**
   - Visit https://streamlit.io/cloud
   - Click "Sign up" or "Sign in"
   - Connect with GitHub account

2. **Create new app**
   - Click "New app"
   - Select your repository: `petikemas-forecasting`
   - Main file path: `app.py`
   - Click "Deploy"

3. **Wait for deployment** (2-3 minutes)
   - Streamlit will install dependencies
   - Build and launch your app
   - You'll get a public URL like: `https://your-app-name.streamlit.app`

4. **Share the link**
   - Copy the URL
   - Anyone can access it via browser
   - No login required for users

### Step 3: Update App (Optional)

When you need to update:
1. Push changes to GitHub repository
2. Streamlit Cloud auto-deploys within minutes

## Usage Instructions

### For End Users

1. **Open the application**
   - Navigate to the provided URL in any browser
   - No installation needed

2. **Upload data file**
   - Click "Choose Excel file"
   - Select your Excel file (must contain sheets: 2023, 2024, 2025)
   - Wait for preprocessing to complete

3. **Start processing**
   - Click "Start Training & Forecasting" button
   - Wait 2-5 minutes for completion
   - Progress bars show real-time status

4. **Download results**
   - View forecast visualization
   - Click "Download Forecast Excel"
   - File saved to your Downloads folder

### Expected Processing Time

- Data upload: < 10 seconds
- Preprocessing: 10-20 seconds
- Model training: 1-2 minutes (24 models)
- Forecasting: 1-2 minutes (365 days)
- **Total: ~3-5 minutes**

## Input Data Format

Excel file must contain:
- **3 sheets**: 2023, 2024, 2025
- **3-level headers**: 
  - Level 1: Section (Shift III, Shift I, Shift II, Actual)
  - Level 2: Activity (Receiving, Delivery, Discharge, Loading, etc.)
  - Level 3: Subcolumns
- **Base columns**: Week, Tanggal, Hari, ket

## Output Format

Excel file contains:
- **Week**: Week number
- **Tanggal**: Date (DD/MM/YYYY format)
- **Hari**: Day name
- **Shift III columns**: 6 activities
- **Shift I columns**: 6 activities
- **Shift II columns**: 6 activities
- **Actual 2026 columns**: 6 activities
- **ket**: Notes column (empty)

## Technical Specifications

### Models
- **Algorithm**: XGBoost Regressor
- **Count**: 24 models (6 Actual + 18 Shift)
- **Hyperparameters**:
  - n_estimators: 100
  - learning_rate: 0.1
  - max_depth: 5
  - random_state: 42

### Features (12 total)
- **Calendar**: day_of_week, month, week_of_year, is_weekend
- **Lag**: lag_1, lag_7, lag_14, lag_30
- **Rolling**: rolling_mean_7, rolling_std_7, rolling_mean_30, rolling_std_30

### Performance Metrics
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)

### Forecasting Method
- **Type**: Recursive forecasting
- **Horizon**: 365 days
- **Constraint**: Non-negative predictions

## Troubleshooting

### Common Issues

**1. Upload Error**
- Ensure Excel file has sheets: 2023, 2024, 2025
- Check file is not corrupted
- Try saving as new Excel file

**2. Processing Takes Too Long**
- Normal processing: 3-5 minutes
- If > 10 minutes, refresh page and retry

**3. Download Not Working**
- Check browser popup blocker
- Try different browser (Chrome recommended)

**4. Deployment Failed**
- Check `requirements.txt` is in repository
- Verify Python version compatibility
- Check Streamlit Cloud logs for errors

## Support

For issues or questions:
1. Check this README first
2. Review error messages in application
3. Contact system administrator

## Version History

- **v1.0** (2025-01-15): Initial release
  - 24 XGBoost models
  - 365-day forecasting
  - Clean modern UI
  - Excel export functionality

## License

This application is for internal use only.

## Credits

Developed for petikemas activity forecasting
Using: Python, Streamlit, XGBoost, Pandas, Scikit-learn