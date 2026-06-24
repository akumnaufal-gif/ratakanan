import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="IHSG Predictor Pro", layout="wide")
st.title("📈 IHSG Predictor Pro")
st.markdown("**XGBoost + Prediksi 5 Hari + Data Ekonomi BI**")

# ====================== AMBIL DATA EKONOMI ======================
@st.cache_data(ttl=86400)
def get_economic_data():
    try:
        # BI Rate & Inflasi terbaru (sumber resmi + fallback)
        url = "https://api.bps.go.id/v1/indicator/1/1/0/0?lang=id"
        resp = requests.get(url, timeout=10)
        inflasi = "N/A"
        if resp.status_code == 200:
            data = resp.json()
            if 'data' in data and len(data['data']) > 0:
                inflasi = data['data'][0]['value']
        
        bi_rate = 5.75  # Update manual terbaru (Juni 2026)
        return {"BI_Rate": bi_rate, "Inflasi": float(inflasi) if inflasi != "N/A" else 2.5}
    except:
        return {"BI_Rate": 5.75, "Inflasi": 2.5}

econ = get_economic_data()

# ====================== LOAD DATA IHSG ======================
@st.cache_data(ttl=3600)
def load_data():
    df = yf.download("^JKSE", start="2018-01-01", progress=False)
    return df

df = load_data()

if df.empty:
    st.error("Gagal mengambil data IHSG")
    st.stop()

# ====================== FEATURE ENGINEERING ======================
df['Return'] = df['Close'].pct_change()
df['SMA_20'] = df['Close'].rolling(20).mean()
df['SMA_50'] = df['Close'].rolling(50).mean()
df['Volatility'] = df['Return'].rolling(20).std()

# RSI Manual
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = -delta.where(delta < 0, 0).rolling(14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# MACD Manual
ema12 = df['Close'].ewm(span=12).mean()
ema26 = df['Close'].ewm(span=26).mean()
df['MACD'] = ema12 - ema26
df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()

# Tambah data ekonomi (sama untuk semua baris)
df['BI_Rate'] = econ['BI_Rate']
df['Inflasi'] = econ['Inflasi']

df = df.dropna()

# ====================== TARGET (Multi-day) ======================
for i in range(1, 6):
    df[f'Target_{i}'] = (df['Close'].shift(-i) > df['Close']).astype(int)

# ====================== SIDEBAR ======================
st.sidebar.header("⚙️ Pengaturan")
days_ahead = st.sidebar.selectbox("Prediksi berapa hari ke depan?", [1, 3, 5], index=2)
n_estimators = st.sidebar.slider("n_estimators (XGBoost)", 100, 500, 300)

if st.button("🚀 Jalankan Prediksi", type="primary"):
    with st.spinner("Melatih model XGBoost..."):
        feature_cols = ['Return', 'SMA_20', 'SMA_50', 'Volatility', 'RSI', 'MACD', 'MACD_Signal', 'BI_Rate', 'Inflasi']
        
        X = df[feature_cols]
        y = df[f'Target_{days_ahead}']
        
        # TimeSeries Cross Validation
        tscv = TimeSeriesSplit(n_splits=5)
        accuracies = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            accuracies.append(accuracy_score(y_test, pred))
        
        avg_acc = np.mean(accuracies)
        
        # Train final model
        final_model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        final_model.fit(X, y)
        
        # Prediksi 5 hari ke depan
        latest = X.iloc[-1:].copy()
        predictions = []
        probs = []
        
        for i in range(1, 6):
            pred = final_model.predict(latest)[0]
            proba = final_model.predict_proba(latest)[0][1]
            predictions.append("Naik" if pred == 1 else "Turun")
            probs.append(proba)
            # Update untuk hari berikutnya (sederhana)
            latest['Return'] = 0.001 if pred == 1 else -0.001
        
        # ====================== TAMPILKAN HASIL ======================
        st.subheader(f"🔮 Prediksi {days_ahead} Hari ke Depan")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("IHSG Terakhir", f"{df['Close'].iloc[-1]:,.2f}")
        with col2:
            st.metric("Akurasi Model (CV)", f"{avg_acc:.1%}")
        with col3:
            st.metric("BI Rate", f"{econ['BI_Rate']}%")
        
        # Hasil Prediksi
        st.markdown("### Prediksi Harian")
        for i, (hari, arah, prob) in enumerate(zip(range(1,6), predictions, probs), 1):
            emoji = "📈" if arah == "Naik" else "📉"
            color = "green" if arah == "Naik" else "red"
            st.markdown(f"**Hari +{hari}**: {emoji} **{arah}** — Probabilitas {prob:.1%}", unsafe_allow_html=True)
        
        # Feature Importance
        importance = pd.Series(final_model.feature_importances_, index=feature_cols).sort_values(ascending=True)
        fig = go.Figure(go.Bar(x=importance.values, y=importance.index, orientation='h'))
        fig.update_layout(title="Fitur Paling Berpengaruh", height=400)
        st.plotly_chart(fig, use_container_width=True)

st.caption("⚠️ Hanya untuk edukasi & eksperimen. Bukan saran investasi.")
