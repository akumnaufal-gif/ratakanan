import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="IHSG Predictor", layout="wide")

st.title("📈 IHSG Predictor - Versi Ringan")
st.markdown("**Random Forest • Prediksi 5 Hari • Cepat Loading**")

# ====================== LOAD DATA ======================
@st.cache_data(ttl=3600)
def load_data():
    df = yf.download("^JKSE", start="2019-01-01", progress=False)
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

delta = df['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = -delta.where(delta < 0, 0).rolling(14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

df = df.dropna()

# ====================== SIDEBAR ======================
st.sidebar.header("Pengaturan")
days = st.sidebar.selectbox("Prediksi berapa hari ke depan?", [1, 3, 5], index=2)

if st.button("🚀 Jalankan Prediksi", type="primary"):
    with st.spinner("Melatih model..."):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import accuracy_score
        
        feature_cols = ['Return', 'SMA_20', 'SMA_50', 'Volatility', 'RSI']
        X = df[feature_cols]
        y = (df['Close'].shift(-days) > df['Close']).astype(int)
        
        # Cross Validation
        tscv = TimeSeriesSplit(n_splits=5)
        acc = []
        for train_idx, test_idx in tscv.split(X):
            model = RandomForestClassifier(n_estimators=150, random_state=42)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = model.predict(X.iloc[test_idx])
            acc.append(accuracy_score(y.iloc[test_idx], pred))
        
        avg_acc = np.mean(acc)
        
        # Final Model
        final_model = RandomForestClassifier(n_estimators=150, random_state=42)
        final_model.fit(X, y)
        
        # Prediksi
        latest = X.iloc[-1:].copy()
        preds = []
        for i in range(5):
            p = final_model.predict(latest)[0]
            prob = final_model.predict_proba(latest)[0][1]
            preds.append((i+1, "📈 Naik" if p == 1 else "📉 Turun", prob))
            latest['Return'] = 0.001 if p == 1 else -0.001  # simulasi sederhana
        
        # Tampilan
        st.success(f"Akurasi Model: **{avg_acc:.1%}**")
        
        st.subheader(f"Prediksi {days} Hari ke Depan")
        for hari, arah, prob in preds[:days]:
            st.markdown(f"**Hari +{hari}** → {arah} **({prob:.1%})**")
        
        # Grafik
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[-200:], y=df['Close'][-200:], name="IHSG"))
        st.plotly_chart(fig, use_container_width=True)

st.caption("⚠️ Versi ringan untuk mempercepat loading. Belum pakai data ekonomi.")
