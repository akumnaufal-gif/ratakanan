import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="IHSG Simple", layout="wide")

st.title("📈 IHSG Simple Predictor")
st.markdown("**Versi paling ringan • Deploy cepat**")

# Load Data
@st.cache_data(ttl=3600)
def load_data():
    df = yf.download("^JKSE", start="2020-01-01", progress=False)
    return df

df = load_data()

if df.empty:
    st.error("Gagal mengambil data IHSG")
    st.stop()

# Feature Sederhana
df['Return'] = df['Close'].pct_change()
df['SMA20'] = df['Close'].rolling(20).mean()
df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().clip(lower=0).rolling(14).mean() / 
                           abs(df['Close'].diff().clip(upper=0).rolling(14).mean()))))

df = df.dropna()

# Sidebar
st.sidebar.header("Pengaturan")
hari = st.sidebar.selectbox("Prediksi berapa hari ke depan?", [1, 3, 5], index=1)

if st.button("🚀 Prediksi Sekarang", type="primary"):
    with st.spinner("Sedang memproses..."):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import accuracy_score
        
        X = df[['Return', 'SMA20', 'RSI']]
        y = (df['Close'].shift(-hari) > df['Close']).astype(int)
        
        # Cross Validation
        tscv = TimeSeriesSplit(n_splits=5)
        accuracies = []
        for train_idx, test_idx in tscv.split(X):
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = model.predict(X.iloc[test_idx])
            accuracies.append(accuracy_score(y.iloc[test_idx], pred))
        
        avg_acc = np.mean(accuracies)
        
        # Final Model
        final_model = RandomForestClassifier(n_estimators=100, random_state=42)
        final_model.fit(X, y)
        
        # Prediksi
        latest = X.iloc[-1:].copy()
        hasil = []
        for i in range(5):
            pred = final_model.predict(latest)[0]
            prob = final_model.predict_proba(latest)[0][1]
            arah = "📈 Naik" if pred == 1 else "📉 Turun"
            hasil.append(f"Hari +{i+1}: {arah} ({prob:.1%})")
            latest['Return'] = 0.001 if pred == 1 else -0.001
        
        # Tampilkan
        st.success(f"Akurasi Model: **{avg_acc:.1%}**")
        st.subheader(f"Prediksi {hari} Hari ke Depan")
        for h in hasil[:hari]:
            st.write(h)
        
        # Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[-300:], y=df['Close'][-300:], name="IHSG Close"))
        fig.update_layout(title="Pergerakan IHSG 300 Hari Terakhir", height=500)
        st.plotly_chart(fig, use_container_width=True)

st.caption("⚠️ Hanya untuk edukasi. Prediksi saham tidak pernah 100% akurat.")
