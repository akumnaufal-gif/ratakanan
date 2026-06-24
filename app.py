import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="IHSG Predictor", layout="wide")

st.title("📈 IHSG Predictor - Versi Diperbaiki")
st.markdown("**Model lebih baik + Probabilitas realistis**")

@st.cache_data(ttl=3600)
def load_data():
    df = yf.download("^JKSE", period="3y", progress=False)
    if df.empty:
        df = yf.download("JKSE.JK", period="3y", progress=False)
    return df

df = load_data()

if df.empty or len(df) < 200:
    st.error("Gagal mengambil data")
    st.stop()

if isinstance(df.columns, pd.MultiIndex):
    df = df.droplevel(1, axis=1)

# Feature Engineering yang lebih lengkap
df['Return'] = df['Close'].pct_change()
df['SMA20'] = df['Close'].rolling(20).mean()
df['SMA50'] = df['Close'].rolling(50).mean()
df['SMA200'] = df['Close'].rolling(200).mean()
df['Volatility'] = df['Return'].rolling(20).std()

# RSI
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = -delta.where(delta < 0, 0).rolling(14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs.fillna(0)))

df_full = df.copy()
df = df.dropna()

st.sidebar.header("Pengaturan")
hari = st.sidebar.selectbox("Prediksi berapa hari ke depan?", [1, 3, 5], index=1)

if st.button("🚀 Jalankan Prediksi", type="primary"):
    with st.spinner("Melatih model..."):
        from sklearn.ensemble import RandomForestClassifier
        
        features = ['Return', 'SMA20', 'SMA50', 'SMA200', 'Volatility', 'RSI']
        X = df[features]
        y = (df['Close'].shift(-hari) > df['Close']).astype(int)
        
        model = RandomForestClassifier(
            n_estimators=300, 
            max_depth=10,
            random_state=42
        )
        model.fit(X, y)
        
        # Prediksi
        latest = X.iloc[-1:].copy()
        hasil = []
        for i in range(5):
            prob = model.predict_proba(latest)[0][1]
            pred = 1 if prob > 0.5 else 0
            arah = "📈 NAIK" if pred == 1 else "📉 TURUN"
            hasil.append(f"Hari +{i+1} → {arah} (**{prob:.1%}**)")
            # Update simulasi yang lebih halus
            latest['Return'] = latest['Return'] * 0.3 + (0.001 if pred == 1 else -0.001)
        
        latest_price = float(df['Close'].iloc[-1])
        st.metric("IHSG Terakhir", f"{latest_price:,.2f}")
        
        st.subheader(f"🔮 Prediksi {hari} Hari ke Depan")
        for h in hasil[:hari]:
            if "NAIK" in h:
                st.success(h)
            else:
                st.error(h)
        
        st.subheader("Prediksi 5 Hari Lengkap")
        for h in hasil:
            st.write(h)

# Chart
st.subheader("Grafik IHSG + Moving Average")
chart_data = df_full[['Close', 'SMA20', 'SMA50', 'SMA200']].tail(400).dropna()
st.line_chart(chart_data)

st.caption("⚠️ Model sudah diperbaiki. Prediksi masih bersifat simulasi edukasi.")
