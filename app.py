import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="IHSG Predictor", layout="wide")

st.title("📈 IHSG Predictor - XGBoost Version")
st.markdown("**Model lebih kuat + Kurangi bias turun**")

@st.cache_data(ttl=3600)
def load_data():
    df = yf.download("^JKSE", period="5y", progress=False)
    if df.empty:
        df = yf.download("JKSE.JK", period="5y", progress=False)
    return df

df = load_data()

if df.empty or len(df) < 500:
    st.error("Gagal mengambil data")
    st.stop()

if isinstance(df.columns, pd.MultiIndex):
    df = df.droplevel(1, axis=1)

# Feature Engineering
df['Return'] = df['Close'].pct_change()
df['SMA20'] = df['Close'].rolling(20).mean()
df['SMA50'] = df['Close'].rolling(50).mean()
df['SMA200'] = df['Close'].rolling(200).mean()
df['Volatility'] = df['Return'].rolling(20).std()

delta = df['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = -delta.where(delta < 0, 0).rolling(14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs.fillna(0)))

ema12 = df['Close'].ewm(span=12).mean()
ema26 = df['Close'].ewm(span=26).mean()
df['MACD'] = ema12 - ema26
df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()

df_full = df.copy()
df = df.dropna()

st.sidebar.header("Pengaturan")
hari = st.sidebar.selectbox("Prediksi berapa hari ke depan?", [1, 3, 5], index=1)
min_conf = st.sidebar.slider("Minimum Confidence (%)", 50, 80, 60)

if st.button("🚀 Jalankan Prediksi", type="primary"):
    with st.spinner("Melatih XGBoost..."):
        from xgboost import XGBClassifier
        
        features = ['Return', 'SMA20', 'SMA50', 'SMA200', 'Volatility', 'RSI', 'MACD', 'MACD_Signal']
        X = df[features]
        y = (df['Close'].shift(-hari) > df['Close']).astype(int)
        
        model = XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        model.fit(X, y)
        
        latest = X.iloc[-1:].copy()
        hasil = []
        
        for i in range(5):
            prob_naik = model.predict_proba(latest)[0][1]
            pred = 1 if prob_naik >= 0.5 else 0
            arah = "📈 NAIK" if pred == 1 else "📉 TURUN"
            confidence = max(prob_naik, 1 - prob_naik) * 100
            
            hasil.append(f"Hari +{i+1} → {arah} (**{prob_naik:.1%}**) - Confidence: {confidence:.1f}%")
            
            latest['Return'] = latest['Return'] * 0.6 + (0.0018 if pred == 1 else -0.0012)
        
        latest_price = float(df['Close'].iloc[-1])
        st.metric("IHSG Terakhir", f"{latest_price:,.2f}")
        
        st.subheader(f"🔮 Prediksi {hari} Hari ke Depan")
        for h in hasil[:hari]:
            if "NAIK" in h and "Confidence" in h and float(h.split("Confidence: ")[1].replace("%","")) >= min_conf:
                st.success(h)
            elif "TURUN" in h and "Confidence" in h and float(h.split("Confidence: ")[1].replace("%","")) >= min_conf:
                st.error(h)
            else:
                st.warning(h)
        
        st.subheader("5 Hari Lengkap")
        for h in hasil:
            st.write(h)

# Chart
st.subheader("Grafik IHSG + Moving Average")
chart_data = df_full[['Close', 'SMA20', 'SMA50', 'SMA200']].tail(500).dropna()
st.line_chart(chart_data)

st.caption("🚀 Sudah pakai XGBoost + MACD. Semoga lebih seimbang.")
