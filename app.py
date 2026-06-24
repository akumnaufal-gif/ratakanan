import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="IHSG Predictor", layout="wide")

st.title("📈 IHSG Predictor - Upgrade Aman")
st.markdown("**Model diperbaiki + Threshold**")

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

# Feature Engineering lebih baik
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

df_full = df.copy()
df = df.dropna()

st.sidebar.header("Pengaturan")
hari = st.sidebar.selectbox("Prediksi berapa hari ke depan?", [1, 3, 5], index=1)
min_confidence = st.sidebar.slider("Minimum Confidence (%)", 50, 70, 55)

if st.button("🚀 Jalankan Prediksi", type="primary"):
    with st.spinner("Melatih model..."):
        from sklearn.ensemble import RandomForestClassifier
        
        features = ['Return', 'SMA20', 'SMA50', 'SMA200', 'Volatility', 'RSI']
        X = df[features]
        y = (df['Close'].shift(-hari) > df['Close']).astype(int)
        
        model = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
        model.fit(X, y)
        
        latest = X.iloc[-1:].copy()
        hasil = []
        
        for i in range(5):
            prob_naik = model.predict_proba(latest)[0][1]
            pred = 1 if prob_naik >= 0.5 else 0
            arah = "📈 NAIK" if pred == 1 else "📉 TURUN"
            confidence = max(prob_naik, 1 - prob_naik)
            
            hasil.append({
                "hari": i+1,
                "arah": arah,
                "prob": prob_naik,
                "confidence": confidence
            })
            
            latest['Return'] = latest['Return'] * 0.4 + (0.0008 if pred == 1 else -0.0008)
        
        latest_price = float(df['Close'].iloc[-1])
        st.metric("IHSG Terakhir", f"{latest_price:,.2f}")
        
        st.subheader(f"🔮 Prediksi {hari} Hari ke Depan")
        for h in hasil[:hari]:
            if h["confidence"] * 100 >= min_confidence:
                if h["arah"] == "📈 NAIK":
                    st.success(f"Hari +{h['hari']} → {h['arah']} (**{h['prob']:.1%}**)")
                else:
                    st.error(f"Hari +{h['hari']} → {h['arah']} (**{h['prob']:.1%}**)")
            else:
                st.warning(f"Hari +{h['hari']} → **Kurang Yakin** ({h['prob']:.1%})")

        st.caption(f"✅ Hanya menampilkan prediksi dengan confidence ≥ {min_confidence}%")

# Chart
st.subheader("Grafik IHSG + Moving Average")
chart_data = df_full[['Close', 'SMA20', 'SMA50', 'SMA200']].tail(400).dropna()
st.line_chart(chart_data)

st.caption("⚠️ Upgrade aman. Kalau masih sering turun, kita tambah fitur lagi secara bertahap.")
