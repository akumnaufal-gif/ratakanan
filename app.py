import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="IHSG Predictor", layout="wide")

st.title("📈 IHSG Predictor Simple")
st.markdown("**Tanpa komplikasi • Prediksi 5 Hari**")

@st.cache_data(ttl=3600)
def load_data():
    df = yf.download("^JKSE", period="2y", progress=False)
    if df.empty:
        df = yf.download("JKSE.JK", period="2y", progress=False)
    return df

df = load_data()

if df.empty or len(df) < 100:
    st.error("Gagal mengambil data. Coba Rerun.")
    st.stop()

# Fix kolom kalau multiindex
if isinstance(df.columns, pd.MultiIndex):
    df = df.droplevel(1, axis=1)

df['Return'] = df['Close'].pct_change()
df['SMA20'] = df['Close'].rolling(20).mean()
df['SMA50'] = df['Close'].rolling(50).mean()

df_full = df.copy()
df = df.dropna()

st.sidebar.header("Pengaturan")
hari = st.sidebar.selectbox("Lihat prediksi berapa hari?", [1, 3, 5], index=1)

if st.button("🚀 Jalankan Prediksi", type="primary"):
    with st.spinner("Sedang memproses..."):
        from sklearn.ensemble import RandomForestClassifier
        
        features = ['Return', 'SMA20', 'SMA50']
        X = df[features]
        y = (df['Close'].shift(-hari) > df['Close']).astype(int)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        latest = X.iloc[-1:].copy()
        hasil = []
        for i in range(5):
            pred = model.predict(latest)[0]
            prob = model.predict_proba(latest)[0][1]
            arah = "📈 NAIK" if pred == 1 else "📉 TURUN"
            hasil.append(f"Hari +{i+1} → {arah} ({prob:.1%})")
            latest['Return'] = 0.001 if pred == 1 else -0.001
        
        latest_price = float(df['Close'].iloc[-1])
        st.metric("IHSG Terakhir", f"{latest_price:,.2f}")
        
        st.subheader(f"Prediksi {hari} Hari ke Depan")
        for h in hasil[:hari]:
            st.success(h) if "NAIK" in h else st.error(h)
        
        st.subheader("Prediksi 5 Hari")
        for h in hasil:
            st.write(h)

st.subheader("Grafik IHSG")
chart_data = df_full[['Close', 'SMA20', 'SMA50']].tail(300).dropna()
st.line_chart(chart_data)

st.caption("⚠️ Hanya alat bantu edukasi.")
