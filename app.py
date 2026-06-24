import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="IHSG Simple", layout="wide")

st.title("📈 IHSG Simple Predictor")
st.markdown("**Ultra Ringan • Dengan Fallback**")

@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    try:
        # Beberapa cara download
        df = yf.download("^JKSE", period="max", progress=False, auto_adjust=True)
        if df.empty or len(df) < 50:
            # Coba alternatif
            df = yf.download("JKSE.JK", period="max", progress=False)
        return df
    except:
        return pd.DataFrame()

df = load_data()

if df.empty or len(df) < 50:
    st.error("❌ Gagal mengambil data IHSG dari yfinance.")
    st.info("Coba refresh beberapa kali atau tunggu 5 menit (rate limit).")
    st.stop()

# Data berhasil diambil
latest_price = df['Close'].iloc[-1]
change = df['Close'].pct_change().iloc[-1] * 100

st.metric(
    label="IHSG Terakhir", 
    value=f"{latest_price:,.2f}", 
    delta=f"{change:.2f}%"
)

# Simple Trend
sma20 = df['Close'].rolling(20).mean().iloc[-1]
sma50 = df['Close'].rolling(50).mean().iloc[-1]

col1, col2 = st.columns(2)
with col1:
    if sma20 > sma50:
        st.success("📈 **TREND BULLISH** (SMA20 > SMA50)")
    else:
        st.error("📉 **TREND BEARISH** (SMA20 < SMA50)")

with col2:
    st.write(f"**Tanggal Terakhir:** {df.index[-1].date()}")

# Chart
st.subheader("Grafik IHSG")
st.line_chart(df['Close'].tail(400))

st.caption("⚠️ Versi test. Kalau ini berhasil muncul datanya, kita lanjut tambah fitur prediksi.")
