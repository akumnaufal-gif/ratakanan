import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="IHSG Simple", layout="wide")

st.title("📈 IHSG Simple Predictor")
st.markdown("**Versi Ultra Ringan — Hanya yfinance + Streamlit**")

@st.cache_data(ttl=1800)
def load_data():
    df = yf.download("^JKSE", start="2020-01-01", progress=False)
    return df

df = load_data()

if df.empty:
    st.error("Gagal mengambil data. Coba refresh.")
    st.stop()

latest_price = df['Close'].iloc[-1]
change = df['Close'].pct_change().iloc[-1] * 100

st.metric(
    label="IHSG Hari Ini", 
    value=f"{latest_price:,.2f}", 
    delta=f"{change:.2f}%"
)

st.subheader("Trend Sederhana")
sma20 = df['Close'].rolling(20).mean().iloc[-1]
sma50 = df['Close'].rolling(50).mean().iloc[-1]

if sma20 > sma50:
    st.success("📈 **Trend Saat Ini BULLISH** (SMA20 > SMA50)")
else:
    st.error("📉 **Trend Saat Ini BEARISH** (SMA20 < SMA50)")

st.write(f"**Harga Penutupan Terakhir:** {latest_price:,.2f}")

# Chart Sederhana
st.line_chart(df['Close'].tail(300))

st.caption("⚠️ Versi test untuk mengatasi stuck build. Kalau ini jalan, baru kita tambah fitur.")
