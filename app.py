import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="IHSG Simple", layout="wide")

st.title("📈 IHSG Simple Predictor")
st.markdown("**Ultra Ringan • Sudah Diperbaiki**")

@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    try:
        df = yf.download("^JKSE", period="max", progress=False, auto_adjust=True)
        if df.empty or len(df) < 50:
            df = yf.download("JKSE.JK", period="max", progress=False)
        return df
    except Exception as e:
        st.error(f"Gagal download data: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty or len(df) < 50:
    st.error("❌ Masih gagal mengambil data IHSG.")
    st.info("Coba klik **Rerun** atau tunggu 2-3 menit lalu refresh.")
    st.stop()

# Ambil nilai scalar (bukan Series)
latest_price = float(df['Close'].iloc[-1])
change = float(df['Close'].pct_change().iloc[-1] * 100)

st.metric(
    label="IHSG Terakhir", 
    value=f"{latest_price:,.2f}", 
    delta=f"{change:.2f}%"
)

# Trend Analysis
sma20 = float(df['Close'].rolling(20).mean().iloc[-1])
sma50 = float(df['Close'].rolling(50).mean().iloc[-1])

col1, col2 = st.columns(2)
with col1:
    if sma20 > sma50:
        st.success("📈 **TREND BULLISH** (SMA20 > SMA50)")
    else:
        st.error("📉 **TREND BEARISH** (SMA20 < SMA50)")

with col2:
    st.write(f"**Tanggal Terakhir:** {df.index[-1].date()}")

# Chart
st.subheader("Grafik IHSG 400 Hari Terakhir")
st.line_chart(df['Close'].tail(400))

st.caption("✅ App sudah berjalan. Kalau ini berhasil, kita tambah fitur prediksi besok.")
