import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="IHSG Simple", layout="wide")

st.title("📈 IHSG Simple Predictor")
st.markdown("**Ultra Ringan • Fix TypeError Terbaru**")

@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    try:
        df = yf.download("^JKSE", period="max", progress=False, auto_adjust=True)
        if df.empty or len(df) < 30:
            df = yf.download("JKSE.JK", period="max", progress=False, auto_adjust=True)
        return df
    except Exception as e:
        st.error(f"Download error: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty or len(df) < 30:
    st.error("❌ Masih gagal mengambil data. Coba **Rerun** beberapa kali.")
    st.stop()

# ================== FIX TYPE ERROR ==================
latest_price = df['Close'].iloc[-1]
if isinstance(latest_price, pd.Series):
    latest_price = float(latest_price.iloc[0])
else:
    latest_price = float(latest_price)

change = df['Close'].pct_change().iloc[-1]
if isinstance(change, pd.Series):
    change = float(change.iloc[0])
else:
    change = float(change) * 100

st.metric(
    label="IHSG Terakhir", 
    value=f"{latest_price:,.2f}", 
    delta=f"{change:.2f}%"
)

# Trend
sma20 = df['Close'].rolling(20).mean().iloc[-1]
sma50 = df['Close'].rolling(50).mean().iloc[-1]

if isinstance(sma20, pd.Series): sma20 = float(sma20.iloc[0])
else: sma20 = float(sma20)

if isinstance(sma50, pd.Series): sma50 = float(sma50.iloc[0])
else: sma50 = float(sma50)

col1, col2 = st.columns(2)
with col1:
    if sma20 > sma50:
        st.success("📈 **BULLISH** (SMA20 > SMA50)")
    else:
        st.error("📉 **BEARISH** (SMA20 < SMA50)")

with col2:
    st.write(f"Tanggal: {df.index[-1].date()}")

st.subheader("Grafik IHSG")
st.line_chart(df['Close'].tail(400))

st.caption("✅ Sudah di-fix. Kalau ini berhasil muncul, kita tambah fitur prediksi.")
