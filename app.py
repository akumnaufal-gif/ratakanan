import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="IHSG Simple", layout="wide")

st.title("📈 IHSG Simple Predictor")
st.markdown("**Versi paling ringan • Tanpa Machine Learning • Deploy cepat**")

# Load Data
@st.cache_data(ttl=1800)  # cache 30 menit
def load_data():
    df = yf.download("^JKSE", start="2020-01-01", progress=False)
    return df

df = load_data()

if df.empty:
    st.error("Gagal mengambil data IHSG")
    st.stop()

# Simple Technical Analysis
df['SMA20'] = df['Close'].rolling(window=20).mean()
df['SMA50'] = df['Close'].rolling(window=50).mean()
df['Return'] = df['Close'].pct_change()

# Simple Signal
df['Signal'] = 0
df['Signal'] = np.where(df['SMA20'] > df['SMA50'], 1, 0)  # 1 = Bullish

latest_price = df['Close'].iloc[-1]
latest_signal = "📈 Bullish (Naik)" if df['Signal'].iloc[-1] == 1 else "📉 Bearish (Turun)"

st.metric(label="IHSG Terakhir", value=f"{latest_price:,.2f}")

col1, col2 = st.columns(2)
with col1:
    st.success(f"**Signal Saat Ini:** {latest_signal}")
with col2:
    st.write(f"**Perubahan Hari Ini:** {df['Return'].iloc[-1]:.2%}")

# Prediksi Sederhana 5 Hari (berdasarkan trend SMA)
st.subheader("🔮 Prediksi Arah 5 Hari ke Depan (Simple Trend)")
for i in range(1, 6):
    if df['Signal'].iloc[-1] == 1:
        st.write(f"Hari +{i}: 📈 **Cenderung Naik** (berdasarkan SMA20 > SMA50)")
    else:
        st.write(f"Hari +{i}: 📉 **Cenderung Turun**")

# Chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index[-400:], y=df['Close'][-400:], name="Close", line=dict(color="blue")))
fig.add_trace(go.Scatter(x=df.index[-400:], y=df['SMA20'][-400:], name="SMA 20"))
fig.add_trace(go.Scatter(x=df.index[-400:], y=df['SMA50'][-400:], name="SMA 50"))
fig.update_layout(title="IHSG + SMA20 & SMA50 (400 Hari Terakhir)", height=600)
st.plotly_chart(fig, use_container_width=True)

st.caption("⚠️ Ini hanya analisis teknikal sederhana untuk edukasi. Bukan rekomendasi investasi.")
