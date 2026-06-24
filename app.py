import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="IHSG Predictor Pro", layout="wide")

st.title("📈 IHSG Predictor Pro")
st.markdown("**Versi Stabil - MultiIndex Fix**")

# ================== LOAD DATA ==================
@st.cache_data(ttl=3600)
def load_data():
    try:
        df = yf.download("^JKSE", period="max", progress=False, auto_adjust=True)
        if df.empty or len(df) < 100:
            df = yf.download("JKSE.JK", period="max", progress=False, auto_adjust=True)
        return df
    except:
        return pd.DataFrame()

df = load_data()

if df.empty or len(df) < 100:
    st.error("Gagal mengambil data IHSG")
    st.stop()

# ================== FIX MULTIINDEX ==================
if isinstance(df.columns, pd.MultiIndex):
    df = df.droplevel(1, axis=1)   # Hapus ticker level

# ================== FEATURE ENGINEERING ==================
df['Return'] = df['Close'].pct_change()
df['SMA20'] = df['Close'].rolling(20).mean()
df['SMA50'] = df['Close'].rolling(50).mean()
df['Volatility'] = df['Return'].rolling(20).std()

delta = df['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = -delta.where(delta < 0, 0).rolling(14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

df_full = df.copy()
df_model = df.dropna()

# ================== SIDEBAR ==================
st.sidebar.header("Pengaturan")
hari = st.sidebar.selectbox("Prediksi berapa hari ke depan?", [1, 3, 5], index=2)

if st.button("🚀 Jalankan Prediksi", type="primary"):
    with st.spinner("Melatih model..."):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import accuracy_score
        
        features = ['Return', 'SMA20', 'SMA50', 'Volatility', 'RSI']
        X = df_model[features]
        y = (df_model['Close'].shift(-hari) > df_model['Close']).astype(int)
        
        tscv = TimeSeriesSplit(n_splits=5)
        accuracies = []
        for train_idx, test_idx in tscv.split(X):
            model = RandomForestClassifier(n_estimators=150, random_state=42)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = model.predict(X.iloc[test_idx])
            accuracies.append(accuracy_score(y.iloc[test_idx], pred))
        
        avg_acc = np.mean(accuracies)
        
        final_model = RandomForestClassifier(n_estimators=150, random_state=42)
        final_model.fit(X, y)
        
        latest = X.iloc[-1:].copy()
        hasil = []
        for i in range(5):
            pred = final_model.predict(latest)[0]
            prob = final_model.predict_proba(latest)[0][1]
            arah = "📈 **NAIK**" if pred == 1 else "📉 **TURUN**"
            hasil.append(f"Hari +{i+1}: {arah} ({prob:.1%})")
            latest['Return'] = 0.001 if pred == 1 else -0.001
        
        latest_price = float(df['Close'].iloc[-1])
        st.metric("IHSG Terakhir", f"{latest_price:,.2f}")
        st.success(f"Akurasi Model: **{avg_acc:.1%}**")
        
        st.subheader(f"🔮 Prediksi {hari} Hari ke Depan")
        for h in hasil[:hari]:
            st.write(h)
        
        st.subheader("Prediksi 5 Hari Lengkap")
        for h in hasil:
            st.write(h)

# ================== CHART ==================
st.subheader("Grafik IHSG + SMA")
chart_df = df_full[['Close', 'SMA20', 'SMA50']].tail(400).dropna()
st.line_chart(chart_df)

st.caption("⚠️ Hanya untuk edukasi. Bukan saran investasi.")
