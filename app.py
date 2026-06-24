import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Prediksi Arah IHSG", page_icon="📈", layout="wide")

st.title("📈 Prediksi Arah IHSG (Versi Lebih Akurat)")
st.markdown("**Dibuat dengan Random Forest + Technical Indicators profesional**")

# ===================== SIDEBAR =====================
st.sidebar.header("⚙️ Pengaturan Model")

start_date = st.sidebar.date_input("Mulai Data Dari", value=pd.to_datetime("2018-01-01"))
n_estimators = st.sidebar.slider("Jumlah Pohon (n_estimators)", 100, 500, 200, step=50)
confidence_threshold = st.sidebar.slider("Confidence Threshold (%)", 50, 85, 60)

st.sidebar.markdown("---")
st.sidebar.info("Semakin tinggi threshold, semakin sedikit sinyal tapi lebih akurat.")

# ===================== FUNGSI =====================
@st.cache_data(ttl=3600)
def load_and_process_data(start_date):
    df = yf.download("^JKSE", start=start_date, progress=False)
    if df.empty:
        st.error("Gagal mengambil data!")
        return None
    
    # Tambah indikator pakai pandas_ta (sangat powerful)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.stoch(k=14, d=3, append=True)
    
    # Target: Besok naik atau turun?
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    df = df.dropna()
    return df

# ===================== MAIN =====================
if st.button("🚀 Jalankan Prediksi", type="primary"):
    with st.spinner("Mengunduh data & melatih model..."):
        df = load_and_process_data(start_date)
        
        if df is None:
            st.stop()
        
        # Pilih fitur terbaik
        feature_cols = [
            'RSI_14', 
            'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
            'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBP_20_2.0',
            'STOCHk_14_3_3', 'STOCHd_14_3_3'
        ]
        
        X = df[feature_cols]
        y = df['Target']
        
        # Time Series Cross Validation (lebih akurat)
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=12,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            scores.append(accuracy_score(y_test, model.predict(X_test)))
        
        avg_accuracy = np.mean(scores)
        
        # Latih model final pakai semua data
        final_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        final_model.fit(X, y)
        
        # Prediksi besok
        latest = X.iloc[-1:].values
        proba = final_model.predict_proba(latest)[0]
        prediksi = final_model.predict(latest)[0]
        
        prob_naik = proba[1] * 100
        prob_turun = proba[0] * 100
        
        # ===================== HASIL =====================
        st.subheader("📊 Hasil Prediksi Besok")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Harga IHSG Terakhir", f"{df['Close'].iloc[-1]:,.2f}")
        
        with col2:
            if prediksi == 1:
                st.success(f"📈 **DIPREDIKSI NAIK**")
            else:
                st.error(f"📉 **DIPREDIKSI TURUN**")
        
        with col3:
            st.metric("Akurasi Rata-rata Model", f"{avg_accuracy:.1%}")
        
        # Probabilitas
        st.markdown("### Probabilitas Prediksi")
        st.progress(int(prob_naik))
        st.write(f"**Probabilitas Naik:** {prob_naik:.1f}% | **Probabilitas Turun:** {prob_turun:.1f}%")
        
        if prob_naik >= confidence_threshold:
            st.success(f"✅ Sinyal Kuat (Confidence ≥ {confidence_threshold}%)")
        else:
            st.warning(f"⚠️ Confidence rendah. Saran: Tunggu sinyal lebih kuat.")
        
        # ===================== FEATURE IMPORTANCE =====================
        st.subheader("🔍 Fitur Paling Berpengaruh")
        importance_df = pd.DataFrame({
            'Fitur': feature_cols,
            'Importance': final_model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig_imp = go.Figure(go.Bar(
            x=importance_df['Importance'],
            y=importance_df['Fitur'],
            orientation='h'
        ))
        fig_imp.update_layout(title="Feature Importance", height=400)
        st.plotly_chart(fig_imp, use_container_width=True)
        
        # ===================== GRAFIK =====================
        st.subheader("📉 Grafik IHSG + Indikator")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[-300:], y=df['Close'][-300:], name='IHSG Close'))
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Klik tombol **Jalankan Prediksi** di atas untuk mulai.")

st.markdown("---")
st.caption("⚠️ Ini hanya alat edukasi. Bukan saran investasi. Pasar saham bisa berubah sewaktu-waktu.")
