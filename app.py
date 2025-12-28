"""Streamlit App - Energy Forecasting"""
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt

# Config page
st.set_page_config(
    page_title="Energy Forecasting",
    page_icon="⚡",
    layout="wide"
)

# Titre
st.title("⚡ Energy Forecasting - Prédiction LSTM")
st.markdown("**Prédisez la consommation électrique des 24 prochaines heures**")

# ============================================================================
# CHARGEMENT MODÈLE
# ============================================================================

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('models/lstm_final.keras')
    return model

@st.cache_resource
def load_scalers():
    with open('data/processed/scaler_X.pkl', 'rb') as f:
        scaler_X = pickle.load(f)
    with open('data/processed/scaler_y.pkl', 'rb') as f:
        scaler_y = pickle.load(f)
    return scaler_X, scaler_y

try:
    model = load_model()
    scaler_X, scaler_y = load_scalers()
    st.sidebar.success("✅ Modèle chargé")
except Exception as e:
    st.error(f"❌ Erreur chargement : {e}")
    st.stop()

# ============================================================================
# SIDEBAR - PARAMÈTRES
# ============================================================================

st.sidebar.header("📅 Paramètres")

# Date picker
selected_date = st.sidebar.date_input(
    "Date de départ",
    value=datetime.now() + timedelta(days=1),
    min_value=datetime.now(),
    max_value=datetime.now() + timedelta(days=365)
)

selected_time = st.sidebar.time_input(
    "Heure de départ",
    value=datetime.now().time()
)

# Combiner date + heure
start_datetime = datetime.combine(selected_date, selected_time)

# Afficher
st.sidebar.markdown(f"**Prédiction à partir de :**")
st.sidebar.info(f"📅 {start_datetime.strftime('%Y-%m-%d %H:%M')}")

# ============================================================================
# GÉNÉRATION FEATURES
# ============================================================================

def generate_features(start_dt, lookback=168):
    """Générer features pour prédiction"""
    
    # Créer timestamps passés (7 jours avant)
    timestamps = [start_dt - timedelta(hours=i) for i in range(lookback, 0, -1)]
    
    # Créer DataFrame
    df = pd.DataFrame({'Datetime': timestamps})
    
    # Features temporelles
    df['hour'] = df['Datetime'].dt.hour
    df['day_of_week'] = df['Datetime'].dt.dayofweek
    df['month'] = df['Datetime'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Cycliques
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Simuler consommation historique (pattern réaliste)
    # Base saisonnière
    seasonal = 35000 + 10000 * np.sin(2 * np.pi * (df['month'] - 1) / 12)
    # Pattern journalier
    daily = 5000 * np.sin(2 * np.pi * (df['hour'] - 6) / 24)
    # Weekend
    weekend_effect = -3000 * df['is_weekend']
    # Bruit
    noise = np.random.normal(0, 1000, len(df))
    
    df['consumption'] = seasonal + daily + weekend_effect + noise
    
    # Lags (simulés à partir de la consommation générée)
    df['lag_24h'] = df['consumption'].shift(24).fillna(df['consumption'].mean())
    df['lag_168h'] = df['consumption'].shift(168).fillna(df['consumption'].mean())
    
    # Rolling stats
    df['rolling_mean_24h'] = df['consumption'].rolling(24, min_periods=1).mean()
    df['rolling_std_24h'] = df['consumption'].rolling(24, min_periods=1).std().fillna(0)
    
    # Features dans le bon ordre
    feature_cols = [
        'consumption', 'hour', 'day_of_week', 'month', 'is_weekend',
        'hour_sin', 'hour_cos', 'lag_24h', 'lag_168h',
        'rolling_mean_24h', 'rolling_std_24h'
    ]
    
    return df[feature_cols].values

# ============================================================================
# PRÉDICTION
# ============================================================================

if st.sidebar.button("🔮 PRÉDIRE", type="primary"):
    
    with st.spinner("Génération des prédictions..."):
        
        # Générer features
        X = generate_features(start_datetime)
        
        # Normaliser
        X_scaled = scaler_X.transform(X)
        
        # Reshape pour LSTM
        X_input = X_scaled.reshape(1, 168, 11)
        
        # Prédiction
        y_pred_scaled = model.predict(X_input, verbose=0)
        
        # Dénormaliser
        y_pred = scaler_y.inverse_transform(y_pred_scaled)[0]
        
        # Créer timestamps futurs
        future_times = [start_datetime + timedelta(hours=i) for i in range(24)]
        
        # DataFrame résultats
        results = pd.DataFrame({
            'Heure': future_times,
            'Consommation (MW)': y_pred
        })
        
        # ================================================================
        # AFFICHAGE RÉSULTATS
        # ================================================================
        
        st.success("✅ Prédiction terminée !")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Moyenne", f"{y_pred.mean():.0f} MW")
        
        with col2:
            st.metric("📈 Maximum", f"{y_pred.max():.0f} MW")
        
        with col3:
            st.metric("📉 Minimum", f"{y_pred.min():.0f} MW")
        
        with col4:
            st.metric("🔄 Amplitude", f"{y_pred.max() - y_pred.min():.0f} MW")
        
        # Graphique
        st.markdown("---")
        st.subheader("📈 Prédictions sur 24 heures")
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        hours = [t.strftime('%H:%M') for t in future_times]
        ax.plot(hours, y_pred, marker='o', linewidth=2, markersize=6, 
                color='#FF6B6B', label='Prédiction')
        ax.fill_between(range(24), y_pred, alpha=0.3, color='#FF6B6B')
        
        ax.set_xlabel('Heure', fontsize=12)
        ax.set_ylabel('Consommation (MW)', fontsize=12)
        ax.set_title(f"Prédiction du {start_datetime.strftime('%Y-%m-%d')}", 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Rotation labels
        plt.xticks(range(0, 24, 3), [hours[i] for i in range(0, 24, 3)], rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Tableau détaillé
        st.markdown("---")
        st.subheader("📋 Détails des prédictions")
        
        # Formater tableau
        results['Heure'] = results['Heure'].dt.strftime('%Y-%m-%d %H:%M')
        results['Consommation (MW)'] = results['Consommation (MW)'].round(2)
        
        st.dataframe(results, use_container_width=True, height=400)
        
        # Download CSV
        csv = results.to_csv(index=False)
        st.download_button(
            label="📥 Télécharger CSV",
            data=csv,
            file_name=f"forecast_{start_datetime.strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

# ============================================================================
# INFORMATIONS
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Informations")
st.sidebar.markdown("""
**Modèle :** LSTM (2 couches)  
**MAPE :** 4.68%  
**Dataset :** PJM East (2002-2018)  
**Lookback :** 168h (7 jours)  
**Horizon :** 24h
""")

st.sidebar.markdown("---")
