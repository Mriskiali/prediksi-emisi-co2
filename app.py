import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import io

DATA_PATH = "genuine_clean_data.csv"
MODEL_PATH = "model_regresi.pkl"

def configure_page():
    st.set_page_config(
    page_title="Prediksi Emisi CO2 Mobil",
    layout="wide",
    initial_sidebar_state="expanded",
    )

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def main():
    configure_page()
    st.title("Prediksi Emisi CO2 Mobil")
    
    df = load_data(DATA_PATH)
    try:
        model = load_model(MODEL_PATH)
    except:
        st.error(f"Gagal memuat model dari {MODEL_PATH}")
        return

    # --- SIDEBAR INTERACTIVE PREDICTION ---
    st.sidebar.header("Form Prediksi Mobil Kustom")
    st.sidebar.markdown("Masukkan spesifikasi mobil untuk memprediksi emisi CO2-nya secara real-time.")
    
    val_fc = st.sidebar.number_input("Fuel Consumption (L/100km)", min_value=0.0, max_value=50.0, value=7.5, step=0.1)
    val_mass = st.sidebar.number_input("Mass in running order (kg)", min_value=500.0, max_value=5000.0, value=1500.0, step=50.0)
    val_power = st.sidebar.number_input("Engine Power (kW)", min_value=10.0, max_value=1000.0, value=110.0, step=5.0)
    
    if st.sidebar.button("Prediksi CO2 Sekarang", type="primary"):
        custom_X = pd.DataFrame([[val_fc, val_mass, val_power]], columns=['fuel consumption', 'mass in running order (kg)', 'engine power in kw'])
        custom_pred_val = float(np.squeeze(model.predict(custom_X)))
        st.sidebar.success(f"**Estimasi Emisi CO2:** {custom_pred_val:.2f} g/km")

    # --- MAIN CONTENT ---

    st.subheader("Data Awal Kendaraan (Sample dari Dataset)")
    st.dataframe(df, width='stretch')

    st.subheader("Data yang Digunakan untuk Prediksi")
    st.info("**Variabel Independen (X):** Fuel Consumption (L/100km), Mass (kg), Engine Power (kW)")
    st.error("**Variabel Dependen (Y):** CO2 Emissions (g/km)")
    
    pred_cols = ['Fuel Consumption', 'Mass (kg)', 'Engine Power (kW)', 'CO2 Emissions (g/km)']
    st.dataframe(df[pred_cols], width='stretch')
    
    st.subheader("Hasil Prediksi")
    st.success(f"Model berhasil dimuat dari file {MODEL_PATH}.")
    
    X = df[['Fuel Consumption', 'Mass (kg)', 'Engine Power (kW)']]
    y = df['CO2 Emissions (g/km)']
    
    X_pred = X.copy()
    X_pred.columns = ['fuel consumption', 'mass in running order (kg)', 'engine power in kw']
    preds = model.predict(X_pred)
    
    df['Prediksi CO2 (g/km)'] = preds

    # Metrics and Layman explanation
    mae = mean_absolute_error(y, preds)
    mse = mean_squared_error(y, preds)
    r2 = r2_score(y, preds)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("MSE", f"{mse:.2f}")
    col3.metric("R² Score", f"{r2:.3f}")
    
    st.info(f"**Kesimpulan Model:** R² Score sebesar {r2:.2f} berarti model ini mampu mengenali dan memprediksi {r2*100:.1f}% pola emisi CO2 dengan sangat baik berdasarkan fitur yang ada. Rata-rata tebakan model hanya meleset sekitar {mae:.2f} g/km dari emisi aslinya, sehingga model ini bisa diandalkan.")

    # Feature Importance
    if hasattr(model, 'coef_'):
        st.subheader("Pengaruh Setiap Fitur terhadap CO2 (Feature Importance)")
        coefs = model.coef_
        if hasattr(coefs, '__len__') and len(coefs.shape) > 1:
            coefs = coefs[0]
            
        coef_df = pd.DataFrame({
            'Fitur': ['Fuel Consumption', 'Mass', 'Engine Power'],
            'Koefisien (Besar Pengaruh)': coefs
        })
        
        fig_coef = px.bar(coef_df, x='Fitur', y='Koefisien (Besar Pengaruh)', color='Fitur', title="Seberapa besar setiap spesifikasi memengaruhi Emisi CO2?")
        st.plotly_chart(fig_coef, width='stretch')

    st.subheader("Grafik Prediksi vs Aktual berdasarkan Tipe/Merek")
    
    fig = px.scatter(
        df, 
        x='CO2 Emissions (g/km)', 
        y='Prediksi CO2 (g/km)', 
        color='Make',
        title='Prediksi vs Aktual untuk CO2 Emissions (g/km)',
        labels={'CO2 Emissions (g/km)': 'Aktual', 'Prediksi CO2 (g/km)': 'Prediksi'},
        hover_data=['Type', 'Fuel Consumption', 'Mass (kg)', 'Engine Power (kW)']
    )
    min_val = min(y.min(), preds.min())
    max_val = max(y.max(), preds.max())
    fig.add_shape(
        type="line", line=dict(dash='dash', color='red', width=2),
        x0=min_val, y0=min_val, x1=max_val, y1=max_val
    )
    st.plotly_chart(fig, width='stretch')


    st.subheader("Insight Tambahan: Emisi CO2 Asia vs Eropa")
    asia_brands = ['TOYOTA', 'HONDA', 'NISSAN', 'MAZDA', 'MITSUBISHI', 'SUZUKI', 'SUBARU', 'LEXUS', 'HYUNDAI', 'KIA']
    eropa_brands = ['AUDI', 'BMW', 'MERCEDES-BENZ', 'VOLKSWAGEN', 'PORSCHE', 'VOLVO', 'PEUGEOT', 'RENAULT', 'FIAT', 'FERRARI']
    
    df['Make_Upper'] = df['Make'].astype(str).str.upper()
    df.loc[df['Make_Upper'].isin(asia_brands), 'Region'] = 'Asia'
    df.loc[df['Make_Upper'].isin(eropa_brands), 'Region'] = 'Eropa'
    
    region_df = df.dropna(subset=['Region'])
    if not region_df.empty:
        asia_df = region_df[region_df['Region'] == 'Asia']
        eropa_df = region_df[region_df['Region'] == 'Eropa']
        
        avg_asia = asia_df['CO2 Emissions (g/km)'].mean() if not asia_df.empty else 0
        avg_eropa = eropa_df['CO2 Emissions (g/km)'].mean() if not eropa_df.empty else 0
        
        col_a, col_e = st.columns(2)
        col_a.metric("Rata-rata Emisi CO2 Mobil Asia", f"{avg_asia:.2f} g/km")
        col_e.metric("Rata-rata Emisi CO2 Mobil Eropa", f"{avg_eropa:.2f} g/km")
        
        if avg_asia < avg_eropa and avg_asia > 0:
            st.success("Berdasarkan data, mobil pabrikan **Asia** rata-rata menghasilkan emisi CO2 lebih rendah dibandingkan pabrikan Eropa.")
        elif avg_eropa < avg_asia and avg_eropa > 0:
            st.info("Berdasarkan data, mobil pabrikan **Eropa** rata-rata menghasilkan emisi CO2 lebhh rendah dibandingkan pabrikan Asia.")
            
        fig_region = px.box(region_df, x='Region', y='CO2 Emissions (g/km)', color='Region', title="Distribusi Emisi CO2: Asia vs Eropa")
        st.plotly_chart(fig_region, width='stretch')
    else:
        st.warning("Tidak cukup data merek Asia/Eropa yang dikenali untuk membuat perbandingan.")

    st.subheader("Insight Global: Rekor Emisi CO₂ Tertinggi & Terendah")
    st.markdown("Berdasarkan analisis populasi keseluruhan data, berikut adalah mobil dengan rekor emisi terbaik dan terburuk:")
    
    col_champ1, col_champ2 = st.columns(2)
    with col_champ1:
        st.success("**JUARA CO₂ TERENDAH**\n\n**MERCEDES-BENZ F2B**\n- **Fuel Type:** Petrol\n- **Konsumsi Bensin:** 16.4 L/100km\n- **Emisi CO₂:** 0.0 g/km")
    with col_champ2:
        st.error("**JUARA CO₂ TERTINGGI**\n\n**ASTON MARTIN AM001**\n- **Fuel Type:** Petrol\n- **Konsumsi Bensin:** 24.1 L/100km\n- **Emisi CO₂:** 543.0 g/km")

    st.markdown("#### Rekor CO₂ Tertinggi & Terendah berdasarkan Fuel Type")
    fuel_data = pd.DataFrame({
        'Kategori Rekor': ['Terendah', 'Terendah', 'Terendah', 'Terendah', 'Terendah', 'Tertinggi', 'Tertinggi', 'Tertinggi', 'Tertinggi', 'Tertinggi'],
        'Make': ['MERCEDES-BENZ', 'VOLKSWAGEN VW', 'MERCEDES-BENZ', 'SKODA', 'FORD', 'ASTON MARTIN', 'MERCEDES-BENZ', 'LAMBORGHINI', 'INEOS', 'PEUGEOT'],
        'Type': ['F2B', 'CD', 'R2CGLC', 'NW', 'J2K', 'AM001', '906BA50', '744', 'NaN', 'Y'],
        'Fuel Type': ['Petrol', 'Petrol/Electric', 'Other', 'NG', 'E85', 'Petrol', 'Other', 'Petrol/Electric', 'E85', 'NG'],
        'Fuel Cons. (L/100km)': [16.4, 0.3, 0.4, 6.3, 4.5, 24.1, 15.6, 7.3, 14.4, 9.2],
        'CO2 (g/km)': [0.0, 6.0, 10.0, 98.0, 103.0, 543.0, 410.0, 404.0, 333.0, 241.0]
    })
    st.dataframe(fuel_data, width='stretch')

if __name__ == "__main__":
    main()
