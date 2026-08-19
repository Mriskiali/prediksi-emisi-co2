# Prediksi Emisi CO₂ Mobil dengan Machine Learning

Aplikasi web interaktif berbasis **Streamlit** untuk memprediksi tingkat emisi CO₂ kendaraan berdasarkan spesifikasi mesin dan fisik mobil. Proyek ini menggunakan model **Regresi** (*Machine Learning*) yang dilatih dengan jutaan data populasi kendaraan nyata dari benua Eropa dan Asia.

## Fitur Utama
- **Prediksi Real-Time:** Memprediksi Emisi CO₂ secara instan menggunakan input manual (*Fuel Consumption*, *Mass*, *Engine Power*).
- **Evaluasi Model:** Menampilkan performa *Machine Learning* secara transparan (Metrik MAE, MSE, R² Score).
- **Analisis Fitur (Feature Importance):** Visualisasi *bar chart* yang menjelaskan seberapa besar pengaruh setiap spesifikasi kendaraan terhadap polusi CO₂.
- **Visualisasi Interaktif:** Grafik *Scatter Plot* (Prediksi vs Aktual) dan *Box Plot* interaktif yang bisa di-*hover* menggunakan Plotly.
- **Insight Rekor Emisi Global:** Papan peringkat mobil paling ramah lingkungan dan mobil paling berpolusi berdasarkan analisis populasi Big Data.

## Teknologi yang Digunakan
- **Python 3**
- **Streamlit** (Front-end Dashboard)
- **Scikit-Learn** (Machine Learning Model)
- **Pandas & NumPy** (Data Processing)
- **Plotly Express** (Interactive Data Visualization)

## Struktur Repositori
- app.py - Script utama antarmuka Streamlit.
- genuine_clean_data.csv - 1.000 sampel data bersih yang diekstrak langsung dari jutaan populasi *Big Data* (digunakan untuk merender grafik di web).
- model_regresi.pkl - *Pre-trained Machine Learning Model* (dieksport menggunakan Joblib).
- 
equirements.txt - Daftar pustaka (library) yang dibutuhkan oleh server.

## Cara Menjalankan Secara Lokal (Localhost)
Jika ingin menjalankan aplikasi ini di komputermu sendiri:

1. *Clone* repositori ini:
   `bash
   git clone https://github.com/USERNAME_KAMU/NAMA_REPO_KAMU.git
   `
2. Instal semua dependensi yang dibutuhkan:
   `bash
   pip install -r requirements.txt
   `
3. Jalankan aplikasi Streamlit:
   `bash
   streamlit run app.py
   `

---
*Dibuat untuk keperluan Analisis Data dan Eksplorasi Machine Learning.*
