import streamlit as st
import pickle
import numpy as np
import pandas as pd
import time
import os

# File riwayat prediksi
riwayat_file = 'riwayat_prediksi.csv'

# Load model
with open('model_jantung.pkl', 'rb') as file:
    model = pickle.load(file)

# Konfigurasi halaman
st.set_page_config(page_title="💓 Deteksi Penyakit Jantung", layout="centered")

# ====================== MENU NAVIGASI ======================
menu = st.sidebar.selectbox("📂 Menu Navigasi", ["🔍 Prediksi Penyakit", "📖 Tentang Aplikasi"])

# ====================== PROFIL ===========================
if menu == "📖 Tentang Aplikasi":
    st.title("📖 Tentang Aplikasi & Profil Developer")

    st.subheader("🧑‍💼 Profil Data Diri")
    st.markdown("""
    - **Nama Lengkap**: Dicky Candid Saragih  
    - **Nama Panggilan**: Syifu  
    - **Usia**: 23 Tahun  
    - **Alamat**: Kerajaan Pantai Selatan  
    - **Asal Pendidikan**: Universitas Medan Area  
    """)

    st.subheader("📝 Tentang Saya")
    st.markdown("""
    Saya adalah individu yang memiliki semangat tinggi dalam bidang data dan teknologi, dengan latar belakang pendidikan dari **Universitas Medan Area**.  
    Memiliki kombinasi kemampuan teknis dan soft skill yang kuat, saya percaya bahwa **data bukan hanya angka—tetapi cerita yang harus diungkap**.  
    Selain itu, saya dikenal sebagai pribadi yang ramah, royal, dan bertanggung jawab dalam setiap aspek kehidupan.
    """)

    st.subheader("🛠️ Kemampuan dan Keahlian")
    st.markdown("""
    ✅ Excel (Advanced)  
    ✅ MySQL (Query Optimization & Database Management)  
    ✅ Data Analyst (Data Wrangling, Visualization, Statistical Analysis)  
    ✅ Data Scientist (Model Building, A/B Testing, Predictive Analytics)  
    ✅ Machine Learning (Supervised & Unsupervised Learning)  
    ✅ Deep Learning (CNN, RNN, NLP dengan TensorFlow & PyTorch)  
    ✅ Python (for Data Science & Automation)  
    """)

    st.subheader("🌟 Nilai Tambah Personal")
    st.markdown("""
    ❤️ Mencintaimu meski tak dicintai balik  
    🤝 Royal dan bertanggung jawab  
    😄 Ramah dan mudah bergaul  
    🌟 Baik hati dan dapat diandalkan  
    💼 Siap menafkahi dan membangun masa depan bersama  
    """)

    st.stop()

# ==================== SIDEBAR RIWAYAT ====================
with st.sidebar:
    st.markdown("---")
    st.subheader("📜 Riwayat Prediksi")

    if os.path.exists(riwayat_file):
        data = pd.read_csv(riwayat_file)
        st.dataframe(data, use_container_width=True)

        nama_hapus = st.text_input("🗑️ Hapus Riwayat Pasien (Nama)")

        if st.button("Hapus Pasien"):
            if nama_hapus.strip() != "":
                new_data = data[data["Nama"].str.lower() != nama_hapus.strip().lower()]
                new_data.to_csv(riwayat_file, index=False)
                st.success(f"✅ Data pasien bernama **{nama_hapus}** berhasil dihapus.")
            else:
                st.warning("⚠️ Masukkan nama pasien terlebih dahulu.")
    else:
        st.info("Belum ada data riwayat prediksi.")

# ==================== PREDIKSI UTAMA ====================
st.title("💓 Sistem Deteksi Risiko Penyakit Jantung")
st.markdown("Silakan isi data berikut untuk mengetahui estimasi risiko penyakit jantung berdasarkan model Machine Learning.")

# Input data
nama = st.text_input("🧑 Nama Pasien")

col1, col2 = st.columns(2)
with col1:
    usia = st.number_input("🧓 Usia (tahun)", min_value=18, max_value=100, value=45)
    kolesterol = st.number_input("🥚 Kolesterol (mg/dL)", min_value=100, max_value=300, value=200)
with col2:
    tekanan_darah = st.number_input("💉 Tekanan Darah (mmHg)", min_value=90, max_value=200, value=120)
    detak_jantung = st.number_input("❤️ Detak Jantung (bpm)", min_value=60, max_value=200, value=100)

# Tampilkan ringkasan
st.markdown("### 📊 Ringkasan Data Masukan")
input_df = pd.DataFrame({
    "Nama": [nama],
    "Usia": [usia],
    "Tekanan Darah": [tekanan_darah],
    "Kolesterol": [kolesterol],
    "Detak Jantung": [detak_jantung]
})
st.table(input_df)

# Tombol prediksi
if st.button("🔍 Lakukan Prediksi"):

    if nama.strip() == "":
        st.warning("⚠️ Silakan isi nama pasien terlebih dahulu.")
    else:
        with st.spinner("🔄 Sedang menganalisis data..."):
            time.sleep(1.5)
            input_data = np.array([[usia, tekanan_darah, kolesterol, detak_jantung]])
            hasil_prediksi = model.predict(input_data)[0]
            probabilitas = model.predict_proba(input_data)[0][hasil_prediksi] * 100

        st.markdown("### 🔎 Hasil Prediksi")
        if hasil_prediksi == 1:
            st.error(f"⚠️ {nama} berisiko tinggi terkena penyakit jantung.\n\nProbabilitas: **{probabilitas:.2f}%**")
        else:
            st.success(f"✅ {nama} rendah risiko terkena penyakit jantung.\n\nProbabilitas: **{probabilitas:.2f}%**")

        # Simpan hasil ke CSV
        hasil = {
            "Nama": nama,
            "Usia": usia,
            "Tekanan Darah": tekanan_darah,
            "Kolesterol": kolesterol,
            "Detak Jantung": detak_jantung,
            "Hasil": "Risiko" if hasil_prediksi == 1 else "Sehat",
            "Probabilitas (%)": round(probabilitas, 2)
        }

        if os.path.exists(riwayat_file):
            riwayat_df = pd.read_csv(riwayat_file)
            riwayat_df = pd.concat([riwayat_df, pd.DataFrame([hasil])], ignore_index=True)
        else:
            riwayat_df = pd.DataFrame([hasil])

        riwayat_df.to_csv(riwayat_file, index=False)

# =================== FOOTER =====================
st.markdown("---")
st.caption("© 2025 Aplikasi Deteksi Jantung - Dibuat oleh Dicky Saragih")
