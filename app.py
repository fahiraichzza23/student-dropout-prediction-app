import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Student Dropout Prediction",
    page_icon="🎓",
    layout="wide",
)

# Daftar fitur yang digunakan (sesuai urutan saat training)
selected_features = [
    "Curricular_units_1st_sem_approved",
    "Curricular_units_1st_sem_grade",
    "Curricular_units_2nd_sem_approved",
    "Curricular_units_2nd_sem_grade",
    "Tuition_fees_up_to_date",
    "Debtor",
    "Age_at_enrollment",
]


@st.cache_resource
def load_artifacts():
    model_obj = joblib.load("model/rdf_model.joblib")
    scaler_obj = joblib.load("model/scaler.joblib")
    encoder_obj = joblib.load("model/label_encoder.joblib")
    return model_obj, scaler_obj, encoder_obj


model, scaler, label_encoder = load_artifacts()

st.markdown(
    """
    <style>
        .hero-card {
            padding: 1.25rem 1.5rem;
            border-radius: 16px;
            border: 1px solid rgba(49, 51, 63, 0.2);
            background: linear-gradient(120deg, #f7f9fc 0%, #eef2f9 100%);
            margin-bottom: 1rem;
        }
        .info-card {
            padding: 1rem 1.2rem;
            border-radius: 14px;
            background-color: #ffffff;
            border: 1px solid rgba(49, 51, 63, 0.12);
            margin-bottom: 0.8rem;
        }
        .status-title {
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .muted {
            color: #5c6470;
            font-size: 0.96rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

header_left, header_right = st.columns([1, 6])
with header_left:
    st.image("https://github.com/dicodingacademy/assets/raw/main/logo.png", width=110)
with header_right:
    st.markdown(
        """
        <div class="hero-card">
            <div class="status-title">🎓 Student Dropout Prediction</div>
            <div class="muted">Jaya Jaya Institut - Early Warning System</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    Aplikasi ini memprediksi status siswa berdasarkan performa akademik dan kondisi finansial.
    Hasil prediksi dapat membantu institusi melakukan intervensi lebih dini untuk siswa berisiko dropout.
    """
)

left_col, right_col = st.columns([2.2, 1.2], gap="large")

with left_col:
    st.subheader("Input Data Siswa")
    with st.form("student_prediction_form", clear_on_submit=False):
        st.markdown("#### Performa Akademik Semester 1")
        col1, col2 = st.columns(2)
        with col1:
            units_approved_sem1 = st.number_input(
                "Jumlah Mata Kuliah Lulus Semester 1",
                min_value=0,
                max_value=30,
                value=5,
                help="Jumlah mata kuliah yang berhasil diluluskan di semester 1",
            )
        with col2:
            grade_sem1 = st.number_input(
                "Rata-rata Nilai Semester 1",
                min_value=0.0,
                max_value=20.0,
                value=12.0,
                step=0.1,
                help="Rata-rata nilai semester 1 (skala 0-20)",
            )

        st.markdown("#### Performa Akademik Semester 2")
        col3, col4 = st.columns(2)
        with col3:
            units_approved_sem2 = st.number_input(
                "Jumlah Mata Kuliah Lulus Semester 2",
                min_value=0,
                max_value=30,
                value=5,
                help="Jumlah mata kuliah yang berhasil diluluskan di semester 2",
            )
        with col4:
            grade_sem2 = st.number_input(
                "Rata-rata Nilai Semester 2",
                min_value=0.0,
                max_value=20.0,
                value=12.0,
                step=0.1,
                help="Rata-rata nilai semester 2 (skala 0-20)",
            )

        st.markdown("#### Faktor Finansial dan Demografi")
        col5, col6, col7 = st.columns(3)
        with col5:
            tuition_fees = st.selectbox(
                "Pembayaran Biaya Kuliah Tepat Waktu?",
                options=[1, 0],
                format_func=lambda x: "Ya" if x == 1 else "Tidak",
            )
        with col6:
            debtor = st.selectbox(
                "Status Hutang",
                options=[0, 1],
                format_func=lambda x: "Tidak Punya Hutang" if x == 0 else "Memiliki Hutang",
            )
        with col7:
            age = st.number_input(
                "Usia Saat Mendaftar",
                min_value=17,
                max_value=70,
                value=20,
            )

        predict_clicked = st.form_submit_button("Prediksi Status Siswa", type="primary")

data = pd.DataFrame(
    [
        {
            "Curricular_units_1st_sem_approved": units_approved_sem1,
            "Curricular_units_1st_sem_grade": grade_sem1,
            "Curricular_units_2nd_sem_approved": units_approved_sem2,
            "Curricular_units_2nd_sem_grade": grade_sem2,
            "Tuition_fees_up_to_date": tuition_fees,
            "Debtor": debtor,
            "Age_at_enrollment": age,
        }
    ]
)

with right_col:
    st.subheader("Ringkasan Cepat")
    st.markdown(
        """
        <div class="info-card">
            <b>Yang dinilai model</b><br>
            - Akademik semester 1 dan 2<br>
            - Ketepatan pembayaran kuliah<br>
            - Status hutang siswa<br>
            - Usia saat pendaftaran
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.metric("Total Fitur Aktif", len(selected_features))
    st.metric("Umur Input Saat Ini", f"{int(age)} tahun")

tab1, tab2 = st.tabs(["Data Input", "Hasil Prediksi"])
with tab1:
    st.dataframe(data, use_container_width=True)

with tab2:
    st.subheader("🔮 Hasil Prediksi")
    if not predict_clicked:
        st.info("Klik tombol **Prediksi Status Siswa** untuk melihat hasil.")
    else:
        data_ordered = data[selected_features]
        data_scaled = scaler.transform(data_ordered)
        prediction = model.predict(data_scaled)
        prediction_label = label_encoder.inverse_transform(prediction)[0]

        try:
            prediction_proba = model.predict_proba(data_scaled)[0]
            proba_dict = dict(zip(label_encoder.classes_, prediction_proba))
        except Exception:
            proba_dict = None

        if prediction_label == "Dropout":
            st.error(f"### Prediksi: **{prediction_label}**")
            st.warning(
                """
                **Rekomendasi Tindakan:**
                - Hubungi siswa untuk konseling akademik
                - Tawarkan program bimbingan belajar
                - Evaluasi kondisi finansial siswa
                - Pertimbangkan pemberian beasiswa atau keringanan biaya
                """
            )
        elif prediction_label == "Graduate":
            st.success(f"### Prediksi: **{prediction_label}**")
            st.info(
                """
                **Insight:**
                - Siswa diprediksi akan menyelesaikan studi dengan baik
                - Tetap monitor perkembangan akademik secara berkala
                """
            )
        else:
            st.info(f"### Prediksi: **{prediction_label}**")
            st.info(
                """
                **Insight:**
                - Siswa masih dalam proses studi
                - Perlu pemantauan lebih lanjut untuk menentukan status akhir
                """
            )

        if proba_dict:
            st.markdown("#### Probabilitas Prediksi")
            for status_name, probability in sorted(
                proba_dict.items(), key=lambda item: item[1], reverse=True
            ):
                st.write(f"**{status_name}** - {probability * 100:.2f}%")
                st.progress(float(probability))

            prob_df = pd.DataFrame(
                {
                    "Status": list(proba_dict.keys()),
                    "Probabilitas": [f"{p * 100:.2f}%" for p in proba_dict.values()],
                }
            )
            st.dataframe(prob_df, use_container_width=True)

        with st.expander("Lihat Data Setelah Preprocessing"):
            scaled_df = pd.DataFrame(data_scaled, columns=selected_features)
            st.dataframe(scaled_df, use_container_width=True)