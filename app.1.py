import joblib
import os
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler


# URL gambar dengan format raw
image_url = "https://raw.githubusercontent.com/Julpian/IconicIT/main/download.jpg"

# Convert image from GitHub raw URL to base64
response = requests.get(image_url)
response.raise_for_status()  # Error jika URL gagal diakses

# Membuka gambar dan mengonversinya ke base64
img = Image.open(BytesIO(response.content))
buffered = BytesIO()
img.save(buffered, format="JPEG")
img_str = base64.b64encode(buffered.getvalue()).decode()

# URL logo dengan format raw
logo_url = "https://raw.githubusercontent.com/Julpian/IconicIT/main/logo.png"

# Convert logo from GitHub raw URL to base64
response = requests.get(logo_url)
response.raise_for_status()  # Error jika URL gagal diakses

# Membuka logo dan mengonversinya ke base64
logo = Image.open(BytesIO(response.content))
logo_buffered = BytesIO()
logo.save(logo_buffered, format="PNG")
logo_str = base64.b64encode(logo_buffered.getvalue()).decode()

print("Logo Base64:", logo_str)

# Set up the page
st.set_page_config(
    page_title="Siswa Profil Prediction",
    layout="wide",
    page_icon="📚",
    initial_sidebar_state="expanded"
)

# Define custom CSS
custom_css = f"""
<style>
.main {{
    background-image: url('data:image/jpeg;base64,{img_str}');
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-color: rgba(0, 0, 0, 0.7);  /* Dark overlay to make text more readable */
    color: #000;  /* Text color */
    font-family: Arial, sans-serif;  /* Font style */
    padding: 20px;
}}

h1, h2, h3, h4, h5, h6 {{
    color: #fff;  /* Header text color */
}}

p {{
    color: #fff;  /* Paragraph text color */
}}

.header-logo {{
    position: absolute;
    top: 10px;
    right: 10px;
}}

.team-name {{
    font-size: 24px;
    color: #fff;
    font-weight: bold;
    margin-bottom: 20px;
}}

.menu {{
    display: flex;
    justify-content: space-around;
    margin-top: 20px;
}}
</style>
"""

# Apply the custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Add logo and team name
st.markdown(f'<img src="data:image/png;base64,{logo_str}" class="header-logo" width="100">', unsafe_allow_html=True)
st.markdown('<div class="team-name">BriliumTeam</div>', unsafe_allow_html=True)

# Sidebar with navigation
st.sidebar.title("Navigasi")
page = st.sidebar.selectbox(
    "Pilih Halaman:",
    ("Home", "Prediksi", "About Us")
)

# Render the appropriate page based on menu selection
if page == "Home":
    st.write(
        """
        🌠 **Algoritma**
       
          Alasan kenapa harus menggunakan KNN?

        - K-Nearest Neighbors (KNN) adalah metode yang sering digunakan dalam klasifikasi dan prediksi karena sejumlah alasan yang diungkapkan oleh para ahli. Berikut adalah beberapa pandangan dari ahli dalam bidang pembelajaran mesin dan statistik mengenai mengapa KNN efektif untuk tugas klasifikasi:

        - Simplicity and Intuitiveness:

          Tom Mitchell: Dalam bukunya "Machine Learning", Tom Mitchell menyebutkan bahwa KNN adalah algoritma yang sederhana dan intuitif. Konsep dasarnya adalah mengklasifikasikan data berdasarkan kedekatannya dengan titik data lain. Ini membuatnya mudah dipahami dan diterapkan.
          Non-parametric Nature:

        - Christopher Bishop: Dalam buku "Pattern Recognition and Machine Learning", Christopher Bishop menjelaskan bahwa KNN adalah metode non-parametrik, yang berarti tidak membuat asumsi eksplisit tentang bentuk distribusi data. Ini memungkinkan KNN untuk bekerja dengan baik dalam kasus di mana distribusi data tidak diketahui.
          Versatility and Flexibility:

        - Ian Goodfellow, Yoshua Bengio, and Aaron Courville: Dalam "Deep Learning", ketiga penulis menyoroti bahwa KNN dapat digunakan untuk berbagai jenis data dan tidak memerlukan pelatihan model yang kompleks. Ini membuatnya sangat fleksibel dan mudah digunakan dalam berbagai aplikasi klasifikasi.
          Adaptability to Local Patterns:

        - Richard O. Duda, Peter E. Hart, and David G. Stork: Dalam buku "Pattern Classification", penulis menjelaskan bahwa KNN dapat menangkap pola lokal dalam data dengan baik karena klasifikasi dilakukan berdasarkan tetangga terdekat. Ini sangat berguna dalam situasi di mana pola lokal lebih relevan daripada pola global.
          Ease of Implementation:

        - Sebastian Raschka: Dalam "Python Machine Learning", Sebastian Raschka menekankan bahwa KNN mudah diimplementasikan dan dapat diterapkan dengan sedikit konfigurasi. Ini membuatnya menjadi pilihan yang baik untuk aplikasi praktis di mana waktu dan sumber daya terbatas.
          Secara keseluruhan, kelebihan KNN dalam klasifikasi terletak pada kesederhanaannya, kemampuannya untuk beradaptasi dengan pola lokal, dan fleksibilitas dalam menangani berbagai jenis data.
        
        🎯 **Tujuan Aplikasi:**
        - Mengklasifikasikan profil teknis siswa berdasarkan data yang diberikan.
        - Menentukan profil terbaik untuk siswa dalam Ilmu Data, Web (Back-End), dan Web (Front-End).
        
        📊 **Masukkan Data:**
        - Jam belajar dan jumlah kursus.
        - Skor rata-rata dari kursus yang telah diambil.
        
        🕵️‍♂️ **Tips:**
        - Cobalah berbagai kombinasi untuk melihat perubahan profil siswa!
        """
    )

elif page == "Prediksi":
    st.write(
        """
        🧪 **Gunakan Formulir di Bawah Ini:**
        - Masukkan data siswa pada formulir.
        - Klik tombol *Prediksi Profil* untuk melihat hasilnya.
        """
    )
    # URL raw untuk scaler dan model
    scaler_url = "https://raw.githubusercontent.com/Julpian/IconicIT/main/scaler.pkl"
    model_url = "https://raw.githubusercontent.com/Julpian/IconicIT/main/knn_model.pkl"

    # Fungsi untuk mengunduh file dari URL dan memuatnya
    def load_file_from_url(url):
        response = requests.get(url)
        response.raise_for_status()  # Pastikan file berhasil diunduh
        return BytesIO(response.content)

    # Memuat scaler
    try:
        scaler_data = load_file_from_url(scaler_url)
        scaler = joblib.load(scaler_data)
        st.write('Scaler loaded successfully!')
    except Exception as e:
        st.error(f"An error occurred while loading the scaler: {e}")

    # Memuat model
    try:
        model_data = load_file_from_url(model_url)
        profil_model = joblib.load(model_data)
        st.write('Model loaded successfully!')
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")

    # Input from user
    st.header('Input Data')
    col1, col2, col3 = st.columns(3)

    with col1:
        hours_datascience = st.number_input("Jumlah Jam Belajar Data Science", min_value=0, max_value=101, step=1)
        hours_backend = st.number_input("Jumlah Jam Belajar Web (Back-End)", min_value=0, max_value=105, step=1)
        hours_frontend = st.number_input("Jumlah Jam Belajar Web (Front-End)", min_value=0, max_value=94, step=1)

    with col2:
        num_courses_beginner_datascience = st.number_input("Jumlah Kursus Pemula Data Science", min_value=0, max_value=9, step=1)
        num_courses_beginner_backend = st.number_input("Jumlah Kursus Pemula Web (Back-End)", min_value=0, max_value=9, step=1)
        num_courses_beginner_frontend = st.number_input("Jumlah Kursus Pemula Web (Front-End)", min_value=0, max_value=12, step=1)

    with col3:
        num_courses_advanced_datascience = st.number_input("Jumlah Kursus Lanjutan Data Science", min_value=0, max_value=9, step=1)
        num_courses_advanced_backend = st.number_input("Jumlah Kursus Lanjutan Web (Back-End)", min_value=0, max_value=10, step=1)
        num_courses_advanced_frontend = st.number_input("Jumlah Kursus Lanjutan Web (Front-End)", min_value=0, max_value=9, step=1)

    st.header('Additional Features')
    col4, col5 = st.columns(2)

    with col4:
        avg_score_datascience = st.number_input("Skor Rata-rata Data Science", min_value=29.0, max_value=100.0, step=0.1)
        avg_score_backend = st.number_input("Skor Rata-rata Web (Back-End)", min_value=30.0, max_value=100.0, step=0.1)

    with col5:
        avg_score_frontend = st.number_input("Skor Rata-rata Web (Front-End)", min_value=30.0, max_value=100.0, step=0.1)

    # Button to make prediction
    if st.button("Prediksi Profil"):
        if 'profil_model' in locals():  # Check if model is loaded
            # Collect user data into array
            user_input_array = np.array([[
                hours_datascience,
                hours_backend,
                hours_frontend,
                num_courses_beginner_datascience,
                num_courses_beginner_backend,
                num_courses_beginner_frontend,
                num_courses_advanced_datascience,
                num_courses_advanced_backend,
                num_courses_advanced_frontend,
                avg_score_datascience,
                avg_score_backend,
                avg_score_frontend
            ]])

            # Normalize user input
            user_input_scaled = scaler.transform(user_input_array)

            # Make prediction
            prediction_index = profil_model.predict(user_input_scaled)[0]

            # Define labels and their descriptions
            labels = {
                0: "advanced_backend",
                1: "advanced_data_science",
                2: "advanced_front_end",
                3: "beginner_backend",
                4: "beginner_data_science",
                5: "beginner_front_end"
            }

            # Get the predicted label
            predicted_label = labels.get(prediction_index, "Label tidak tersedia")

            # Display result with description
            st.write(f'🎉 **Profil yang Diprediksi:** {predicted_label} 🎉')
        else:
            st.error("Model not loaded. Please check the model file path and try again.")

elif page == "About Us":
    st.write(
        """
        🤖 Tentang Kami:

        - Kami adalah tim pengembang yang terdiri dari Lutfi Julpian dan Mohammad Faikar Natsir.
        - Aplikasi ini dirancang untuk membantu menentukan profil teknis siswa dengan menggunakan model prediksi berbasis data.
        - Kami percaya bahwa pendidikan adalah fondasi utama dalam membentuk masa depan. Dengan mengidentifikasi kekuatan dan minat teknis siswa, kami bertujuan untuk memberikan wawasan yang dapat membantu dalam merencanakan dan mengembangkan keterampilan mereka lebih lanjut.
        
        🌟 Kontak Kami:
        - Email: misteralfikri@gmail.com
        - Email: lutfijulpian@gmail.com
        
        💬 Masukkan Masukan:

        - Kami senang mendengar dari Anda! Jangan ragu untuk memberikan masukan atau saran. Keterlibatan Anda sangat berarti bagi kami dalam upaya untuk terus meningkatkan aplikasi ini.
        """
    )
