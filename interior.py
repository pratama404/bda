import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Daftar label kelas arsitektur
LABELS = ["Kontemporer", "Modern", "Tradisional", "Klasik"]

# Fungsi untuk memuat model CNN
def load_cnn_model(path):
    """Memuat model CNN dari file .h5"""
    return load_model(path)

# Fungsi untuk memproses gambar
def prepare_image(img, target_size):
    """Mengubah gambar agar sesuai dengan input model"""
    img = img.resize(target_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalisasi gambar
    return img

# Streamlit UI
st.title("Klasifikasi Gaya Arsitektur")
st.write("Unggah model CNN dan gambar untuk diklasifikasikan.")

# File uploader untuk model
uploaded_model = st.file_uploader("Pilih model .h5", type=["h5"])
uploaded_image = st.file_uploader("Pilih gambar untuk klasifikasi", type=["jpg", "jpeg", "png"])

if uploaded_model is not None:
    try:
        # Simpan model sementara
        with open('uploaded_model.h5', 'wb') as f:
            f.write(uploaded_model.getbuffer())
        
        st.write("Memuat model...")
        model = load_cnn_model('uploaded_model.h5')
        st.success("Model berhasil dimuat!")
        
        # Menentukan ukuran input model
        input_shape = model.input_shape
        target_size = (input_shape[1], input_shape[2])
        
        if uploaded_image is not None:
            img = Image.open(uploaded_image)
            st.image(img, caption="Gambar yang diunggah.", use_column_width=True)
            
            img_array = prepare_image(img, target_size)
            
            # Prediksi
            st.write("Memproses gambar...")
            predictions = model.predict(img_array)
            class_idx = np.argmax(predictions)
            class_prob = predictions[0][class_idx]
            class_name = LABELS[class_idx]
            
            st.success(f"Prediksi: {class_name} ({class_prob:.4f})")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
