# 🏙️ Big Data Analytics — Arsitektur dan Tata Kota

## 📚 Tentang Mata Kuliah
Mata kuliah **Big Data Analytics (BDA)** membahas konsep, teknologi, dan implementasi terkait pemrosesan data berukuran besar, baik dalam bentuk batch maupun streaming. Pembelajaran berfokus pada teori dan praktik, mulai dari setup lingkungan, pemrosesan data, hingga penerapan machine learning untuk pengambilan keputusan berbasis data.

## 🔧 Tools & Teknologi yang Dipelajari
- **Apache Kafka** — Message broker untuk streaming data real-time.
- **Apache Hadoop** — Framework distributed processing berbasis MapReduce.
- **RabbitMQ** — Message queue untuk komunikasi antar layanan.
- **ETL (Extract, Transform, Load)** — Konsep pemrosesan data dari sumber ke sistem analitik.
- **Stream Processing** — Pemrosesan data secara real-time.
- **Batch Processing** — Pemrosesan data dalam jumlah besar secara berkala.
- **Linux Environment** — Setup CLI, server, package manager, dan file system.
- **Machine Learning** — Penerapan model klasifikasi dan prediksi.
- **Data Visualization** — Analisis dan visualisasi hasil data untuk interpretasi.

## 🧠 Konsep Big Data
- **3V**: Volume, Velocity, Variety
- **5V**: + Veracity, Value
- **7V**: + Variability, Visualization

## 🧪 Studi Kasus: Klasifikasi Gaya Arsitektur
Studi kasus ini merupakan penerapan BDA dalam bidang **tata kota & gaya hidup**. Dengan model deep learning, gambar bangunan dapat dikenali dan diklasifikasikan ke dalam beberapa gaya arsitektur, seperti:
- Klasik
- Modern
- Kontemporer
- Tradisional
- Kolonial

### Tujuan:
Membantu pemerintah, arsitek, atau masyarakat untuk:
- Menganalisis tren arsitektur berdasarkan lokasi.
- Mendorong konservasi gaya arsitektur khas daerah.
- Mendukung tata kota berbasis data visual.

## 🧰 Arsitektur Solusi

- Dataset disimpan di Google Drive
- Model berbasis **ResNet50** (transfer learning)
- Training dilakukan di Google Colab
- Gambar di-preprocessing & di-augmentasi
- Model disimpan dalam format `.h5`

## 🧪 Contoh Implementasi

Berikut ini cuplikan kode dari model training:

```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(5, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
```

### 📊 Hasil Model

Training Accuracy: ~38%  
Validation Accuracy: ~31%

Masih bisa ditingkatkan dengan fine-tuning, optimasi augmentasi, dan penambahan data.

#### Visualisasi Akurasi dan Loss

| Epoch | Training Acc | Val Acc |
|-------|--------------|---------|
| 1     | 13.5%        | 27.1%   |
| 10    | 33.2%        | 24.7%   |
| 20    | 36.9%        | 29.2%   |

## 🧪 Contoh Prediksi

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

image_path = '.../Arsitektur/Klasik/Image_2.jpg'
img = load_img(image_path, target_size=(150, 150))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
class_names = ['Klasik', 'Modern', 'Kontemporer', 'Tradisional', 'Kolonial']
print("Prediksi:", class_names[np.argmax(prediction)])
```

## 📁 Struktur Folder

```
bda/
├── dataset/
│   ├── Klasik/
│   ├── Modern/
│   └── ...
├── notebooks/
├── model/               # <- Dikecualikan dari Git
├── uploaded_model.h5    # <- Dihapus karena terlalu besar
├── train.py
├── predict.py
├── README.md
└── .gitignore
```

## 🚫 File yang Dikecualikan (.gitignore)

```gitignore
model/
uploaded_model.h5
*.h5
```

## 📌 Catatan Tambahan

Model `.h5` tidak diupload karena ukuran file melebihi limit GitHub. Disarankan menggunakan Git LFS atau menyimpan model di cloud (Google Drive, HuggingFace, dll).

## 📖 Referensi

- https://kafka.apache.org
- https://hadoop.apache.org
- https://www.rabbitmq.com
- https://towardsdatascience.com/introduction-to-etl-pipeline-7c8f5a3212d2
- ResNet Paper
- Deep Learning with Keras & TensorFlow
- Materi perkuliahan BDA 2024/2025
