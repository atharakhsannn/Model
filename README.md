# 🔬 Density Map Estimation for Industrial Part Counting

> **Capstone Project** — Estimasi jumlah objek (baut, mur, komponen industri) menggunakan Density Map Regression berbasis Deep Learning.

Proyek ini membangun pipeline end-to-end untuk menghitung jumlah part/komponen industri dalam sebuah gambar tanpa perlu mendeteksi setiap objek secara individual. Pendekatan yang digunakan adalah **Density Map Estimation**, di mana model memprediksi sebuah continuous heatmap dan jumlah objek diperoleh dari integrasi (penjumlahan piksel) density map tersebut.

---

## 📋 Daftar Isi

- [Arsitektur Proyek](#-arsitektur-proyek)
- [Struktur Folder](#-struktur-folder)
- [Prasyarat & Instalasi](#-prasyarat--instalasi)
- [Pipeline Penggunaan](#-pipeline-penggunaan)
  - [Step 1: Persiapan Gambar](#step-1-persiapan-gambar)
  - [Step 2: Anotasi Titik Koordinat](#step-2-anotasi-titik-koordinat-point_labelerpy)
  - [Step 3: Generate Ground Truth](#step-3-generate-ground-truth-density-map-generate_ground_truthpy)
  - [Step 4: Arsitektur Model](#step-4-arsitektur-model-model_dmepy)
  - [Step 5: Utilitas & Visualisasi](#step-5-utilitas--visualisasi-density_utilspy)
- [Detail Teknis](#-detail-teknis)
  - [Density Map Generation](#density-map-generation)
  - [Model Architecture](#model-architecture)
  - [Augmentasi Skala dan Resize (Training)](#augmentasi-skala-dan-resize-training)
  - [Format Anotasi](#format-anotasi-json)
- [Cara Menjalankan](#-cara-menjalankan)
- [Teknologi yang Digunakan](#-teknologi-yang-digunakan)
- [Roadmap & Pengembangan](#-roadmap--pengembangan)
- [Catatan Penting](#-catatan-penting)

---

## 🏗 Arsitektur Proyek

```
┌─────────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Gambar Asli    │────▶│ Point Labeler│────▶│ Anotasi (.json)  │
│  (dataset/      │     │ (GUI Tool)   │     │ (dataset/        │
│   images/)      │     └──────────────┘     │  annotations/)   │
└─────────────────┘                          └────────┬─────────┘
                                                      │
                                                      ▼
                                             ┌──────────────────┐
                                             │ Generate Ground  │
                                             │ Truth Script     │
                                             └────────┬─────────┘
                                                      │
                                    ┌─────────────────┴────────────────┐
                                    ▼                                  ▼
                           ┌────────────────┐              ┌───────────────────┐
                           │ Density Map    │              │ Visualisasi       │
                           │ (.npy)         │              │ (_vis.jpg)        │
                           │ (ground_truth/)│              │ (ground_truth/)   │
                           └───────┬────────┘              └───────────────────┘
                                   │
                                   ▼
                          ┌─────────────────┐
                          │ DensityMap      │
                          │ Regressor       │
                          │ (MobileNetV2 +  │
                          │  Dilated Conv)  │
                          └────────┬────────┘
                                   │
                                   ▼
                          ┌─────────────────┐
                          │ Predicted       │
                          │ Density Map     │
                          │ (sum = count)   │
                          └─────────────────┘
```

---

## 📁 Struktur Folder

```
CAPSTONE (Density Mapping)/
│
├── 📄 README.md                    # Dokumentasi proyek ini
├── 📄 requirements.txt             # Dependensi Python
│
├── 🐍 point_labeler.py             # GUI tool anotasi titik koordinat
├── 🐍 generate_ground_truth.py     # Script generate density map ground truth
├── 🐍 density_utils.py             # Shared module utilitas density map (KDTree & Visualisasi)
├── 🐍 model_dme.py                 # Arsitektur model deep learning
├── 🐍 dataset_loader.py            # PyTorch Dataset & augmentasi (Albumentations)
├── 🐍 train.py                     # Script utama proses training dengan MAE
├── 🐍 predict.py                   # Script inference / prediksi pada gambar baru
├── 🐍 fix_images.py                # Script utilitas perbaikan resolusi gambar dataset
│
└── 📂 dataset/
    ├── 📂 images/                   # Foto asli baut/part (.png, .jpg, .bmp)
    │   └── sample_bolts.png
    │
    ├── 📂 annotations/              # File koordinat titik (.json)
    │   └── sample_bolts.json
    │
    └── 📂 ground_truth/             # Hasil generate density map
        ├── sample_bolts.npy         # Density map (numpy array, float32)
        └── sample_bolts_vis.jpg     # Visualisasi overlay heatmap
```

---

## ⚙ Prasyarat & Instalasi

### System Requirements

- **Python** 3.8+
- **OS**: Windows / Linux / macOS
- **GPU** (opsional, untuk training model): CUDA-compatible NVIDIA GPU

### Instalasi Dependencies

```bash
# Install dependencies via requirements.txt
pip install -r requirements.txt
```

| Library | Versi Min. | Kegunaan |
|---------|-----------|----------|
| `numpy` | 1.21+ | Operasi matriks & array |
| `opencv-python` | 4.5+ | Image processing & GUI labeler |
| `scipy` | 1.7+ | KDTree untuk density map |
| `matplotlib` | 3.4+ | Visualisasi heatmap overlay |
| `torch` | 2.0+ | Deep learning framework |
| `torchvision` | 0.15+ | Pretrained MobileNetV2 |
| `albumentations` | 1.3+ | Pipeline augmentasi gambar dan density map secara sinkron |

> **Catatan Windows:** Jika `python` tidak dikenali, gunakan `py` sebagai gantinya (Python Launcher for Windows).

---

## 🚀 Pipeline Penggunaan

### Step 1: Persiapan Gambar

Letakkan semua foto baut/part yang ingin dihitung ke dalam folder:

```
dataset/images/
```

Format yang didukung: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`

---

### Step 2: Anotasi Titik Koordinat (`point_labeler.py`)

Tool GUI interaktif untuk menandai lokasi setiap objek pada gambar.

```bash
python point_labeler.py
```

#### Kontrol Keyboard

| Tombol | Fungsi |
|--------|--------|
| **Klik Kiri** | Tandai titik pada objek (muncul dot hijau) |
| **`z`** | Undo — hapus titik terakhir |
| **`s`** | Simpan anotasi ke `dataset/annotations/<nama>.json` |
| **`d`** | Lanjut ke gambar berikutnya |
| **`q`** | Keluar dari program |

#### Fitur

- ✅ Auto-resize gambar besar (max 900px) agar muat di layar
- ✅ Koordinat tetap disimpan pada resolusi asli gambar
- ✅ Load anotasi sebelumnya secara otomatis jika sudah pernah di-save
- ✅ Progres ditampilkan di terminal (`[1/N] filename.jpg`)

---

### Step 3: Generate Ground Truth Density Map (`generate_ground_truth.py`)

Mengonversi anotasi titik menjadi density map (ground truth) untuk training model. Menggunakan modul terpusat `density_utils.py`.

```bash
python generate_ground_truth.py
```

#### Proses Internal

1. Membaca semua `.json` dari `dataset/annotations/`
2. Membuka gambar asli untuk mendapatkan dimensi (H × W)
3. Mengaplikasikan algoritma **KDTree Adaptive Sigma** di mana lebar Gaussian blob akan mengecil di area titik yang sangat padat dan membesar di area yang longgar (membantu model membedakan objek yang bertumpuk).
4. Menyimpan hasil sebagai `.npy` (presisi float32) dan `_vis.jpg` (visualisasi)

#### Output

| File | Format | Keterangan |
|------|--------|------------|
| `<nama>.npy` | NumPy float32 | Density map dengan presisi penuh |
| `<nama>_vis.jpg` | JPEG | Overlay heatmap JET di atas gambar asli |

#### Terminal Output

```
============================================================
  GENERATE GROUND TRUTH - Density Map dari Anotasi
============================================================
  Annotations : dataset\annotations
  Images      : dataset\images
  Output      : dataset\ground_truth
  Algorithm   : KDTree Adaptive Sigma
  Total file  : 1
============================================================

  Found 1 existing .npy files. Delete and regenerate? [y/N]: y
  Deleted 1 .npy and 1 _vis.jpg files.

[1/1] Memproses: sample_bolts.json
  Gambar  : sample_bolts.png
  Jumlah titik : 7
  Dimensi : 1024 x 1024
  Density map shape : (1024, 1024)
  Density map max   : 0.00070736
  Density map sum   : 7.0000 (idealnya ~ 7)
  Tersimpan (.npy)  : dataset\ground_truth\sample_bolts.npy
  Tersimpan (vis)   : dataset\ground_truth\sample_bolts_vis.jpg
```

> **Catatan:** Nilai `Density map sum` harus mendekati jumlah titik anotasi. Ini membuktikan bahwa Gaussian filter mempertahankan integritas jumlah objek.

---

### Step 4: Arsitektur Model (`model_dme.py`)

Model deep learning untuk memprediksi density map dari gambar input.

```bash
# Quick test arsitektur model
python model_dme.py
```

#### Arsitektur: `DensityMapRegressor`

```
Input Image (3, 672, 512)
         │
         ▼
┌─────────────────────────┐
│  MobileNetV2 (Pretrained)│   Feature Extractor
│  Output: (1280, H/32, W/32) │   (Classifier dihapus)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Dilated Conv2D Layer 1   │   1280 → 512 ch, dilation=2
│ + BatchNorm + ReLU       │
├─────────────────────────┤
│ Dilated Conv2D Layer 2   │   512 → 128 ch, dilation=2
│ + BatchNorm + ReLU       │
├─────────────────────────┤
│ Dilated Conv2D Layer 3   │   128 → 1 ch, dilation=2
│ + ReLU (non-negatif)     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Bilinear Upsample (32x)  │   Mengembalikan spasial input
└────────────┬────────────┘
             │
             ▼
    Density Map (1, 672, 512)
    sum(pixels) ≈ object count
```

#### Spesifikasi Model

| Property | Value |
|----------|-------|
| Backbone | MobileNetV2 (pretrained on ImageNet) |
| Dilated Conv Layers | 3 layers, dilation=2 |
| Upsample | Bilinear, scale_factor=32 |
| Input Shape | `(batch, 3, 672, 512)` |
| Output Shape | `(batch, 1, 672, 512)` |
| Total Parameters | 8,715,009 |
| Aktivasi Akhir | ReLU (output ≥ 0) |

---

### Step 5: Utilitas & Visualisasi (`density_utils.py`)

Modul utilitas terpusat (single source of truth) untuk mencegah divergensi kode antar pipeline:

| Fungsi | Deskripsi |
|--------|-----------|
| `generate_density_map(image_shape, points)` | Konversi koordinat titik → density map menggunakan KDTree Adaptive Sigma |
| `create_visualization(image, density_map)` | Overlay density map di atas gambar dengan parameter input_is_rgb dinamis |

---

## 🔍 Detail Teknis

### Density Map Generation

Proses mengonversi anotasi titik menjadi density map menggunakan algoritma:

```
Anotasi Titik          Matriks Delta           Density Map (Gaussian)
                       
  • (x1, y1)    →     0 0 0 0 0 0      →     0.0 0.1 0.3 0.1 0.0
  • (x2, y2)           0 0 1 0 0 0             0.1 0.5 1.0 0.5 0.1
  • (x3, y3)           0 0 0 0 0 0             0.0 0.1 0.3 0.1 0.0
                       0 0 0 0 1 0             0.0 0.0 0.1 0.3 0.5
                       0 0 0 0 0 0             0.0 0.0 0.0 0.1 0.3
```

**KDTree Adaptive Sigma**: Model dinamis ini menyebarkan setiap titik menjadi distribusi kontinu di mana lebarnya (`sigma`) dikalkulasi berdasarkan jarak ke 3 tetangga terdekat:
- Area Padat: Objek saling tumpang tindih -> Jarak antar titik sangat dekat -> `sigma` kecil -> Blob menjadi lebih kecil dan tajam (mencegah blob bergabung menjadi satu gumpalan yang membingungkan model).
- Area Jarang: Objek terpisah -> `sigma` membesar mengikuti jarak terdekat -> Coverage lebih baik.

**Parameter Standar:** `BETA = 0.3`, `MIN_SIGMA = 4.0`, `MAX_SIGMA = 15.0`, `DEFAULT_SIGMA = 8.0`.
**Properti kunci:** `sum(density_map) ≈ jumlah_objek`

### Model Architecture

**MobileNetV2** dipilih sebagai backbone karena:
- ✅ Ringan dan efisien (cocok untuk deployment)
- ✅ Pretrained pada ImageNet (transfer learning)
- ✅ Depthwise separable convolution (mengurangi parameter)

**Dilated Convolution** digunakan karena:
- ✅ Receptive field lebih luas tanpa menambah parameter
- ✅ Mempertahankan resolusi spasial
- ✅ Menangkap konteks multi-scale untuk mengenali objek yang saling tumpang tindih

**Bilinear Upsample** digunakan karena:
- ✅ Mengembalikan resolusi output ke ukuran input
- ✅ Tidak menambah parameter (parameter-free)
- ✅ Menghasilkan transisi piksel yang halus

### Augmentasi Skala dan Resize (Training)

Untuk mempertahankan *scale-invariance* saat training, gambar diaugmentasi menggunakan rasio skala *(0.75 - 1.25)*. 
Kunci penting dalam augmentasi ini adalah densitas per piksel wajib dikoreksi `/ scale_factor**2` agar nilai `sum(density_map)` tetap sesuai dengan jumlah anotasi asli. 
Setelah di-scale, seluruh data training dan validasi akan di-resize statis secara seragam ke `672x512` agar batching stabil. Model hanya akan melihat data berukuran `672x512` selama proses training.

### Format Anotasi (JSON)

```json
{
  "image": "nama_file.png",
  "image_width": 1024,
  "image_height": 1024,
  "count": 7,
  "points": [
    [x1, y1],
    [x2, y2]
  ]
}
```

---

## 💻 Cara Menjalankan

```bash
# 1. Clone / masuk ke folder proyek
cd "CAPSTONE (Density Mapping)"

# 2. Install dependencies
pip install -r requirements.txt
pip install -r requirements.txt

# 3. Letakkan gambar ke dataset/images/

# 4. Jalankan Point Labeler untuk anotasi
python point_labeler.py

# 5. Generate ground truth density map (WAJIB JALANKAN SEBELUM TRAINING)
python generate_ground_truth.py

# 6. Test arsitektur model
python model_dme.py

# 7. Training Model (Menggunakan Stratified Split Train/Val)
python train.py

# 8. Prediksi / Inference menggunakan Model terlatih (akan memuat checkpoints/best_dme_model.pth)
python predict.py "path/to/gambar/test.jpg"
```

---

## 🛠 Teknologi yang Digunakan

| Teknologi | Kegunaan |
|-----------|----------|
| Python | Bahasa pemrograman utama |
| PyTorch | Deep learning framework |
| TorchVision | Pretrained models (MobileNetV2) |
| OpenCV | Image processing & GUI annotation tool |
| NumPy | Operasi matriks & penyimpanan density map |
| SciPy | KDTree perhitungan adaptive sigma |
| Matplotlib | Visualisasi heatmap overlay |
| Albumentations | Pipeline augmentasi & transformasi gambar |

---

## 🗺 Roadmap & Pengembangan

- [x] **Struktur dataset** — Folder `images/`, `annotations/`, `ground_truth/`
- [x] **Point Labeler GUI** — Tool anotasi titik interaktif dengan undo
- [x] **Ground Truth Generator** — Konversi anotasi → density map dengan KDTree Adaptive Sigma
- [x] **Model Architecture** — MobileNetV2 + Dilated Conv + Upsample
- [x] **Dataset Loader** — PyTorch `Dataset` dan `DataLoader` dengan **Stratified Split**
- [x] **Training Script** — Training dengan MAE, Adam, dan checkpoint otomatis pada `best_val_mae`
- [x] **Inference Script** — Modul prediksi *scale-invariant* `predict.py` dengan koreksi area
- [ ] **Model Export** — Export model ke ONNX untuk deployment
- [ ] **Web Interface** — Dashboard visualisasi hasil prediksi

---

## 📝 Catatan Penting

1. **Koordinat (x, y) vs (y, x):**
   - OpenCV dan JSON menggunakan format `(x, y)` — kolom dulu, baris kemudian
   - NumPy menggunakan format `(y, x)` — baris dulu, kolom kemudian
   - Konversi ini sudah ditangani secara otomatis di semua modul.

2. **Integritas Jumlah:**
   - Secara fundamental, Gaussian filter mempertahankan integral.
   - `sum(density_map)` harus selalu mendekati jumlah objek yang dianotasi, bahkan setelah proses rotasi atau skala, asalkan koreksi densitas yang tepat diaplikasikan.

---

<p align="center">
  <b>Capstone Project — Density Map Estimation</b><br>
  Built with 🐍 Python | 🔥 PyTorch | 👁 OpenCV
</p>
