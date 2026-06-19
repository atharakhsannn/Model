"""
density_utils.py — Shared Density Map Utilities

Modul utilitas bersama untuk generate density map dan visualisasi heatmap.
Digunakan oleh:
  - generate_ground_truth.py  (offline ground truth generation)
  - dataset_loader.py         (online density map generation saat training)
  - predict.py                (visualisasi hasil prediksi)

Algoritma: KDTree Adaptive Sigma
  - Sigma dihitung per-titik berdasarkan jarak ke tetangga terdekat (KDTree).
  - Pada area padat (titik berdekatan), sigma kecil → blob tajam → model bisa
    membedakan objek individual.
  - Pada area jarang, sigma besar → blob lebar → coverage lebih baik.
"""

import math
import numpy as np
import cv2
from scipy.spatial import KDTree


# ==============================
# Konstanta KDTree Adaptive Sigma
# ==============================
BETA = 0.3            # Faktor skala: sigma = BETA * avg_distance
MIN_SIGMA = 4.0       # Sigma minimum (mencegah blob terlalu kecil/noise)
MAX_SIGMA = 15.0      # Sigma maksimum (mencegah blob terlalu besar/melebur)
DEFAULT_SIGMA = 8.0   # Sigma default jika titik terlalu sedikit untuk KDTree

# Resolusi referensi (resolusi training asli)
REFERENCE_RESOLUTION = (672, 512)  # (width, height)


def generate_density_map(image_shape, points, base_sigma=DEFAULT_SIGMA,
                         reference_resolution=REFERENCE_RESOLUTION):
    """
    Generate density map dari list koordinat titik dengan KDTree adaptive sigma.

    Untuk setiap titik, sigma dihitung berdasarkan rata-rata jarak ke 3 tetangga
    terdekat (KDTree). Ini menghasilkan blob yang lebih kecil di area padat dan
    blob yang lebih besar di area jarang.

    Jika jumlah titik <= 3, sigma default digunakan (tidak cukup titik untuk KDTree).

    Parameters:
        image_shape (tuple): (height, width) dari gambar.
        points (list): List koordinat [[x1, y1], [x2, y2], ...].
        base_sigma (float): Sigma default jika titik terlalu sedikit.
        reference_resolution (tuple): (ref_w, ref_h) resolusi referensi.

    Returns:
        numpy.ndarray: Density map (float32) dengan sum ≈ jumlah titik.
    """
    h, w = image_shape
    density = np.zeros(image_shape, dtype=np.float32)
    num_points = len(points)

    if num_points == 0:
        return density

    points_array = np.array(points, dtype=np.float64)

    # --- Hitung sigma per-titik menggunakan KDTree ---
    if num_points > 3:
        tree = KDTree(points_array)
        # Ambil 4 tetangga terdekat (k=4).
        # Hasil pertama adalah titik itu sendiri (jarak 0), jadi ambil k=2,3,4 (indeks 1,2,3)
        distances, _ = tree.query(points_array, k=4)

        # Rata-rata jarak ke 3 titik terdekat
        avg_distances = np.mean(distances[:, 1:], axis=1)
        sigmas = BETA * avg_distances
        # Clipping sigma
        sigmas = np.clip(sigmas, MIN_SIGMA, MAX_SIGMA)
    else:
        sigmas = np.full(num_points, base_sigma, dtype=np.float32)

    # --- Letakkan Gaussian Kernel untuk setiap titik ---
    for i, point in enumerate(points):
        pt_x, pt_y = int(point[0]), int(point[1])
        sigma = sigmas[i]

        # Batas sebaran kernel (3*sigma mencakup ~99.7% area Gaussian)
        k_size = int(3 * sigma)

        # Buat grid untuk kernel secara independen
        y_grid, x_grid = np.ogrid[-k_size:k_size + 1, -k_size:k_size + 1]
        H = np.exp(-(x_grid ** 2 + y_grid ** 2) / (2 * sigma ** 2))

        H_sum = H.sum()
        if H_sum == 0:
            continue

        # Normalisasi kernel agar total nilainya = 1
        H = H / H_sum

        # Tentukan letak kernel pada gambar
        y1, y2 = pt_y - k_size, pt_y + k_size + 1
        x1, x2 = pt_x - k_size, pt_x + k_size + 1

        # Cek jika kernel keluar dari batas gambar, potong kernel-nya
        k_y1, k_y2 = 0, 2 * k_size + 1
        k_x1, k_x2 = 0, 2 * k_size + 1

        if y1 < 0:
            k_y1 = -y1
            y1 = 0
        if y2 > image_shape[0]:
            k_y2 -= (y2 - image_shape[0])
            y2 = image_shape[0]

        if x1 < 0:
            k_x1 = -x1
            x1 = 0
        if x2 > image_shape[1]:
            k_x2 -= (x2 - image_shape[1])
            x2 = image_shape[1]

        # Tambahkan nilai kernel ke area gambar yang valid
        if y1 < y2 and x1 < x2:
            density[y1:y2, x1:x2] += H[k_y1:k_y2, k_x1:k_x2]

    return density


def create_visualization(image, density_map, colormap=cv2.COLORMAP_JET,
                         alpha=0.5, input_is_rgb=False):
    """
    Buat overlay heatmap di atas gambar asli.

    Fungsi ini menangani baik input BGR (OpenCV default) maupun RGB
    (matplotlib/predict.py), sehingga satu fungsi bisa dipakai di semua script.

    Parameters:
        image (numpy.ndarray): Gambar asli.
        density_map (numpy.ndarray): Density map (float32).
        colormap (int): OpenCV colormap constant (default: cv2.COLORMAP_JET).
        alpha (float): Bobot blending gambar asli (0.0–1.0).
        input_is_rgb (bool): True jika gambar input dalam format RGB.
                             Output akan dikembalikan dalam format yang sama.

    Returns:
        numpy.ndarray: Gambar overlay dalam format yang sama dengan input.
    """
    # Normalisasi density map ke 0-255 untuk colormap
    if density_map.max() > 0:
        density_norm = (density_map / density_map.max() * 255).astype(np.uint8)
    else:
        density_norm = np.zeros_like(density_map, dtype=np.uint8)

    # Terapkan colormap (menghasilkan BGR)
    heatmap_bgr = cv2.applyColorMap(density_norm, colormap)

    if input_is_rgb:
        # Konversi heatmap ke RGB agar cocok dengan input
        heatmap = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    else:
        heatmap = heatmap_bgr

    # Overlay dengan alpha blending
    overlay = cv2.addWeighted(image, alpha, heatmap, 1.0 - alpha, 0)

    return overlay
