import os
import json
import numpy as np
import cv2

# Import dari shared module
from density_utils import generate_density_map, create_visualization


# ==============================
# Konfigurasi
# ==============================
IMAGES_DIR = os.path.join('dataset', 'images')
ANNOTATIONS_DIR = os.path.join('dataset', 'annotations')
GROUND_TRUTH_DIR = os.path.join('dataset', 'ground_truth')


def clean_existing_ground_truth():
    """
    Hapus semua file .npy dan _vis.jpg yang ada di GROUND_TRUTH_DIR.
    Meminta konfirmasi user terlebih dahulu.

    Returns:
        bool: True jika proses dilanjutkan, False jika user membatalkan.
    """
    if not os.path.exists(GROUND_TRUTH_DIR):
        return True

    existing_npy = [f for f in os.listdir(GROUND_TRUTH_DIR) if f.endswith('.npy')]
    existing_vis = [f for f in os.listdir(GROUND_TRUTH_DIR) if f.endswith('_vis.jpg')]

    if existing_npy:
        confirm = input(
            f"  Found {len(existing_npy)} existing .npy files. "
            f"Delete and regenerate? [y/N]: "
        )
        if confirm.lower() != 'y':
            print("  Aborted.")
            return False

        # Hapus semua file .npy dan _vis.jpg
        for f in existing_npy + existing_vis:
            filepath = os.path.join(GROUND_TRUTH_DIR, f)
            os.remove(filepath)
        print(f"  Deleted {len(existing_npy)} .npy and {len(existing_vis)} _vis.jpg files.")

    return True


def main():
    # Pastikan folder ground_truth ada
    os.makedirs(GROUND_TRUTH_DIR, exist_ok=True)

    # Ambil semua file JSON
    json_files = sorted([
        f for f in os.listdir(ANNOTATIONS_DIR)
        if f.lower().endswith('.json')
    ])

    if not json_files:
        print(f"Tidak ada file .json di folder '{ANNOTATIONS_DIR}'")
        print("Silakan buat anotasi terlebih dahulu menggunakan point_labeler.py")
        return

    print("\n" + "=" * 60)
    print("  GENERATE GROUND TRUTH - Density Map dari Anotasi")
    print("=" * 60)
    print(f"  Annotations : {ANNOTATIONS_DIR}")
    print(f"  Images      : {IMAGES_DIR}")
    print(f"  Output      : {GROUND_TRUTH_DIR}")
    print(f"  Algorithm   : KDTree Adaptive Sigma")
    print(f"  Total file  : {len(json_files)}")
    print("=" * 60)

    # --- Clean slate: hapus ground truth lama ---
    if not clean_existing_ground_truth():
        return

    success_count = 0
    error_count = 0

    for idx, json_file in enumerate(json_files):
        name_without_ext = os.path.splitext(json_file)[0]
        json_path = os.path.join(ANNOTATIONS_DIR, json_file)

        print(f"\n[{idx + 1}/{len(json_files)}] Memproses: {json_file}")

        # ---- 1. Baca file JSON ----
        with open(json_path, 'r') as f:
            data = json.load(f)

        image_filename = data.get('image', f"{name_without_ext}.png")
        points = data.get('points', [])
        num_points = len(points)

        print(f"  Gambar  : {image_filename}")
        print(f"  Jumlah titik : {num_points}")

        # ---- 2. Buka gambar untuk mendapatkan dimensi ----
        image_path = os.path.join(IMAGES_DIR, image_filename)

        if not os.path.exists(image_path):
            print(f"  [ERROR] Gambar tidak ditemukan: {image_path}")
            error_count += 1
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"  [ERROR] Gagal membaca gambar: {image_path}")
            error_count += 1
            continue

        h, w = image.shape[:2]
        print(f"  Dimensi : {w} x {h}")

        # ---- 3. Generate density map (KDTree adaptive sigma) ----
        density_map = generate_density_map((h, w), points)

        print(f"  Density map shape : {density_map.shape}")
        print(f"  Density map max   : {density_map.max():.8f}")
        print(f"  Density map sum   : {density_map.sum():.4f} "
              f"(idealnya ~ {num_points})")

        # ---- 4. Simpan density map sebagai .npy ----
        npy_path = os.path.join(GROUND_TRUTH_DIR, f"{name_without_ext}.npy")
        np.save(npy_path, density_map)
        print(f"  Tersimpan (.npy)  : {npy_path}")

        # ---- 5. Buat dan simpan visualisasi overlay ----
        vis_image = create_visualization(image, density_map)
        vis_path = os.path.join(GROUND_TRUTH_DIR, f"{name_without_ext}_vis.jpg")
        cv2.imwrite(vis_path, vis_image)
        print(f"  Tersimpan (vis)   : {vis_path}")

        success_count += 1

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  SELESAI!")
    print(f"  Berhasil : {success_count} file")
    if error_count > 0:
        print(f"  Gagal    : {error_count} file")
    print(f"  Output   : {GROUND_TRUTH_DIR}/")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
